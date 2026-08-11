#import "coreml_bridge.h"

#import <CoreML/CoreML.h>
#import <CoreML/MLModel+MLComputeDevice.h>
#import <CoreML/MLModel+MLModelCompilation.h>
#import <CoreML/MLNeuralEngineComputeDevice.h>
#import <Foundation/Foundation.h>
#include <stdatomic.h>

static const uint32_t VA_COREML_BRIDGE_DOMAIN = 0x434d4c42;
static const uint32_t VA_COREML_NSError_DOMAIN = 0x434d4c45;

@interface VAFeatureSpec : NSObject
@property(nonatomic) uint32_t slot;
@property(nonatomic) uint8_t role;
@property(nonatomic, copy) NSString *name;
@property(nonatomic, copy) NSArray<NSNumber *> *shape;
@property(nonatomic, copy) NSArray<NSNumber *> *strides;
@property(nonatomic) MLMultiArrayDataType dataType;
@property(nonatomic) uint64_t bytes;
@end

@implementation VAFeatureSpec
@end

@interface VAProgram : NSObject
@property(nonatomic, strong) MLModel *model;
@property(nonatomic, copy) NSArray<VAFeatureSpec *> *features;
@property(nonatomic, copy) NSSet<NSNumber *> *slots;
@property(nonatomic, strong, nullable) NSURL *temporaryCompiledURL;
@end

@implementation VAProgram
- (void)dealloc {
    if (_temporaryCompiledURL != nil) {
        [[NSFileManager defaultManager] removeItemAtURL:_temporaryCompiledURL error:nil];
    }
}
@end

struct VAEvent {
    _Atomic uint32_t references;
    _Atomic uint32_t status;
    _Atomic uint32_t error_kind;
    _Atomic uint32_t error_domain;
    _Atomic int64_t error_code;
};

static void va_set_error(struct va_coreml_error *error,
                         uint32_t kind,
                         uint32_t domain,
                         int64_t code) {
    if (error != NULL) {
        error->kind = kind;
        error->domain = domain;
        error->code = code;
    }
}

static uint32_t va_hash_domain(NSString *domain) {
    const char *bytes = domain.UTF8String;
    if (bytes == NULL) {
        return VA_COREML_NSError_DOMAIN;
    }
    uint32_t hash = 2166136261u;
    for (const unsigned char *cursor = (const unsigned char *)bytes; *cursor != 0; cursor++) {
        hash ^= *cursor;
        hash *= 16777619u;
    }
    return hash == 0 ? VA_COREML_NSError_DOMAIN : hash;
}

static void va_set_nserror(struct va_coreml_error *target, NSError *error) {
    if (error == nil) {
        va_set_error(target, VA_COREML_EXTERNAL, VA_COREML_NSError_DOMAIN, 0);
        return;
    }
    va_set_error(target, VA_COREML_EXTERNAL, va_hash_domain(error.domain), error.code);
}

static void va_event_release_inner(struct VAEvent *event) {
    if (event != NULL && atomic_fetch_sub_explicit(&event->references, 1, memory_order_acq_rel) == 1) {
        free(event);
    }
}

static struct VAEvent *va_event_create(struct va_coreml_error *error) {
    struct VAEvent *event = calloc(1, sizeof(struct VAEvent));
    if (event == NULL) {
        va_set_error(error, VA_COREML_OUT_OF_MEMORY, VA_COREML_BRIDGE_DOMAIN, 0);
        return NULL;
    }
    atomic_init(&event->references, 2);
    atomic_init(&event->status, VA_COREML_EVENT_PENDING);
    atomic_init(&event->error_kind, VA_COREML_OK);
    atomic_init(&event->error_domain, 0);
    atomic_init(&event->error_code, 0);
    return event;
}

int va_coreml_has_neural_engine(void) {
    @autoreleasepool {
        if (@available(macOS 14.0, *)) {
            for (id device in MLModel.availableComputeDevices) {
                if ([device isKindOfClass:MLNeuralEngineComputeDevice.class]) {
                    return 1;
                }
            }
        }
        return 0;
    }
}

static NSString *va_string(const uint8_t *bytes, size_t length) {
    if (bytes == NULL || length == 0 || length > NSIntegerMax) {
        return nil;
    }
    return [[NSString alloc] initWithBytes:bytes length:length encoding:NSUTF8StringEncoding];
}

static BOOL va_data_type_size(MLMultiArrayDataType type, uint64_t *size) {
    switch (type) {
    case MLMultiArrayDataTypeDouble:
        *size = 8;
        return YES;
    case MLMultiArrayDataTypeFloat32:
    case MLMultiArrayDataTypeInt32:
        *size = 4;
        return YES;
    case MLMultiArrayDataTypeFloat16:
        *size = 2;
        return YES;
    default:
        return NO;
    }
}

static VAFeatureSpec *va_feature_spec(MLFeatureDescription *description,
                                      uint32_t slot,
                                      uint8_t role,
                                      struct va_coreml_error *error) {
    if (description == nil || description.type != MLFeatureTypeMultiArray || description.optional) {
        va_set_error(error, VA_COREML_UNSUPPORTED, VA_COREML_BRIDGE_DOMAIN, 1);
        return nil;
    }
    MLMultiArrayConstraint *constraint = description.multiArrayConstraint;
    if (constraint == nil || constraint.shape.count == 0) {
        va_set_error(error, VA_COREML_UNSUPPORTED, VA_COREML_BRIDGE_DOMAIN, 2);
        return nil;
    }

    uint64_t elementSize = 0;
    if (!va_data_type_size(constraint.dataType, &elementSize)) {
        va_set_error(error, VA_COREML_UNSUPPORTED, VA_COREML_BRIDGE_DOMAIN, 3);
        return nil;
    }

    uint64_t elements = 1;
    NSMutableArray<NSNumber *> *strides = [NSMutableArray arrayWithCapacity:constraint.shape.count];
    for (NSNumber *dimension in constraint.shape.reverseObjectEnumerator) {
        NSInteger value = dimension.integerValue;
        if (value <= 0 || elements > UINT64_MAX / (uint64_t)value) {
            va_set_error(error, VA_COREML_RESOURCE_LIMIT, VA_COREML_BRIDGE_DOMAIN, 4);
            return nil;
        }
        [strides insertObject:@(elements) atIndex:0];
        elements *= (uint64_t)value;
    }
    if (elements > UINT64_MAX / elementSize) {
        va_set_error(error, VA_COREML_RESOURCE_LIMIT, VA_COREML_BRIDGE_DOMAIN, 5);
        return nil;
    }

    VAFeatureSpec *spec = [VAFeatureSpec new];
    spec.slot = slot;
    spec.role = role;
    spec.name = description.name;
    spec.shape = constraint.shape;
    spec.strides = strides;
    spec.dataType = constraint.dataType;
    spec.bytes = elements * elementSize;
    return spec;
}

static BOOL va_same_layout(VAFeatureSpec *left, VAFeatureSpec *right) {
    return left.bytes == right.bytes && left.dataType == right.dataType &&
           [left.shape isEqualToArray:right.shape] && [left.strides isEqualToArray:right.strides];
}

void *va_coreml_model_load(const uint8_t *path,
                           size_t path_len,
                           const struct va_coreml_feature_mapping *mappings,
                           size_t mapping_count,
                           struct va_coreml_error *error) {
    @autoreleasepool {
        va_set_error(error, VA_COREML_OK, 0, 0);
        if (mappings == NULL || mapping_count == 0) {
            va_set_error(error, VA_COREML_INVALID_ARGUMENT, VA_COREML_BRIDGE_DOMAIN, 6);
            return NULL;
        }
        NSString *pathString = va_string(path, path_len);
        if (pathString == nil) {
            va_set_error(error, VA_COREML_INVALID_ARGUMENT, VA_COREML_BRIDGE_DOMAIN, 7);
            return NULL;
        }

        @try {
            NSURL *sourceURL = [NSURL fileURLWithPath:pathString];
            NSURL *modelURL = sourceURL;
            NSURL *temporaryURL = nil;
            NSError *nativeError = nil;
            if ([sourceURL.pathExtension.lowercaseString isEqualToString:@"mlmodel"]) {
                modelURL = [MLModel compileModelAtURL:sourceURL error:&nativeError];
                if (modelURL == nil) {
                    va_set_nserror(error, nativeError);
                    return NULL;
                }
                temporaryURL = modelURL;
            } else if (![sourceURL.pathExtension.lowercaseString isEqualToString:@"mlmodelc"]) {
                va_set_error(error, VA_COREML_UNSUPPORTED, VA_COREML_BRIDGE_DOMAIN, 8);
                return NULL;
            }

            MLModelConfiguration *configuration = [MLModelConfiguration new];
            configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
            MLModel *model = [MLModel modelWithContentsOfURL:modelURL
                                              configuration:configuration
                                                      error:&nativeError];
            if (model == nil) {
                if (temporaryURL != nil) {
                    [[NSFileManager defaultManager] removeItemAtURL:temporaryURL error:nil];
                }
                va_set_nserror(error, nativeError);
                return NULL;
            }
            VAProgram *program = [VAProgram new];
            program.model = model;
            program.temporaryCompiledURL = temporaryURL;

            NSDictionary<NSString *, MLFeatureDescription *> *inputs =
                model.modelDescription.inputDescriptionsByName;
            NSDictionary<NSString *, MLFeatureDescription *> *outputs =
                model.modelDescription.outputDescriptionsByName;
            if (mapping_count != inputs.count + outputs.count) {
                va_set_error(error, VA_COREML_INCOMPATIBLE, VA_COREML_BRIDGE_DOMAIN, 9);
                return NULL;
            }

            NSMutableArray<VAFeatureSpec *> *features = [NSMutableArray arrayWithCapacity:mapping_count];
            NSMutableSet<NSString *> *mappedInputs = [NSMutableSet set];
            NSMutableSet<NSString *> *mappedOutputs = [NSMutableSet set];
            NSMutableSet<NSNumber *> *slots = [NSMutableSet set];
            for (size_t index = 0; index < mapping_count; index++) {
                const struct va_coreml_feature_mapping *mapping = &mappings[index];
                NSString *name = va_string(mapping->name, mapping->name_len);
                if (name == nil) {
                    va_set_error(error, VA_COREML_INVALID_ARGUMENT, VA_COREML_BRIDGE_DOMAIN, 10);
                    return NULL;
                }
                MLFeatureDescription *description = nil;
                NSMutableSet<NSString *> *mapped = nil;
                if (mapping->role == VA_COREML_INPUT) {
                    description = inputs[name];
                    mapped = mappedInputs;
                } else if (mapping->role == VA_COREML_OUTPUT) {
                    description = outputs[name];
                    mapped = mappedOutputs;
                } else {
                    va_set_error(error, VA_COREML_INVALID_ARGUMENT, VA_COREML_BRIDGE_DOMAIN, 11);
                    return NULL;
                }
                if (description == nil || [mapped containsObject:name]) {
                    va_set_error(error, VA_COREML_INCOMPATIBLE, VA_COREML_BRIDGE_DOMAIN, 12);
                    return NULL;
                }
                VAFeatureSpec *spec = va_feature_spec(description, mapping->slot, mapping->role, error);
                if (spec == nil) {
                    return NULL;
                }
                [mapped addObject:name];
                [slots addObject:@(mapping->slot)];
                [features addObject:spec];
            }
            if (mappedInputs.count != inputs.count || mappedOutputs.count != outputs.count) {
                va_set_error(error, VA_COREML_INCOMPATIBLE, VA_COREML_BRIDGE_DOMAIN, 13);
                return NULL;
            }
            for (VAFeatureSpec *left in features) {
                for (VAFeatureSpec *right in features) {
                    if (left != right && left.slot == right.slot && !va_same_layout(left, right)) {
                        va_set_error(error, VA_COREML_INCOMPATIBLE, VA_COREML_BRIDGE_DOMAIN, 14);
                        return NULL;
                    }
                }
            }

            program.features = features;
            program.slots = slots;
            return (__bridge_retained void *)program;
        } @catch (NSException *exception) {
            va_set_error(error, VA_COREML_EXTERNAL, VA_COREML_BRIDGE_DOMAIN, 15);
            return NULL;
        }
    }
}

void va_coreml_model_release(void *model) {
    if (model != NULL) {
        @autoreleasepool {
            CFBridgingRelease(model);
        }
    }
}

static const struct va_coreml_binding *va_binding_for_slot(
    const struct va_coreml_binding *bindings, size_t binding_count, uint32_t slot) {
    for (size_t index = 0; index < binding_count; index++) {
        if (bindings[index].slot == slot) {
            return &bindings[index];
        }
    }
    return NULL;
}

static uint8_t va_access_for_slot(NSArray<VAFeatureSpec *> *features, uint32_t slot) {
    BOOL reads = NO;
    BOOL writes = NO;
    for (VAFeatureSpec *feature in features) {
        if (feature.slot == slot) {
            reads |= feature.role == VA_COREML_INPUT;
            writes |= feature.role == VA_COREML_OUTPUT;
        }
    }
    if (reads && writes) {
        return VA_COREML_READ_WRITE;
    }
    return reads ? VA_COREML_READ : VA_COREML_WRITE;
}

void *va_coreml_submit(void *model,
                       const struct va_coreml_binding *bindings,
                       size_t binding_count,
                       void *context,
                       va_coreml_release_context_fn release_context,
                       struct va_coreml_error *error) {
    struct VAEvent *eventForException = NULL;
    @autoreleasepool {
        va_set_error(error, VA_COREML_OK, 0, 0);
        if (model == NULL || bindings == NULL || context == NULL || release_context == NULL) {
            va_set_error(error, VA_COREML_INVALID_ARGUMENT, VA_COREML_BRIDGE_DOMAIN, 16);
            return NULL;
        }
        VAProgram *program = (__bridge VAProgram *)model;
        if (binding_count != program.slots.count) {
            va_set_error(error, VA_COREML_INCOMPATIBLE, VA_COREML_BRIDGE_DOMAIN, 17);
            return NULL;
        }
        for (NSNumber *slotNumber in program.slots) {
            uint32_t slot = slotNumber.unsignedIntValue;
            const struct va_coreml_binding *binding =
                va_binding_for_slot(bindings, binding_count, slot);
            if (binding == NULL || binding->access != va_access_for_slot(program.features, slot)) {
                va_set_error(error, VA_COREML_INCOMPATIBLE, VA_COREML_BRIDGE_DOMAIN, 18);
                return NULL;
            }
        }

        @try {
            NSError *nativeError = nil;
            NSMutableDictionary<NSString *, MLFeatureValue *> *inputValues = [NSMutableDictionary dictionary];
            NSMutableDictionary<NSString *, MLMultiArray *> *outputBackings = [NSMutableDictionary dictionary];
            NSMutableArray<VAFeatureSpec *> *outputFeatures = [NSMutableArray array];
            for (VAFeatureSpec *feature in program.features) {
                const struct va_coreml_binding *binding =
                    va_binding_for_slot(bindings, binding_count, feature.slot);
                if (binding == NULL || binding->data == NULL || binding->bytes != feature.bytes) {
                    va_set_error(error, VA_COREML_INCOMPATIBLE, VA_COREML_BRIDGE_DOMAIN, 19);
                    return NULL;
                }
                MLMultiArray *array = [[MLMultiArray alloc] initWithDataPointer:binding->data
                                                                          shape:feature.shape
                                                                       dataType:feature.dataType
                                                                        strides:feature.strides
                                                                    deallocator:nil
                                                                          error:&nativeError];
                if (array == nil) {
                    va_set_nserror(error, nativeError);
                    return NULL;
                }
                if (feature.role == VA_COREML_INPUT) {
                    inputValues[feature.name] = [MLFeatureValue featureValueWithMultiArray:array];
                } else {
                    outputBackings[feature.name] = array;
                    [outputFeatures addObject:feature];
                }
            }

            MLDictionaryFeatureProvider *provider =
                [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputValues error:&nativeError];
            if (provider == nil) {
                va_set_nserror(error, nativeError);
                return NULL;
            }
            MLPredictionOptions *options = [MLPredictionOptions new];
            options.outputBackings = outputBackings;
            eventForException = va_event_create(error);
            if (eventForException == NULL) {
                return NULL;
            }
            struct VAEvent *event = eventForException;

            [program.model predictionFromFeatures:provider
                                          options:options
                                completionHandler:^(id<MLFeatureProvider> output, NSError *predictionError) {
                uint32_t terminalStatus = VA_COREML_EVENT_COMPLETE;
                @try {
                    if (predictionError != nil || output == nil) {
                        atomic_store_explicit(&event->error_kind,
                                              VA_COREML_EXTERNAL,
                                              memory_order_relaxed);
                        atomic_store_explicit(&event->error_domain,
                                              va_hash_domain(predictionError.domain),
                                              memory_order_relaxed);
                        atomic_store_explicit(&event->error_code,
                                              predictionError.code,
                                              memory_order_relaxed);
                        terminalStatus = VA_COREML_EVENT_FAILED;
                    } else {
                        for (VAFeatureSpec *feature in outputFeatures) {
                            MLMultiArray *actual =
                                [output featureValueForName:feature.name].multiArrayValue;
                            if (actual != outputBackings[feature.name]) {
                                atomic_store_explicit(&event->error_kind,
                                                      VA_COREML_INCOMPATIBLE,
                                                      memory_order_relaxed);
                                atomic_store_explicit(&event->error_domain,
                                                      VA_COREML_BRIDGE_DOMAIN,
                                                      memory_order_relaxed);
                                atomic_store_explicit(&event->error_code, 20, memory_order_relaxed);
                                terminalStatus = VA_COREML_EVENT_FAILED;
                                break;
                            }
                        }
                    }
                } @catch (NSException *exception) {
                    atomic_store_explicit(&event->error_kind,
                                          VA_COREML_EXTERNAL,
                                          memory_order_relaxed);
                    atomic_store_explicit(&event->error_domain,
                                          VA_COREML_BRIDGE_DOMAIN,
                                          memory_order_relaxed);
                    atomic_store_explicit(&event->error_code, 23, memory_order_relaxed);
                    terminalStatus = VA_COREML_EVENT_FAILED;
                }
                release_context(context);
                atomic_store_explicit(&event->status, terminalStatus, memory_order_release);
                va_event_release_inner(event);
            }];
            eventForException = NULL;
            return event;
        } @catch (NSException *exception) {
            if (eventForException != NULL) {
                va_event_release_inner(eventForException);
                va_event_release_inner(eventForException);
            }
            va_set_error(error, VA_COREML_EXTERNAL, VA_COREML_BRIDGE_DOMAIN, 21);
            return NULL;
        }
    }
}

uint32_t va_coreml_event_poll(void *opaque, struct va_coreml_error *error) {
    struct VAEvent *event = opaque;
    if (event == NULL) {
        va_set_error(error, VA_COREML_INVALID_ARGUMENT, VA_COREML_BRIDGE_DOMAIN, 22);
        return VA_COREML_EVENT_FAILED;
    }
    uint32_t status = atomic_load_explicit(&event->status, memory_order_acquire);
    if (status == VA_COREML_EVENT_FAILED) {
        va_set_error(error,
                     atomic_load_explicit(&event->error_kind, memory_order_relaxed),
                     atomic_load_explicit(&event->error_domain, memory_order_relaxed),
                     atomic_load_explicit(&event->error_code, memory_order_relaxed));
    } else {
        va_set_error(error, VA_COREML_OK, 0, 0);
    }
    return status;
}

void va_coreml_event_release(void *event) {
    va_event_release_inner(event);
}
