// Apple's anticipated compute plan, not a timer or resident-memory measurement.
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
static NSDictionary *usage(MLComputePlanDeviceUsage *u) {
    NSMutableArray *supported = [NSMutableArray array];
    for (id d in u.supportedComputeDevices) [supported addObject:NSStringFromClass([d class])];
    return @{@"preferred":u ? NSStringFromClass([u.preferredComputeDevice class]) : @"unknown", @"supported":supported};
}
static void operations(MLComputePlan *plan, MLModelStructureProgramBlock *block, NSMutableArray *rows) {
    for (MLModelStructureProgramOperation *op in block.operations) {
        [rows addObject:@{@"operator":op.operatorName, @"device_usage":usage([plan computeDeviceUsageForMLProgramOperation:op])}];
        for (MLModelStructureProgramBlock *child in op.blocks) operations(plan,child,rows);
    }
}
int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 4) { fprintf(stderr,"usage: inspect_coreml model.mlmodel cpu-gpu|cpu-ane|all report.json\n"); return 2; }
        NSString *policy = @(argv[2]);
        MLModelConfiguration *config = [MLModelConfiguration new];
        if ([policy isEqualToString:@"cpu-gpu"]) config.computeUnits = MLComputeUnitsCPUAndGPU;
        else if ([policy isEqualToString:@"cpu-ane"]) config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        else if ([policy isEqualToString:@"all"]) config.computeUnits = MLComputeUnitsAll;
        else { fprintf(stderr,"invalid policy\n"); return 2; }
        config.allowLowPrecisionAccumulationOnGPU = NO;
        MLOptimizationHints *hints = [MLOptimizationHints new];
        hints.reshapeFrequency = MLReshapeFrequencyHintInfrequent;
        if (@available(macOS 15.0, *)) hints.specializationStrategy = MLSpecializationStrategyFastPrediction;
        config.optimizationHints = hints;
        NSError *error = nil;
        NSURL *url = [MLModel compileModelAtURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
        if (!url) { fprintf(stderr,"%s\n",error.description.UTF8String); return 1; }
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        __block MLComputePlan *plan = nil;
        __block NSError *planError = nil;
        [MLComputePlan loadContentsOfURL:url configuration:config completionHandler:^(MLComputePlan *p, NSError *e) {
            plan=p; planError=e; dispatch_semaphore_signal(done);
        }];
        if (dispatch_semaphore_wait(done,dispatch_time(DISPATCH_TIME_NOW,60*NSEC_PER_SEC))) {
            fprintf(stderr,"compute plan timed out\n"); return 1;
        }
        if (!plan) {
            fprintf(stderr,"%s\n",planError.description.UTF8String);
            [NSFileManager.defaultManager removeItemAtURL:url error:nil]; return 1;
        }
        NSMutableArray *rows = [NSMutableArray array];
        for (MLModelStructureProgramFunction *f in plan.modelStructure.program.functions.allValues) operations(plan,f.block,rows);
        for (MLModelStructureNeuralNetworkLayer *layer in plan.modelStructure.neuralNetwork.layers) {
            [rows addObject:@{@"operator":layer.type,@"name":layer.name,@"device_usage":usage([plan computeDeviceUsageForNeuralNetworkLayer:layer])}];
        }
        NSDictionary *report = @{@"schema":@1,@"policy":policy,@"model":@(argv[1]),@"os":NSProcessInfo.processInfo.operatingSystemVersionString,
            @"evidence":@"Apple anticipated compute plan; not measured engine execution, kernel precision or resident memory",@"operations":rows};
        NSData *data = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&error];
        BOOL ok = data && [data writeToFile:@(argv[3]) options:NSDataWritingAtomic error:&error];
        [NSFileManager.defaultManager removeItemAtURL:url error:nil];
        if (!ok) fprintf(stderr,"%s\n",error.description.UTF8String);
        return ok ? 0 : 1;
    }
}
