// SPDX-License-Identifier: MIT OR Apache-2.0

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winsvc.h>

#include <algorithm>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <domain.h>
#include <remote.h>
#include <rpcmem.h>

#include "host_bridge.h"
#include "va_htp.h"

namespace {
constexpr int kCdspDomain = 3;
constexpr uint32_t kControlBytes = 64 * 1024;

using RpcmemAlloc2 = void *(*)(int, uint32_t, size_t);
using RpcmemAlloc = void *(*)(int, uint32_t, int);
using RpcmemFree = void (*)(void *);
using RpcmemToFd = int (*)(void *);
using FastrpcMmap = int (*)(int, int, void *, int, size_t, enum fastrpc_map_flags);
using FastrpcMunmap = int (*)(int, int, void *, size_t);
using RemoteOpen = int (*)(const char *, remote_handle64 *);
using RemoteInvoke = int (*)(remote_handle64, uint32_t, remote_arg *);
using RemoteClose = int (*)(remote_handle64);
using RemoteControl = int (*)(uint32_t, void *, uint32_t);
using RemoteSessionControl = int (*)(uint32_t, void *, uint32_t);

struct Driver {
    HMODULE module = nullptr;
    RpcmemAlloc2 rpcmem_alloc2 = nullptr;
    RpcmemAlloc rpcmem_alloc = nullptr;
    RpcmemFree rpcmem_free = nullptr;
    RpcmemToFd rpcmem_to_fd = nullptr;
    FastrpcMmap fastrpc_mmap = nullptr;
    FastrpcMunmap fastrpc_munmap = nullptr;
    RemoteOpen remote_open = nullptr;
    RemoteInvoke remote_invoke = nullptr;
    RemoteClose remote_close = nullptr;
    RemoteControl remote_control = nullptr;
    RemoteSessionControl remote_session_control = nullptr;
};

Driver g_driver;

void message(char *out, size_t bytes, const char *format, ...) {
    if (!out || bytes == 0) return;
    va_list args;
    va_start(args, format);
    vsnprintf(out, bytes, format, args);
    va_end(args);
    out[bytes - 1] = '\0';
}

std::wstring driver_directory() {
    SC_HANDLE manager = OpenSCManagerW(nullptr, nullptr, STANDARD_RIGHTS_READ);
    if (!manager) return {};
    SC_HANDLE service = OpenServiceW(manager, L"qcnspmcdm", SERVICE_QUERY_CONFIG);
    if (!service) {
        CloseServiceHandle(manager);
        return {};
    }
    DWORD bytes = 0;
    QueryServiceConfigW(service, nullptr, 0, &bytes);
    std::vector<unsigned char> storage(bytes);
    auto *config = reinterpret_cast<QUERY_SERVICE_CONFIGW *>(storage.data());
    if (!QueryServiceConfigW(service, config, bytes, &bytes)) {
        CloseServiceHandle(service);
        CloseServiceHandle(manager);
        return {};
    }
    std::wstring path(config->lpBinaryPathName);
    CloseServiceHandle(service);
    CloseServiceHandle(manager);
    auto slash = path.find_last_of(L"\\/");
    if (slash == std::wstring::npos) return {};
    path.resize(slash);
    constexpr wchar_t prefix[] = L"\\SystemRoot";
    if (path.rfind(prefix, 0) == 0) {
        wchar_t windows[MAX_PATH] = {};
        UINT length = GetWindowsDirectoryW(windows, MAX_PATH);
        if (!length || length >= MAX_PATH) return {};
        path.replace(0, std::size(prefix) - 1, windows);
    }
    return path;
}

template <typename T>
bool symbol(HMODULE module, const char *name, T *out) {
    *out = reinterpret_cast<T>(GetProcAddress(module, name));
    return *out != nullptr;
}

bool load_driver(char *out, size_t out_bytes) {
    if (g_driver.module) return true;
    auto directory = driver_directory();
    if (directory.empty()) {
        message(out, out_bytes, "Qualcomm qcnspmcdm driver service was not found");
        return false;
    }
    auto path = directory + L"\\libcdsprpc.dll";
    HMODULE module = LoadLibraryW(path.c_str());
    if (!module) {
        message(out, out_bytes, "LoadLibrary(libcdsprpc.dll) failed: %lu", GetLastError());
        return false;
    }
    bool ok = symbol(module, "rpcmem_alloc", &g_driver.rpcmem_alloc) &&
        symbol(module, "rpcmem_free", &g_driver.rpcmem_free) &&
        symbol(module, "rpcmem_to_fd", &g_driver.rpcmem_to_fd) &&
        symbol(module, "fastrpc_mmap", &g_driver.fastrpc_mmap) &&
        symbol(module, "fastrpc_munmap", &g_driver.fastrpc_munmap) &&
        symbol(module, "remote_handle64_open", &g_driver.remote_open) &&
        symbol(module, "remote_handle64_invoke", &g_driver.remote_invoke) &&
        symbol(module, "remote_handle64_close", &g_driver.remote_close) &&
        symbol(module, "remote_handle_control", &g_driver.remote_control) &&
        symbol(module, "remote_session_control", &g_driver.remote_session_control);
    symbol(module, "rpcmem_alloc2", &g_driver.rpcmem_alloc2);
    if (!ok) {
        message(out, out_bytes, "libcdsprpc.dll is missing a required FastRPC entry point");
        FreeLibrary(module);
        g_driver = {};
        return false;
    }
    g_driver.module = module;
    return true;
}

struct Allocation {
    void *address;
    uint32_t bytes;
};
}

struct VaHtpRuntime {
    remote_handle64 handle = 0;
    unsigned char *arena = nullptr;
    uint32_t arena_bytes = 0;
    int fd = -1;
    bool host_mapped = false;
    bool dsp_mapped = false;
    std::vector<Allocation> allocations;
};

extern "C" int remote_handle64_open(const char *name, remote_handle64 *handle) {
    return g_driver.remote_open(name, handle);
}
extern "C" int remote_handle64_invoke(remote_handle64 handle, uint32_t scalars, remote_arg *args) {
    return g_driver.remote_invoke(handle, scalars, args);
}
extern "C" int remote_handle64_close(remote_handle64 handle) {
    return g_driver.remote_close(handle);
}

extern "C" uint64_t va_htp_runtime_create(
    const char *module_directory,
    uint32_t arena_bytes,
    VaHtpRuntime **runtime,
    VaHtpRuntimeInfo *info,
    char *out,
    size_t out_bytes) {
    if (!module_directory || !runtime || !info || arena_bytes <= kControlBytes) {
        return VA_HTP_ERROR_INVALID_ARGUMENT;
    }
    *runtime = nullptr;
    // The Windows FastRPC DLL snapshots its module search path while loading.
    // Set it before resolving the driver, not immediately before open().
    SetEnvironmentVariableA("ADSP_LIBRARY_PATH", module_directory);
    if (!load_driver(out, out_bytes)) return VA_HTP_ERROR_UNAVAILABLE;
    remote_dsp_capability capability = {};
    capability.domain = kCdspDomain;
    capability.attribute_ID = ARCH_VER;
    if (g_driver.remote_control(DSPRPC_GET_DSP_INFO, &capability, sizeof(capability)) != 0 ||
        (capability.capability & 0xffu) != 0x73u) {
        message(out, out_bytes, "direct HTP requires V73, driver reported 0x%x", capability.capability);
        return VA_HTP_ERROR_INCOMPATIBLE;
    }
    remote_rpc_control_unsigned_module unsigned_module = {};
    unsigned_module.domain = kCdspDomain;
    unsigned_module.enable = 1;
    if (g_driver.remote_session_control(
            DSPRPC_CONTROL_UNSIGNED_MODULE, &unsigned_module, sizeof(unsigned_module)) != 0) {
        message(out, out_bytes, "driver refused the HTP module policy request");
        return VA_HTP_ERROR_UNAVAILABLE;
    }

    std::unique_ptr<VaHtpRuntime> result(new (std::nothrow) VaHtpRuntime());
    if (!result) return VA_HTP_ERROR_OUT_OF_MEMORY;
    const char *uri = "file:///libvirtio-accel-htp-v73.so?va_htp_skel_handle_invoke&_modver=1.0&_dom=cdsp";
    int error = va_htp_open(uri, &result->handle);
    if (error != 0) {
        message(out, out_bytes, "FastRPC could not load the signed V73 skel: 0x%x", error);
        return VA_HTP_ERROR_UNAVAILABLE;
    }
    result->arena_bytes = arena_bytes;
    uint32_t arch = 0, hvx = 0, vtcm = 0;
    if (va_htp_hwinfo(result->handle, &arch, &hvx, &vtcm) != 0 || arch != 73) {
        va_htp_runtime_free(result.release());
        return VA_HTP_ERROR_INCOMPATIBLE;
    }
    *info = {arch, hvx, vtcm, arena_bytes};
    *runtime = result.release();
    message(out, out_bytes, "V73 direct HTP ready");
    return VA_HTP_SUCCESS;
}

extern "C" uint64_t va_htp_runtime_free(VaHtpRuntime *runtime) {
    if (!runtime) return VA_HTP_SUCCESS;
    uint64_t result = VA_HTP_SUCCESS;
    for (const Allocation allocation : runtime->allocations) {
        g_driver.rpcmem_free(allocation.address);
    }
    if (runtime->handle && va_htp_close(runtime->handle) != 0) result = VA_HTP_ERROR_DEVICE_LOST;
    delete runtime;
    return result;
}

extern "C" uint64_t va_htp_buffer_alloc(
    VaHtpRuntime *runtime,
    uint32_t bytes,
    uint32_t alignment,
    uint32_t *offset,
    void **address) {
    if (!runtime || !bytes || !alignment || (alignment & (alignment - 1)) || !offset || !address) {
        return VA_HTP_ERROR_INVALID_ARGUMENT;
    }
    void *allocation = g_driver.rpcmem_alloc2
        ? g_driver.rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes)
        : g_driver.rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, static_cast<int>(bytes));
    if (!allocation) return VA_HTP_ERROR_OUT_OF_MEMORY;
    if ((reinterpret_cast<uintptr_t>(allocation) & (alignment - 1)) != 0) {
        g_driver.rpcmem_free(allocation);
        return VA_HTP_ERROR_INCOMPATIBLE;
    }
    runtime->allocations.push_back({allocation, bytes});
    *offset = 0;
    *address = allocation;
    return VA_HTP_SUCCESS;
}

extern "C" uint64_t va_htp_buffer_free(VaHtpRuntime *runtime, void *address, uint32_t bytes) {
    if (!runtime || !address || !bytes) return VA_HTP_ERROR_INVALID_ARGUMENT;
    auto allocation = std::find_if(runtime->allocations.begin(), runtime->allocations.end(),
        [address, bytes](const Allocation &item) { return item.address == address && item.bytes == bytes; });
    if (allocation == runtime->allocations.end()) return VA_HTP_ERROR_INVALID_ARGUMENT;
    g_driver.rpcmem_free(address);
    runtime->allocations.erase(allocation);
    return VA_HTP_SUCCESS;
}

extern "C" uint64_t va_htp_execute_direct(
    VaHtpRuntime *runtime,
    uint32_t opcode,
    uint32_t lanes,
    const void *parameters,
    uint32_t parameter_bytes,
    const VaHtpBinding *bindings,
    uint32_t binding_count,
    uint64_t *elapsed_cycles) {
    if (!runtime || !bindings || binding_count < 2 || binding_count > 3 || !elapsed_cycles) {
        return VA_HTP_ERROR_INVALID_ARGUMENT;
    }
    if (parameter_bytes && !parameters) return VA_HTP_ERROR_INVALID_ARGUMENT;
    const VaHtpBinding &input0 = bindings[0];
    const VaHtpBinding &input1 = binding_count == 3 ? bindings[1] : bindings[0];
    const VaHtpBinding &output = bindings[binding_count - 1];
    auto valid = [runtime](const VaHtpBinding &binding) {
        return binding.address && std::any_of(runtime->allocations.begin(), runtime->allocations.end(),
            [&binding](const Allocation &item) { return item.address == binding.address && item.bytes >= binding.bytes; });
    };
    if (!valid(input0) || !valid(input1) || !valid(output) ||
        parameter_bytes > static_cast<uint32_t>(INT_MAX) || input0.bytes > static_cast<uint32_t>(INT_MAX) ||
        input1.bytes > static_cast<uint32_t>(INT_MAX) || output.bytes > static_cast<uint32_t>(INT_MAX)) {
        return VA_HTP_ERROR_INVALID_ARGUMENT;
    }
    const unsigned char *parameter_data = parameter_bytes
        ? static_cast<const unsigned char *>(parameters)
        : static_cast<const unsigned char *>(input0.address);
    int error = va_htp_execute(runtime->handle, opcode, lanes,
        parameter_data, static_cast<int>(parameter_bytes),
        static_cast<const unsigned char *>(input0.address), static_cast<int>(input0.bytes),
        static_cast<const unsigned char *>(input1.address), static_cast<int>(input1.bytes),
        static_cast<unsigned char *>(output.address), static_cast<int>(output.bytes), elapsed_cycles);
    if (error != 0) fprintf(stderr, "direct HTP execute failed: 0x%x\n", error);
    return error == 0 ? VA_HTP_SUCCESS : VA_HTP_ERROR_DEVICE_LOST;
}
