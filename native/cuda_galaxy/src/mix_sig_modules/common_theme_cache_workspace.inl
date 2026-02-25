#include "dsp_cuda_galaxy.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace
{
constexpr int kMBig = 2147483647;
constexpr int kMSeed = 161803398;
constexpr int kStarPlanMaxPlanets = 6;
constexpr unsigned long long kFnvOffset = 14695981039346656037ull;
constexpr unsigned long long kFnvPrime = 1099511628211ull;
inline int ResolveSigBlockSize();

inline unsigned long long HashBytesFnv64(unsigned long long h, const void* data, size_t bytes)
{
    if (data == nullptr || bytes == 0)
        return h;
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < bytes; ++i)
    {
        h ^= static_cast<unsigned long long>(p[i]);
        h *= kFnvPrime;
    }
    return h;
}

struct ThemeDeviceCacheEntry
{
    int device_id = -1;
    int theme_count = 0;
    int max_planet_type = 0;
    int theme_vein_total = 0;
    int theme_rare_total = 0;
    int theme_settings_total = 0;
    int type_theme_offsets_count = 0;
    int type_theme_values_count = 0;
    unsigned long long host_hash = 0;

    int* d_theme_ids = nullptr;
    int* d_theme_planet_types = nullptr;
    float* d_theme_temperatures = nullptr;
    int* d_theme_distributes = nullptr;
    int* d_theme_water_item_ids = nullptr;
    int* d_theme_vein_spot_offsets = nullptr;
    int* d_theme_vein_spot_values = nullptr;
    int* d_theme_rare_vein_offsets = nullptr;
    int* d_theme_rare_vein_values = nullptr;
    int* d_theme_rare_settings_offsets = nullptr;
    float* d_theme_rare_settings_values = nullptr;
    int* d_type_theme_offsets = nullptr;
    int* d_type_theme_values = nullptr;
};

struct ThemeDeviceCacheStore
{
    std::mutex mu;
    std::vector<ThemeDeviceCacheEntry> entries;
};

inline ThemeDeviceCacheStore& GetThemeDeviceCacheStore()
{
    static ThemeDeviceCacheStore store;
    return store;
}

inline void FreeThemeDeviceCacheEntryBuffers(ThemeDeviceCacheEntry& e)
{
    cudaFree(e.d_type_theme_values);
    cudaFree(e.d_type_theme_offsets);
    cudaFree(e.d_theme_rare_settings_values);
    cudaFree(e.d_theme_rare_settings_offsets);
    cudaFree(e.d_theme_rare_vein_values);
    cudaFree(e.d_theme_rare_vein_offsets);
    cudaFree(e.d_theme_vein_spot_values);
    cudaFree(e.d_theme_vein_spot_offsets);
    cudaFree(e.d_theme_water_item_ids);
    cudaFree(e.d_theme_distributes);
    cudaFree(e.d_theme_temperatures);
    cudaFree(e.d_theme_planet_types);
    cudaFree(e.d_theme_ids);
    e = ThemeDeviceCacheEntry{};
}

inline bool EnsureThemeDeviceCache(
    int device_id,
    int theme_count,
    int max_planet_type,
    const int* theme_ids,
    const int* theme_planet_types,
    const float* theme_temperatures,
    const int* theme_distributes,
    const int* theme_water_item_ids,
    const int* theme_vein_spot_offsets,
    const int* theme_vein_spot_values,
    const int* theme_rare_vein_offsets,
    const int* theme_rare_vein_values,
    const int* theme_rare_settings_offsets,
    const float* theme_rare_settings_values,
    const std::vector<int>& type_theme_offsets,
    const std::vector<int>& type_theme_values,
    const ThemeDeviceCacheEntry** out_cache)
{
    if (out_cache == nullptr || theme_count <= 0 || theme_ids == nullptr || theme_planet_types == nullptr ||
        theme_temperatures == nullptr || theme_distributes == nullptr || theme_water_item_ids == nullptr ||
        theme_vein_spot_offsets == nullptr || theme_rare_vein_offsets == nullptr || theme_rare_settings_offsets == nullptr)
        return false;

    const int theme_vein_total = theme_vein_spot_offsets[theme_count];
    const int theme_rare_total = theme_rare_vein_offsets[theme_count];
    const int theme_settings_total = theme_rare_settings_offsets[theme_count];
    if (theme_vein_total < 0 || theme_rare_total < 0 || theme_settings_total < 0)
        return false;

    unsigned long long host_hash = kFnvOffset;
    host_hash = HashBytesFnv64(host_hash, &theme_count, sizeof(theme_count));
    host_hash = HashBytesFnv64(host_hash, &max_planet_type, sizeof(max_planet_type));
    host_hash = HashBytesFnv64(host_hash, theme_ids, static_cast<size_t>(theme_count) * sizeof(int));
    host_hash = HashBytesFnv64(host_hash, theme_planet_types, static_cast<size_t>(theme_count) * sizeof(int));
    host_hash = HashBytesFnv64(host_hash, theme_temperatures, static_cast<size_t>(theme_count) * sizeof(float));
    host_hash = HashBytesFnv64(host_hash, theme_distributes, static_cast<size_t>(theme_count) * sizeof(int));
    host_hash = HashBytesFnv64(host_hash, theme_water_item_ids, static_cast<size_t>(theme_count) * sizeof(int));
    host_hash = HashBytesFnv64(host_hash, theme_vein_spot_offsets, static_cast<size_t>(theme_count + 1) * sizeof(int));
    host_hash = HashBytesFnv64(host_hash, theme_vein_spot_values, static_cast<size_t>(theme_vein_total) * sizeof(int));
    host_hash = HashBytesFnv64(host_hash, theme_rare_vein_offsets, static_cast<size_t>(theme_count + 1) * sizeof(int));
    host_hash = HashBytesFnv64(host_hash, theme_rare_vein_values, static_cast<size_t>(theme_rare_total) * sizeof(int));
    host_hash = HashBytesFnv64(host_hash, theme_rare_settings_offsets, static_cast<size_t>(theme_count + 1) * sizeof(int));
    host_hash = HashBytesFnv64(host_hash, theme_rare_settings_values, static_cast<size_t>(theme_settings_total) * sizeof(float));
    host_hash = HashBytesFnv64(host_hash, type_theme_offsets.data(), type_theme_offsets.size() * sizeof(int));
    host_hash = HashBytesFnv64(host_hash, type_theme_values.data(), type_theme_values.size() * sizeof(int));

    ThemeDeviceCacheStore& store = GetThemeDeviceCacheStore();
    std::lock_guard<std::mutex> lock(store.mu);
    for (const ThemeDeviceCacheEntry& e : store.entries)
    {
        if (e.device_id == device_id &&
            e.theme_count == theme_count &&
            e.max_planet_type == max_planet_type &&
            e.theme_vein_total == theme_vein_total &&
            e.theme_rare_total == theme_rare_total &&
            e.theme_settings_total == theme_settings_total &&
            e.type_theme_offsets_count == static_cast<int>(type_theme_offsets.size()) &&
            e.type_theme_values_count == static_cast<int>(type_theme_values.size()) &&
            e.host_hash == host_hash)
        {
            *out_cache = &e;
            return true;
        }
    }

    if (device_id >= 0 && cudaSetDevice(device_id) != cudaSuccess)
        return false;

    ThemeDeviceCacheEntry fresh{};
    fresh.device_id = device_id;
    fresh.theme_count = theme_count;
    fresh.max_planet_type = max_planet_type;
    fresh.theme_vein_total = theme_vein_total;
    fresh.theme_rare_total = theme_rare_total;
    fresh.theme_settings_total = theme_settings_total;
    fresh.type_theme_offsets_count = static_cast<int>(type_theme_offsets.size());
    fresh.type_theme_values_count = static_cast<int>(type_theme_values.size());
    fresh.host_hash = host_hash;

    auto alloc_int = [&](int** p, size_t count) -> bool {
        size_t bytes = count > 0 ? count * sizeof(int) : sizeof(int);
        return cudaMalloc(reinterpret_cast<void**>(p), bytes) == cudaSuccess;
    };
    auto alloc_float = [&](float** p, size_t count) -> bool {
        size_t bytes = count > 0 ? count * sizeof(float) : sizeof(float);
        return cudaMalloc(reinterpret_cast<void**>(p), bytes) == cudaSuccess;
    };
    auto h2d_raw = [&](void* dst, const void* src, size_t bytes) -> bool {
        if (bytes == 0)
            return true;
        return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    };

    if (!alloc_int(&fresh.d_theme_ids, static_cast<size_t>(theme_count)) ||
        !alloc_int(&fresh.d_theme_planet_types, static_cast<size_t>(theme_count)) ||
        !alloc_float(&fresh.d_theme_temperatures, static_cast<size_t>(theme_count)) ||
        !alloc_int(&fresh.d_theme_distributes, static_cast<size_t>(theme_count)) ||
        !alloc_int(&fresh.d_theme_water_item_ids, static_cast<size_t>(theme_count)) ||
        !alloc_int(&fresh.d_theme_vein_spot_offsets, static_cast<size_t>(theme_count + 1)) ||
        !alloc_int(&fresh.d_theme_vein_spot_values, static_cast<size_t>(theme_vein_total)) ||
        !alloc_int(&fresh.d_theme_rare_vein_offsets, static_cast<size_t>(theme_count + 1)) ||
        !alloc_int(&fresh.d_theme_rare_vein_values, static_cast<size_t>(theme_rare_total)) ||
        !alloc_int(&fresh.d_theme_rare_settings_offsets, static_cast<size_t>(theme_count + 1)) ||
        !alloc_float(&fresh.d_theme_rare_settings_values, static_cast<size_t>(theme_settings_total)) ||
        !alloc_int(&fresh.d_type_theme_offsets, type_theme_offsets.size()) ||
        !alloc_int(&fresh.d_type_theme_values, type_theme_values.size()) ||
        !h2d_raw(fresh.d_theme_ids, theme_ids, static_cast<size_t>(theme_count) * sizeof(int)) ||
        !h2d_raw(fresh.d_theme_planet_types, theme_planet_types, static_cast<size_t>(theme_count) * sizeof(int)) ||
        !h2d_raw(fresh.d_theme_temperatures, theme_temperatures, static_cast<size_t>(theme_count) * sizeof(float)) ||
        !h2d_raw(fresh.d_theme_distributes, theme_distributes, static_cast<size_t>(theme_count) * sizeof(int)) ||
        !h2d_raw(fresh.d_theme_water_item_ids, theme_water_item_ids, static_cast<size_t>(theme_count) * sizeof(int)) ||
        !h2d_raw(fresh.d_theme_vein_spot_offsets, theme_vein_spot_offsets, static_cast<size_t>(theme_count + 1) * sizeof(int)) ||
        !h2d_raw(fresh.d_theme_vein_spot_values, theme_vein_spot_values, static_cast<size_t>(theme_vein_total) * sizeof(int)) ||
        !h2d_raw(fresh.d_theme_rare_vein_offsets, theme_rare_vein_offsets, static_cast<size_t>(theme_count + 1) * sizeof(int)) ||
        !h2d_raw(fresh.d_theme_rare_vein_values, theme_rare_vein_values, static_cast<size_t>(theme_rare_total) * sizeof(int)) ||
        !h2d_raw(fresh.d_theme_rare_settings_offsets, theme_rare_settings_offsets, static_cast<size_t>(theme_count + 1) * sizeof(int)) ||
        !h2d_raw(fresh.d_theme_rare_settings_values, theme_rare_settings_values, static_cast<size_t>(theme_settings_total) * sizeof(float)) ||
        !h2d_raw(fresh.d_type_theme_offsets, type_theme_offsets.data(), type_theme_offsets.size() * sizeof(int)) ||
        !h2d_raw(fresh.d_type_theme_values, type_theme_values.data(), type_theme_values.size() * sizeof(int)))
    {
        FreeThemeDeviceCacheEntryBuffers(fresh);
        return false;
    }

    store.entries.push_back(fresh);
    *out_cache = &store.entries.back();
    return true;
}

inline bool AcquireSigThreadStream(int device_id, cudaStream_t* out_stream)
{
    if (out_stream == nullptr)
        return false;
    struct SigThreadStream
    {
        int device_id = -1;
        cudaStream_t stream = nullptr;
        ~SigThreadStream()
        {
            if (stream != nullptr)
            {
                if (device_id >= 0)
                    cudaSetDevice(device_id);
                cudaStreamDestroy(stream);
                stream = nullptr;
            }
        }
    };
    thread_local SigThreadStream tls;

    int resolved_device = device_id;
    if (resolved_device < 0)
    {
        if (cudaGetDevice(&resolved_device) != cudaSuccess)
            resolved_device = 0;
    }
    if (tls.stream == nullptr || tls.device_id != resolved_device)
    {
        if (tls.stream != nullptr)
        {
            if (tls.device_id >= 0)
                cudaSetDevice(tls.device_id);
            cudaStreamDestroy(tls.stream);
            tls.stream = nullptr;
        }
        if (resolved_device >= 0 && cudaSetDevice(resolved_device) != cudaSuccess)
            return false;
        if (cudaStreamCreateWithFlags(&tls.stream, cudaStreamNonBlocking) != cudaSuccess)
            return false;
        tls.device_id = resolved_device;
    }
    *out_stream = tls.stream;
    return true;
}

enum class DirectSigBufferId : int
{
    GalaxySeeds = 0,
    SeedStarCounts,
    SeedStarOffsets,
    PoseRaw,
    StarIds,
    StarSeeds,
    StarTypes,
    StarSpectrs,
    StarIndexes,
    StarCountsInGalaxy,
    StarPosQx,
    StarPosQy,
    StarPosQz,
    StarOrbitScalers,
    StarMasses,
    StarHabitableRadiuses,
    StarLightBalanceRadiuses,
    OutPlanetCounts,
    DebugPlanOrbitArounds,
    DebugPlanOrbitIndexes,
    DebugPlanNumbers,
    DebugPlanGasGiants,
    DebugPlanInfoSeeds,
    DebugPlanGenSeeds,
    DebugCoreFlat,
    StarPlanetOffsets,
    PlanetIds,
    PlanetIndexes,
    PlanetOrbitIndexes,
    PlanetOrbitArounds,
    PlanetGenSeeds,
    PlanetGasGiants,
    PlanetCoreHabitableBias,
    PlanetCoreSunDistance,
    PlanetCoreTemperatureBias,
    PlanetCoreNum13,
    PlanetCoreNum14,
    PlanetCoreRand1,
    TotalPlanetsDev,
    OutGalaxySigs,
    OutPlanetSigs,
    OutVeinSigs,
    OutPipelineSigs,
    Count
};

struct DirectSigWorkspace
{
    int device_id = -1;
    cudaStream_t stream = nullptr;
    void* ptrs[static_cast<int>(DirectSigBufferId::Count)]{};
    size_t bytes[static_cast<int>(DirectSigBufferId::Count)]{};

    ~DirectSigWorkspace()
    {
        Reset();
    }

    void Reset()
    {
        if (device_id >= 0)
            cudaSetDevice(device_id);
        for (int i = 0; i < static_cast<int>(DirectSigBufferId::Count); ++i)
        {
            if (ptrs[i] != nullptr)
            {
                cudaFree(ptrs[i]);
                ptrs[i] = nullptr;
            }
            bytes[i] = 0;
        }
        stream = nullptr;
    }

    bool Prepare(int requested_device, cudaStream_t requested_stream)
    {
        int resolved_device = requested_device;
        if (resolved_device < 0)
        {
            if (cudaGetDevice(&resolved_device) != cudaSuccess)
                resolved_device = 0;
        }
        if (device_id != resolved_device)
        {
            Reset();
            device_id = resolved_device;
        }
        stream = requested_stream;
        return true;
    }

    bool Ensure(DirectSigBufferId id, size_t need_bytes, void** out_ptr)
    {
        if (out_ptr == nullptr)
            return false;
        if (need_bytes == 0)
            need_bytes = 1;
        const int slot = static_cast<int>(id);
        if (ptrs[slot] != nullptr && bytes[slot] >= need_bytes)
        {
            *out_ptr = ptrs[slot];
            return true;
        }
        if (device_id >= 0 && cudaSetDevice(device_id) != cudaSuccess)
            return false;
        if (ptrs[slot] != nullptr)
        {
            cudaFree(ptrs[slot]);
            ptrs[slot] = nullptr;
            bytes[slot] = 0;
        }
        if (cudaMalloc(&ptrs[slot], need_bytes) != cudaSuccess)
            return false;
        bytes[slot] = need_bytes;
        *out_ptr = ptrs[slot];
        return true;
    }
};

inline DirectSigWorkspace& AcquireDirectSigWorkspace(int device_id, cudaStream_t stream)
{
    thread_local DirectSigWorkspace workspace;
    workspace.Prepare(device_id, stream);
    return workspace;
}

inline int AddWrapI32(int a, int b)
{
    unsigned int ua = static_cast<unsigned int>(a);
    unsigned int ub = static_cast<unsigned int>(b);
    return static_cast<int>(ua + ub);
}

inline int SubWrapI32(int a, int b)
{
    unsigned int ua = static_cast<unsigned int>(a);
    unsigned int ub = static_cast<unsigned int>(b);
    return static_cast<int>(ua - ub);
}

