#include "dsp_cuda_galaxy.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>
#include <chrono>

namespace
{
constexpr int kMBig = 2147483647;
constexpr int kMSeed = 161803398;

struct PoseGenSeedProfile
{
    unsigned long long phase1_cycles;
    unsigned long long phase2_cycles;
    unsigned int attempts;
    unsigned int collision_rejects;
    unsigned int sphere_rejects;
    unsigned int gate_skips;
};

struct BatchDeviceBuffers
{
    int device_id;
    int* d_seeds;
    int* d_counts;
    dsp_vec3d_t* d_poses;
    dsp_vec3d_t* d_drunk;
    dsp_vec3d_t* d_head_poses;
    PoseGenSeedProfile* d_gen_profiles;
    int* h_counts_pinned;
    dsp_vec3d_t* h_head_poses_pinned;
    cudaStream_t stream;
    size_t seeds_capacity_bytes;
    size_t counts_capacity_bytes;
    size_t poses_capacity_bytes;
    size_t drunk_capacity_bytes;
    size_t head_poses_capacity_bytes;
    size_t gen_profiles_capacity_bytes;
    size_t host_counts_capacity_bytes;
    size_t host_head_poses_capacity_bytes;

    BatchDeviceBuffers()
        : device_id(-2),
          d_seeds(nullptr),
          d_counts(nullptr),
          d_poses(nullptr),
          d_drunk(nullptr),
          d_head_poses(nullptr),
          d_gen_profiles(nullptr),
          h_counts_pinned(nullptr),
          h_head_poses_pinned(nullptr),
          stream(nullptr),
          seeds_capacity_bytes(0),
          counts_capacity_bytes(0),
          poses_capacity_bytes(0),
          drunk_capacity_bytes(0),
          head_poses_capacity_bytes(0),
          gen_profiles_capacity_bytes(0),
          host_counts_capacity_bytes(0),
          host_head_poses_capacity_bytes(0)
    {
    }
};

thread_local BatchDeviceBuffers g_batch_buffers;
thread_local dsp_cuda_pose_batch_head_timing_t g_last_pose_batch_head_timing =
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

struct PlanetCoreBatchDeviceBuffers
{
    int device_id;
    int* d_int_pool;
    float* d_float_pool;
    double* d_double_pool;
    int* d_info_seeds;
    int* d_orbit_arounds;
    int* d_orbit_indexes;
    int* d_gas_giants;
    int* d_star_indexes;
    int* d_galaxy_star_counts;
    int* d_galaxy_habitable_counts;
    int* d_boost_inclination_ns;
    int* d_compact_type_cases;
    float* d_star_orbit_scalers;
    double* d_star_masses;
    float* d_star_habitable_radiuses;
    float* d_star_light_balance_radiuses;
    float* d_orbit_around_planet_real_radiuses;
    float* d_orbit_around_planet_orbit_radiuses;
    double* d_orbit_around_planet_orbital_periods;
    dsp_planet_core_f32_out_t* d_out_results;
    size_t ints_capacity_bytes;
    size_t floats_capacity_bytes;
    size_t doubles_capacity_bytes;
    size_t out_capacity_bytes;

    PlanetCoreBatchDeviceBuffers()
        : device_id(-2),
          d_int_pool(nullptr),
          d_float_pool(nullptr),
          d_double_pool(nullptr),
          d_info_seeds(nullptr),
          d_orbit_arounds(nullptr),
          d_orbit_indexes(nullptr),
          d_gas_giants(nullptr),
          d_star_indexes(nullptr),
          d_galaxy_star_counts(nullptr),
          d_galaxy_habitable_counts(nullptr),
          d_boost_inclination_ns(nullptr),
          d_compact_type_cases(nullptr),
          d_star_orbit_scalers(nullptr),
          d_star_masses(nullptr),
          d_star_habitable_radiuses(nullptr),
          d_star_light_balance_radiuses(nullptr),
          d_orbit_around_planet_real_radiuses(nullptr),
          d_orbit_around_planet_orbit_radiuses(nullptr),
          d_orbit_around_planet_orbital_periods(nullptr),
          d_out_results(nullptr),
          ints_capacity_bytes(0),
          floats_capacity_bytes(0),
          doubles_capacity_bytes(0),
          out_capacity_bytes(0)
    {
    }
};

thread_local PlanetCoreBatchDeviceBuffers g_planet_core_batch_buffers;

void ReleaseBatchDeviceBuffers(BatchDeviceBuffers& buffers)
{
    if (buffers.stream != nullptr)
    {
        cudaStreamDestroy(buffers.stream);
        buffers.stream = nullptr;
    }
    if (buffers.d_head_poses != nullptr)
    {
        cudaFree(buffers.d_head_poses);
        buffers.d_head_poses = nullptr;
    }
    if (buffers.d_gen_profiles != nullptr)
    {
        cudaFree(buffers.d_gen_profiles);
        buffers.d_gen_profiles = nullptr;
    }
    if (buffers.d_drunk != nullptr)
    {
        cudaFree(buffers.d_drunk);
        buffers.d_drunk = nullptr;
    }
    if (buffers.d_poses != nullptr)
    {
        cudaFree(buffers.d_poses);
        buffers.d_poses = nullptr;
    }
    if (buffers.d_counts != nullptr)
    {
        cudaFree(buffers.d_counts);
        buffers.d_counts = nullptr;
    }
    if (buffers.d_seeds != nullptr)
    {
        cudaFree(buffers.d_seeds);
        buffers.d_seeds = nullptr;
    }
    if (buffers.h_head_poses_pinned != nullptr)
    {
        cudaFreeHost(buffers.h_head_poses_pinned);
        buffers.h_head_poses_pinned = nullptr;
    }
    if (buffers.h_counts_pinned != nullptr)
    {
        cudaFreeHost(buffers.h_counts_pinned);
        buffers.h_counts_pinned = nullptr;
    }
    buffers.seeds_capacity_bytes = 0;
    buffers.counts_capacity_bytes = 0;
    buffers.poses_capacity_bytes = 0;
    buffers.drunk_capacity_bytes = 0;
    buffers.head_poses_capacity_bytes = 0;
    buffers.gen_profiles_capacity_bytes = 0;
    buffers.host_counts_capacity_bytes = 0;
    buffers.host_head_poses_capacity_bytes = 0;
}

void ReleasePlanetCoreBatchDeviceBuffers(PlanetCoreBatchDeviceBuffers& buffers)
{
    cudaFree(buffers.d_out_results);
    cudaFree(buffers.d_double_pool);
    cudaFree(buffers.d_float_pool);
    cudaFree(buffers.d_int_pool);

    buffers.d_int_pool = nullptr;
    buffers.d_float_pool = nullptr;
    buffers.d_double_pool = nullptr;
    buffers.d_info_seeds = nullptr;
    buffers.d_orbit_arounds = nullptr;
    buffers.d_orbit_indexes = nullptr;
    buffers.d_gas_giants = nullptr;
    buffers.d_star_indexes = nullptr;
    buffers.d_galaxy_star_counts = nullptr;
    buffers.d_galaxy_habitable_counts = nullptr;
    buffers.d_boost_inclination_ns = nullptr;
    buffers.d_compact_type_cases = nullptr;
    buffers.d_star_orbit_scalers = nullptr;
    buffers.d_star_masses = nullptr;
    buffers.d_star_habitable_radiuses = nullptr;
    buffers.d_star_light_balance_radiuses = nullptr;
    buffers.d_orbit_around_planet_real_radiuses = nullptr;
    buffers.d_orbit_around_planet_orbit_radiuses = nullptr;
    buffers.d_orbit_around_planet_orbital_periods = nullptr;
    buffers.d_out_results = nullptr;
    buffers.ints_capacity_bytes = 0;
    buffers.floats_capacity_bytes = 0;
    buffers.doubles_capacity_bytes = 0;
    buffers.out_capacity_bytes = 0;
}

cudaError_t EnsureCudaBuffer(void** ptr, size_t* capacity_bytes, size_t required_bytes)
{
    if (*capacity_bytes >= required_bytes)
        return cudaSuccess;

    if (*ptr != nullptr)
    {
        cudaError_t free_rc = cudaFree(*ptr);
        if (free_rc != cudaSuccess)
            return free_rc;
        *ptr = nullptr;
    }

    cudaError_t alloc_rc = cudaMalloc(ptr, required_bytes);
    if (alloc_rc != cudaSuccess)
    {
        *capacity_bytes = 0;
        return alloc_rc;
    }

    *capacity_bytes = required_bytes;
    return cudaSuccess;
}

cudaError_t EnsurePinnedHostBuffer(void** ptr, size_t* capacity_bytes, size_t required_bytes)
{
    if (*capacity_bytes >= required_bytes)
        return cudaSuccess;

    if (*ptr != nullptr)
    {
        cudaError_t free_rc = cudaFreeHost(*ptr);
        if (free_rc != cudaSuccess)
            return free_rc;
        *ptr = nullptr;
    }

    cudaError_t alloc_rc = cudaMallocHost(ptr, required_bytes);
    if (alloc_rc != cudaSuccess)
    {
        *capacity_bytes = 0;
        return alloc_rc;
    }

    *capacity_bytes = required_bytes;
    return cudaSuccess;
}

cudaError_t EnsureBatchBuffers(
    int device_id,
    size_t seeds_bytes,
    size_t counts_bytes,
    size_t poses_bytes,
    size_t head_poses_bytes,
    size_t gen_profiles_bytes,
    int** d_seeds,
    int** d_counts,
    dsp_vec3d_t** d_poses,
    dsp_vec3d_t** d_drunk,
    dsp_vec3d_t** d_head_poses,
    PoseGenSeedProfile** d_gen_profiles,
    cudaStream_t* stream)
{
    BatchDeviceBuffers& buffers = g_batch_buffers;
    if (buffers.device_id != device_id)
    {
        ReleaseBatchDeviceBuffers(buffers);
        buffers.device_id = device_id;
    }
    if (buffers.stream == nullptr)
    {
        cudaError_t rc_stream = cudaStreamCreateWithFlags(&buffers.stream, cudaStreamNonBlocking);
        if (rc_stream != cudaSuccess)
            return rc_stream;
    }

    cudaError_t rc = EnsureCudaBuffer(reinterpret_cast<void**>(&buffers.d_seeds), &buffers.seeds_capacity_bytes, seeds_bytes);
    if (rc != cudaSuccess)
        return rc;
    rc = EnsureCudaBuffer(reinterpret_cast<void**>(&buffers.d_counts), &buffers.counts_capacity_bytes, counts_bytes);
    if (rc != cudaSuccess)
        return rc;
    rc = EnsureCudaBuffer(reinterpret_cast<void**>(&buffers.d_poses), &buffers.poses_capacity_bytes, poses_bytes);
    if (rc != cudaSuccess)
        return rc;
    rc = EnsureCudaBuffer(reinterpret_cast<void**>(&buffers.d_drunk), &buffers.drunk_capacity_bytes, poses_bytes);
    if (rc != cudaSuccess)
        return rc;
    if (head_poses_bytes > 0)
    {
        rc = EnsureCudaBuffer(reinterpret_cast<void**>(&buffers.d_head_poses), &buffers.head_poses_capacity_bytes, head_poses_bytes);
        if (rc != cudaSuccess)
            return rc;
    }
    if (gen_profiles_bytes > 0)
    {
        rc = EnsureCudaBuffer(reinterpret_cast<void**>(&buffers.d_gen_profiles), &buffers.gen_profiles_capacity_bytes, gen_profiles_bytes);
        if (rc != cudaSuccess)
            return rc;
    }

    *d_seeds = buffers.d_seeds;
    *d_counts = buffers.d_counts;
    *d_poses = buffers.d_poses;
    *d_drunk = buffers.d_drunk;
    if (d_head_poses != nullptr)
        *d_head_poses = buffers.d_head_poses;
    if (d_gen_profiles != nullptr)
        *d_gen_profiles = buffers.d_gen_profiles;
    if (stream != nullptr)
        *stream = buffers.stream;
    return cudaSuccess;
}

cudaError_t EnsurePlanetCoreBatchBuffers(
    int device_id,
    size_t ints_bytes,
    size_t floats_bytes,
    size_t doubles_bytes,
    size_t out_bytes,
    PlanetCoreBatchDeviceBuffers** out_buffers)
{
    PlanetCoreBatchDeviceBuffers& buffers = g_planet_core_batch_buffers;
    if (buffers.device_id != device_id)
    {
        ReleasePlanetCoreBatchDeviceBuffers(buffers);
        buffers.device_id = device_id;
    }

    const size_t ints_pool_bytes = ints_bytes * static_cast<size_t>(9);
    const size_t floats_pool_bytes = floats_bytes * static_cast<size_t>(5);
    const size_t doubles_pool_bytes = doubles_bytes * static_cast<size_t>(2);

    cudaError_t rc = EnsureCudaBuffer(reinterpret_cast<void**>(&buffers.d_int_pool), &buffers.ints_capacity_bytes, ints_pool_bytes);
    if (rc != cudaSuccess) return rc;
    rc = EnsureCudaBuffer(reinterpret_cast<void**>(&buffers.d_float_pool), &buffers.floats_capacity_bytes, floats_pool_bytes);
    if (rc != cudaSuccess) return rc;
    rc = EnsureCudaBuffer(reinterpret_cast<void**>(&buffers.d_double_pool), &buffers.doubles_capacity_bytes, doubles_pool_bytes);
    if (rc != cudaSuccess) return rc;
    rc = EnsureCudaBuffer(reinterpret_cast<void**>(&buffers.d_out_results), &buffers.out_capacity_bytes, out_bytes);
    if (rc != cudaSuccess) return rc;

    int* ip = buffers.d_int_pool;
    buffers.d_info_seeds = ip; ip += ints_bytes / sizeof(int);
    buffers.d_orbit_arounds = ip; ip += ints_bytes / sizeof(int);
    buffers.d_orbit_indexes = ip; ip += ints_bytes / sizeof(int);
    buffers.d_gas_giants = ip; ip += ints_bytes / sizeof(int);
    buffers.d_star_indexes = ip; ip += ints_bytes / sizeof(int);
    buffers.d_galaxy_star_counts = ip; ip += ints_bytes / sizeof(int);
    buffers.d_galaxy_habitable_counts = ip; ip += ints_bytes / sizeof(int);
    buffers.d_boost_inclination_ns = ip; ip += ints_bytes / sizeof(int);
    buffers.d_compact_type_cases = ip;

    float* fp = buffers.d_float_pool;
    buffers.d_star_orbit_scalers = fp; fp += floats_bytes / sizeof(float);
    buffers.d_star_habitable_radiuses = fp; fp += floats_bytes / sizeof(float);
    buffers.d_star_light_balance_radiuses = fp; fp += floats_bytes / sizeof(float);
    buffers.d_orbit_around_planet_real_radiuses = fp; fp += floats_bytes / sizeof(float);
    buffers.d_orbit_around_planet_orbit_radiuses = fp;

    double* dp = buffers.d_double_pool;
    buffers.d_star_masses = dp; dp += doubles_bytes / sizeof(double);
    buffers.d_orbit_around_planet_orbital_periods = dp;

    *out_buffers = &buffers;
    return cudaSuccess;
}

__device__ __forceinline__ int SubWrapI32(int a, int b)
{
    unsigned int ua = static_cast<unsigned int>(a);
    unsigned int ub = static_cast<unsigned int>(b);
    return static_cast<int>(ua - ub);
}

__device__ __forceinline__ int AddWrapI32(int a, int b)
{
    unsigned int ua = static_cast<unsigned int>(a);
    unsigned int ub = static_cast<unsigned int>(b);
    return static_cast<int>(ua + ub);
}

struct DotNet35RandomDevice
{
    int inext;
    int inextp;
    int seed_array[56];

    __device__ explicit DotNet35RandomDevice(int seed)
    {
        int subtraction = seed == static_cast<int>(0x80000000u) ? kMBig : (seed < 0 ? -seed : seed);
        int mj = SubWrapI32(kMSeed, subtraction);

        seed_array[55] = mj;
        int mk = 1;
        for (int i = 1; i < 55; ++i)
        {
            int ii = (21 * i) % 55;
            seed_array[ii] = mk;
            mk = SubWrapI32(mj, mk);
            if (mk < 0)
                mk = AddWrapI32(mk, kMBig);
            mj = seed_array[ii];
        }
        for (int k = 1; k < 5; ++k)
        {
            for (int i = 1; i < 56; ++i)
            {
                seed_array[i] = SubWrapI32(seed_array[i], seed_array[1 + (i + 30) % 55]);
                if (seed_array[i] < 0)
                    seed_array[i] = AddWrapI32(seed_array[i], kMBig);
            }
        }
        inext = 0;
        inextp = 31;
    }

    __device__ int InternalSample()
    {
        int loc_inext = inext + 1;
        if (loc_inext >= 56)
            loc_inext = 1;

        int loc_inextp = inextp + 1;
        if (loc_inextp >= 56)
            loc_inextp = 1;

        int ret = SubWrapI32(seed_array[loc_inext], seed_array[loc_inextp]);
        if (ret < 0)
            ret = AddWrapI32(ret, kMBig);

        seed_array[loc_inext] = ret;
        inext = loc_inext;
        inextp = loc_inextp;
        return ret;
    }

    __device__ double NextDouble()
    {
        return static_cast<double>(InternalSample()) * 4.6566128752457969e-10;
    }
};

__device__ __forceinline__ bool CheckCollisionF32Fast(
    const dsp_vec3d_t* pts,
    int count,
    const dsp_vec3d_t& pt,
    float min_dist_f,
    double min2_f32)
{
    for (int i = 0; i < count; ++i)
    {
        const dsp_vec3d_t& p = pts[i];
        float dx = static_cast<float>(pt.x - p.x);
        if (dx >= min_dist_f || dx <= -min_dist_f)
            continue;
        float dy = static_cast<float>(pt.y - p.y);
        if (dy >= min_dist_f || dy <= -min_dist_f)
            continue;
        float dz = static_cast<float>(pt.z - p.z);
        if (dz >= min_dist_f || dz <= -min_dist_f)
            continue;
        double dist2 = static_cast<double>(dx) * static_cast<double>(dx) +
                       static_cast<double>(dy) * static_cast<double>(dy) +
                       static_cast<double>(dz) * static_cast<double>(dz);
        if (dist2 < min2_f32)
            return true;
    }
    return false;
}

__device__ __forceinline__ bool CheckCollisionFp64Fast(
    const dsp_vec3d_t* pts,
    int count,
    const dsp_vec3d_t& pt,
    double min_dist,
    double min2_fp64)
{
    for (int i = 0; i < count; ++i)
    {
        const dsp_vec3d_t& p = pts[i];
        double dx = pt.x - p.x;
        if (dx >= min_dist || dx <= -min_dist)
            continue;
        double dy = pt.y - p.y;
        if (dy >= min_dist || dy <= -min_dist)
            continue;
        double dz = pt.z - p.z;
        if (dz >= min_dist || dz <= -min_dist)
            continue;
        if (dx * dx + dy * dy + dz * dz < min2_fp64)
            return true;
    }
    return false;
}

template <bool CollisionFp64>
__device__ __forceinline__ bool CheckCollisionFastT(
    const dsp_vec3d_t* pts,
    int count,
    const dsp_vec3d_t& pt,
    double min_dist,
    float min_dist_f,
    double min2_fp64,
    double min2_f32);

template <>
__device__ __forceinline__ bool CheckCollisionFastT<true>(
    const dsp_vec3d_t* pts,
    int count,
    const dsp_vec3d_t& pt,
    double min_dist,
    float min_dist_f,
    double min2_fp64,
    double min2_f32)
{
    (void)min_dist_f;
    (void)min2_f32;
    return CheckCollisionFp64Fast(pts, count, pt, min_dist, min2_fp64);
}

template <>
__device__ __forceinline__ bool CheckCollisionFastT<false>(
    const dsp_vec3d_t* pts,
    int count,
    const dsp_vec3d_t& pt,
    double min_dist,
    float min_dist_f,
    double min2_fp64,
    double min2_f32)
{
    (void)min_dist;
    (void)min2_fp64;
    return CheckCollisionF32Fast(pts, count, pt, min_dist_f, min2_f32);
}

template <bool CollectProfile, bool CollisionFp64>
__device__ __forceinline__ int GenerateRandomPosesParamsFp64Impl(
    int seed,
    int max_count,
    double min_dist,
    double min_step_len,
    double max_step_len,
    double flatten,
    dsp_vec3d_t* poses,
    dsp_vec3d_t* drunk,
    PoseGenSeedProfile* profile)
{
    DotNet35RandomDevice rng(seed);
    const float min_dist_f = static_cast<float>(min_dist);
    const double min2_f32 = static_cast<double>(min_dist_f) * static_cast<double>(min_dist_f);
    const double min2_fp64 = min_dist * min_dist;
    const double step_span = max_step_len - min_step_len;
    unsigned int attempts = 0;
    unsigned int collision_rejects = 0;
    unsigned int sphere_rejects = 0;
    unsigned int gate_skips = 0;
    unsigned long long phase1_cycles = 0;
    unsigned long long phase2_cycles = 0;
    const unsigned long long phase1_begin = CollectProfile ? clock64() : 0ULL;

    int poses_count = 0;
    int drunk_count = 0;

    poses[poses_count++] = dsp_vec3d_t{0.0, 0.0, 0.0};

    double num1 = rng.NextDouble();
    int num2 = 6;
    int num3 = 8;
    if (num2 < 1)
        num2 = 1;
    if (num3 < 1)
        num3 = 1;

    double num4 = static_cast<double>(num3 - num2);
    int num5 = static_cast<int>(num1 * num4 + static_cast<double>(num2));

    for (int i = 0; i < num5; ++i)
    {
        int tries = 0;
        while (tries++ < 256)
        {
            if (CollectProfile)
                ++attempts;
            double xd = rng.NextDouble() * 2.0 - 1.0;
            double yd = (rng.NextDouble() * 2.0 - 1.0) * flatten;
            double zd = rng.NextDouble() * 2.0 - 1.0;
            double num10 = rng.NextDouble();

            double dd = xd * xd + yd * yd + zd * zd;
            if (dd <= 1.0 && dd >= 1e-8)
            {
                double inv_len = 1.0 / sqrt(dd);
                double scale = (num10 * step_span + min_dist) * inv_len;
                dsp_vec3d_t pt{xd * scale, yd * scale, zd * scale};

                if (!CheckCollisionFastT<CollisionFp64>(poses, poses_count, pt, min_dist, min_dist_f, min2_fp64, min2_f32))
                {
                    drunk[drunk_count++] = pt;
                    poses[poses_count++] = pt;
                    if (poses_count >= max_count)
                    {
                        if (CollectProfile)
                        {
                            phase1_cycles = clock64() - phase1_begin;
                            profile->phase1_cycles = phase1_cycles;
                            profile->phase2_cycles = 0ULL;
                            profile->attempts = attempts;
                            profile->collision_rejects = collision_rejects;
                            profile->sphere_rejects = sphere_rejects;
                            profile->gate_skips = gate_skips;
                        }
                        return poses_count;
                    }
                    break;
                }
                if (CollectProfile)
                    ++collision_rejects;
            }
            else
            {
                if (CollectProfile)
                    ++sphere_rejects;
            }
        }
    }
    const unsigned long long phase2_begin = CollectProfile ? clock64() : 0ULL;
    if (CollectProfile)
        phase1_cycles = phase2_begin - phase1_begin;

    int outer = 0;
    while (outer++ < 256)
    {
        for (int i = 0; i < drunk_count; ++i)
        {
            if (rng.NextDouble() <= 0.7)
            {
                int tries = 0;
                while (tries++ < 256)
                {
                    if (CollectProfile)
                        ++attempts;
                    double xd = rng.NextDouble() * 2.0 - 1.0;
                    double yd = (rng.NextDouble() * 2.0 - 1.0) * flatten;
                    double zd = rng.NextDouble() * 2.0 - 1.0;
                    double num18 = rng.NextDouble();

                    double dd = xd * xd + yd * yd + zd * zd;
                    if (dd <= 1.0 && dd >= 1e-8)
                    {
                        double inv_len = 1.0 / sqrt(dd);
                        double scale = (num18 * step_span + min_dist) * inv_len;
                        const dsp_vec3d_t base_pt = drunk[i];
                        dsp_vec3d_t pt{
                            base_pt.x + xd * scale,
                            base_pt.y + yd * scale,
                            base_pt.z + zd * scale};

                        if (!CheckCollisionFastT<CollisionFp64>(poses, poses_count, pt, min_dist, min_dist_f, min2_fp64, min2_f32))
                        {
                            drunk[i] = pt;
                            poses[poses_count++] = pt;
                            if (poses_count >= max_count)
                            {
                                if (CollectProfile)
                                {
                                    phase2_cycles = clock64() - phase2_begin;
                                    profile->phase1_cycles = phase1_cycles;
                                    profile->phase2_cycles = phase2_cycles;
                                    profile->attempts = attempts;
                                    profile->collision_rejects = collision_rejects;
                                    profile->sphere_rejects = sphere_rejects;
                                    profile->gate_skips = gate_skips;
                                }
                                return poses_count;
                            }
                            break;
                        }
                        if (CollectProfile)
                            ++collision_rejects;
                    }
                    else
                    {
                        if (CollectProfile)
                            ++sphere_rejects;
                    }
                }
            }
            else
            {
                if (CollectProfile)
                    ++gate_skips;
            }
        }
    }

    if (CollectProfile)
    {
        phase2_cycles = clock64() - phase2_begin;
        profile->phase1_cycles = phase1_cycles;
        profile->phase2_cycles = phase2_cycles;
        profile->attempts = attempts;
        profile->collision_rejects = collision_rejects;
        profile->sphere_rejects = sphere_rejects;
        profile->gate_skips = gate_skips;
    }
    return poses_count;
}

