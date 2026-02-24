#include "dsp_cuda_galaxy.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
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
    cudaStream_t stream;
    size_t seeds_capacity_bytes;
    size_t counts_capacity_bytes;
    size_t poses_capacity_bytes;
    size_t drunk_capacity_bytes;
    size_t head_poses_capacity_bytes;
    size_t gen_profiles_capacity_bytes;

    BatchDeviceBuffers()
        : device_id(-2),
          d_seeds(nullptr),
          d_counts(nullptr),
          d_poses(nullptr),
          d_drunk(nullptr),
          d_head_poses(nullptr),
          d_gen_profiles(nullptr),
          stream(nullptr),
          seeds_capacity_bytes(0),
          counts_capacity_bytes(0),
          poses_capacity_bytes(0),
          drunk_capacity_bytes(0),
          head_poses_capacity_bytes(0),
          gen_profiles_capacity_bytes(0)
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
    buffers.seeds_capacity_bytes = 0;
    buffers.counts_capacity_bytes = 0;
    buffers.poses_capacity_bytes = 0;
    buffers.drunk_capacity_bytes = 0;
    buffers.head_poses_capacity_bytes = 0;
    buffers.gen_profiles_capacity_bytes = 0;
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

__device__ __forceinline__ bool CheckCollisionFast(
    const dsp_vec3d_t* pts,
    int count,
    const dsp_vec3d_t& pt,
    double min_dist,
    float min_dist_f,
    double min2_fp64,
    double min2_f32,
    bool collision_fp64)
{
    if (collision_fp64)
        return CheckCollisionFp64Fast(pts, count, pt, min_dist, min2_fp64);
    return CheckCollisionF32Fast(pts, count, pt, min_dist_f, min2_f32);
}

template <bool CollectProfile>
__device__ __forceinline__ int GenerateRandomPosesParamsFp64Impl(
    int seed,
    int max_count,
    double min_dist,
    double min_step_len,
    double max_step_len,
    double flatten,
    bool collision_fp64,
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

                if (!CheckCollisionFast(poses, poses_count, pt, min_dist, min_dist_f, min2_fp64, min2_f32, collision_fp64))
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

                        if (!CheckCollisionFast(poses, poses_count, pt, min_dist, min_dist_f, min2_fp64, min2_f32, collision_fp64))
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

__global__ void GenerateTempPosesParamsFp64BatchKernel(
    const int* seeds,
    int seed_count,
    int max_count,
    double min_dist,
    double min_step_len,
    double max_step_len,
    double flatten,
    int collision_fp64,
    dsp_vec3d_t* out_poses,
    dsp_vec3d_t* out_drunk,
    int out_stride,
    int* out_counts,
    PoseGenSeedProfile* out_profiles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seed_count)
        return;

    dsp_vec3d_t* poses_seg = out_poses + static_cast<long long>(idx) * out_stride;
    dsp_vec3d_t* drunk_seg = out_drunk + static_cast<long long>(idx) * out_stride;
    int count = 0;
    if (out_profiles != nullptr)
    {
        count = GenerateRandomPosesParamsFp64Impl<true>(
            seeds[idx],
            max_count,
            min_dist,
            min_step_len,
            max_step_len,
            flatten,
            collision_fp64 != 0,
            poses_seg,
            drunk_seg,
            out_profiles + idx);
    }
    else
    {
        count = GenerateRandomPosesParamsFp64Impl<false>(
            seeds[idx],
            max_count,
            min_dist,
            min_step_len,
            max_step_len,
            flatten,
            collision_fp64 != 0,
            poses_seg,
            drunk_seg,
            nullptr);
    }
    out_counts[idx] = count;
}

__global__ void DebugRngNextDoubleKernel(int seed, int count, double* out_values)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    DotNet35RandomDevice rng(seed);
    for (int i = 0; i < count; ++i)
        out_values[i] = rng.NextDouble();
}

__global__ void DebugRngStateAfterCtorKernel(int seed, int* out_seed_array_56, int* out_inext, int* out_inextp)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    DotNet35RandomDevice rng(seed);
    for (int i = 0; i < 56; ++i)
        out_seed_array_56[i] = rng.seed_array[i];
    *out_inext = rng.inext;
    *out_inextp = rng.inextp;
}

__device__ __forceinline__ bool ProbHit(DotNet35RandomDevice& rng, float threshold, bool use_fp32_prob_compare)
{
    double rv = rng.NextDouble();
    if (use_fp32_prob_compare)
    {
        float rvf = static_cast<float>(rv);
        return rvf < threshold || (rvf == threshold && rv < static_cast<double>(threshold));
    }
    return rv < static_cast<double>(threshold);
}

constexpr int kPlanetTypeGas = 0;
constexpr int kPlanetTypeOcean = 1;
constexpr int kPlanetTypeVocano = 2;
constexpr int kPlanetTypeDesert = 3;
constexpr int kPlanetTypeIce = 4;

constexpr int kSingularityLaySide = 1;
constexpr int kSingularityTidal1 = 2;
constexpr int kSingularityTidal2 = 4;
constexpr int kSingularityTidal4 = 8;
constexpr int kSingularityClockwise = 16;

__device__ __forceinline__ void EvalPlanetCoreF32One(
    int info_seed,
    int orbit_around,
    int orbit_index,
    int gas_giant,
    int star_index,
    int galaxy_star_count,
    int galaxy_habitable_count,
    int boost_inclination_ns,
    int compact_type_case,
    float star_orbit_scaler,
    double star_mass,
    float star_habitable_radius,
    float star_light_balance_radius,
    float orbit_around_planet_real_radius,
    float orbit_around_planet_orbit_radius,
    double orbit_around_planet_orbital_period,
    dsp_planet_core_f32_out_t* out_result)
{
    if (out_result == nullptr)
        return;

    DotNet35RandomDevice rng(info_seed);
    double num3 = rng.NextDouble();
    double num4 = rng.NextDouble();
    double num5 = rng.NextDouble();
    double num6 = rng.NextDouble();
    double num7 = rng.NextDouble();
    double num8 = rng.NextDouble();
    double num9 = rng.NextDouble();
    double num10 = rng.NextDouble();
    double num11 = rng.NextDouble();
    double num12 = rng.NextDouble();
    double num13 = rng.NextDouble();
    double num14 = rng.NextDouble();
    double rand1 = rng.NextDouble();
    double num15 = rng.NextDouble();
    double rand2 = rng.NextDouble();
    double rand3 = rng.NextDouble();
    double rand4 = rng.NextDouble();
    int theme_seed = rng.InternalSample();

    float a = powf(1.2f, static_cast<float>(num3 * (num4 - 0.5) * 0.5));
    float orbit_radius = 0.0f;
    if (orbit_around == 0)
    {
        if (orbit_index < 0 || orbit_index >= 18)
            return;
        const float orbit_radius_table[18] = {
            0.0f, 0.4f, 0.7f, 1.0f, 1.4f, 1.9f, 2.5f, 3.3f, 4.3f,
            5.5f, 6.9f, 8.4f, 10.0f, 11.7f, 13.5f, 15.4f, 17.5f, 19.7f};
        float b = orbit_radius_table[orbit_index] * star_orbit_scaler;
        float num16 = (a - 1.0f) / fmaxf(1.0f, b) + 1.0f;
        orbit_radius = b * num16;
    }
    else
    {
        orbit_radius = static_cast<float>(
            ((1600.0 * static_cast<double>(orbit_index) + 200.0) *
                 static_cast<double>(powf(star_orbit_scaler, 0.3f)) *
                 static_cast<double>(a * 0.5f + 0.5f) +
             static_cast<double>(orbit_around_planet_real_radius)) /
            40000.0);
    }

    float orbit_inclination = static_cast<float>(num5 * 16.0 - 8.0);
    if (orbit_around > 0)
        orbit_inclination *= 2.2f;
    float orbit_longitude = static_cast<float>(num6 * 360.0);
    if (boost_inclination_ns != 0)
        orbit_inclination += (orbit_inclination > 0.0f) ? 3.0f : -3.0f;

    double f1 = static_cast<double>(orbit_radius);
    double f1_3 = f1 * f1 * f1;
    double orbital_period = 0.0;
    if (orbit_around > 0)
    {
        orbital_period = sqrt(39.4784176043574 * f1_3 / 1.08308421068537E-08);
    }
    else
    {
        orbital_period = sqrt(39.4784176043574 * f1_3 / (1.35385519905204E-06 * star_mass));
    }

    float orbit_phase = static_cast<float>(num7 * 360.0);
    float obliquity = 0.0f;
    int singularity_flags = 0;
    if (num15 < 0.0399999991059303)
    {
        obliquity = static_cast<float>(num8 * (num9 - 0.5) * 39.9);
        obliquity += (obliquity < 0.0f) ? -70.0f : 70.0f;
        singularity_flags |= kSingularityLaySide;
    }
    else if (num15 < 0.100000001490116)
    {
        obliquity = static_cast<float>(num8 * (num9 - 0.5) * 80.0);
        obliquity += (obliquity < 0.0f) ? -30.0f : 30.0f;
    }
    else
    {
        obliquity = static_cast<float>(num8 * (num9 - 0.5) * 60.0);
    }

    double rotation_period = (num10 * num11 * 1000.0 + 400.0) *
                             (orbit_around == 0 ? static_cast<double>(powf(orbit_radius, 0.25f)) : 1.0) *
                             (gas_giant != 0 ? 0.2 : 1.0);
    if (gas_giant == 0)
    {
        if (compact_type_case == 1)
            rotation_period *= 0.5;
        else if (compact_type_case == 2)
            rotation_period *= 0.2;
        else if (compact_type_case == 3)
            rotation_period *= 0.15;
    }
    float rotation_phase = static_cast<float>(num12 * 360.0);
    float sun_distance = (orbit_around == 0) ? orbit_radius : orbit_around_planet_orbit_radius;
    float scale = 1.0f;

    double num17 = (orbit_around == 0) ? orbital_period : orbit_around_planet_orbital_period;
    rotation_period = 1.0 / (1.0 / num17 + 1.0 / rotation_period);

    if (orbit_around == 0 && orbit_index <= 4 && gas_giant == 0)
    {
        if (num15 > 0.959999978542328)
        {
            obliquity *= 0.01f;
            rotation_period = orbital_period;
            singularity_flags |= kSingularityTidal1;
        }
        else if (num15 > 0.930000007152557)
        {
            obliquity *= 0.1f;
            rotation_period = orbital_period * 0.5;
            singularity_flags |= kSingularityTidal2;
        }
        else if (num15 > 0.899999976158142)
        {
            obliquity *= 0.2f;
            rotation_period = orbital_period * 0.25;
            singularity_flags |= kSingularityTidal4;
        }
    }

    if (num15 > 0.85 && num15 <= 0.9)
    {
        rotation_period = -rotation_period;
        singularity_flags |= kSingularityClockwise;
    }

    float habitable_bias = 0.0f;
    float temperature_bias = 0.0f;
    float radius = 0.0f;
    int type_case = kPlanetTypeDesert;
    int habitable_count_delta = 0;

    if (gas_giant != 0)
    {
        type_case = kPlanetTypeGas;
        radius = 80.0f;
        scale = 10.0f;
        habitable_bias = 100.0f;
    }
    else
    {
        float num18 = ceilf(static_cast<float>(galaxy_star_count) * 0.29f);
        if (num18 < 11.0f)
            num18 = 11.0f;
        float num19 = num18 - static_cast<float>(galaxy_habitable_count);
        float num20 = static_cast<float>(galaxy_star_count - star_index);
        float f2 = 1000.0f;
        float num21 = 1000.0f;
        if (star_habitable_radius > 0.0f && sun_distance > 0.0f)
        {
            f2 = sun_distance / star_habitable_radius;
            num21 = fabsf(logf(f2));
        }

        float num22 = fminf(fmaxf(sqrtf(star_habitable_radius), 1.0f), 2.0f) - 0.04f;
        float num24 = fminf(fmaxf(((num19 / fmaxf(1.0f, num20)) * (1.0f - 0.5f) + 0.5f) * 0.35f + ((num19 / fmaxf(1.0f, num20)) * 0.5f), 0.08f), 0.8f);
        // 上面等价于 Mathf.Lerp(num19/num20, 0.35f, 0.5f) 再 clamp
        num24 = fminf(fmaxf((num19 / fmaxf(1.0f, num20)) * 0.5f + 0.175f, 0.08f), 0.8f);

        habitable_bias = num21 * num22;
        temperature_bias = static_cast<float>(1.20000004768372 / (f2 + 0.2f) - 1.0);
        float num25 = powf(fminf(fmaxf(habitable_bias / num24, 0.0f), 1.0f), num24 * 10.0f);

        if ((num13 > num25 && star_index > 0) || (orbit_around > 0 && orbit_index == 1 && star_index == 0))
        {
            type_case = kPlanetTypeOcean;
            habitable_count_delta = 1;
        }
        else if (f2 < 0.833333f)
        {
            float num26 = fmaxf(0.15f, f2 * 2.5f - 0.85f);
            type_case = (num14 >= num26) ? kPlanetTypeVocano : kPlanetTypeDesert;
        }
        else if (f2 < 1.2f)
        {
            type_case = kPlanetTypeDesert;
        }
        else
        {
            float num27 = 0.9f / f2 - 0.1f;
            type_case = (num14 >= num27) ? kPlanetTypeIce : kPlanetTypeDesert;
        }

        radius = 200.0f;
    }

    int precision = (type_case == kPlanetTypeGas) ? 64 : 200;
    int segment = (type_case == kPlanetTypeGas) ? 2 : 5;

    float luminosity = powf(star_light_balance_radius / (sun_distance + 0.01f), 0.6f);
    if (luminosity > 1.0f)
    {
        luminosity = logf(luminosity) + 1.0f;
        luminosity = logf(luminosity) + 1.0f;
        luminosity = logf(luminosity) + 1.0f;
    }
    luminosity = roundf(luminosity * 100.0f) / 100.0f;

    out_result->orbit_radius = orbit_radius;
    out_result->orbit_inclination = orbit_inclination;
    out_result->orbit_longitude = orbit_longitude;
    out_result->orbital_period = orbital_period;
    out_result->orbit_phase = orbit_phase;
    out_result->obliquity = obliquity;
    out_result->rotation_period = rotation_period;
    out_result->rotation_phase = rotation_phase;
    out_result->sun_distance = sun_distance;
    out_result->scale = scale;
    out_result->habitable_bias = habitable_bias;
    out_result->temperature_bias = temperature_bias;
    out_result->radius = radius;
    out_result->luminosity = luminosity;
    out_result->rand1 = rand1;
    out_result->rand2 = rand2;
    out_result->rand3 = rand3;
    out_result->rand4 = rand4;
    out_result->num13 = num13;
    out_result->num14 = num14;
    out_result->theme_seed = theme_seed;
    out_result->type_case = type_case;
    out_result->singularity_flags = singularity_flags;
    out_result->habitable_count_delta = habitable_count_delta;
    out_result->precision = precision;
    out_result->segment = segment;
}

__global__ void EvalPlanetCoreF32Kernel(
    int info_seed,
    int orbit_around,
    int orbit_index,
    int gas_giant,
    int star_index,
    int galaxy_star_count,
    int galaxy_habitable_count,
    int boost_inclination_ns,
    int compact_type_case,
    float star_orbit_scaler,
    double star_mass,
    float star_habitable_radius,
    float star_light_balance_radius,
    float orbit_around_planet_real_radius,
    float orbit_around_planet_orbit_radius,
    double orbit_around_planet_orbital_period,
    dsp_planet_core_f32_out_t* out_result)
{
    if (blockIdx.x != 0 || threadIdx.x != 0 || out_result == nullptr)
        return;

    EvalPlanetCoreF32One(
        info_seed,
        orbit_around,
        orbit_index,
        gas_giant,
        star_index,
        galaxy_star_count,
        galaxy_habitable_count,
        boost_inclination_ns,
        compact_type_case,
        star_orbit_scaler,
        star_mass,
        star_habitable_radius,
        star_light_balance_radius,
        orbit_around_planet_real_radius,
        orbit_around_planet_orbit_radius,
        orbit_around_planet_orbital_period,
        out_result);
}

__global__ void EvalPlanetCoreF32BatchKernel(
    const int* info_seeds,
    const int* orbit_arounds,
    const int* orbit_indexes,
    const int* gas_giants,
    const int* star_indexes,
    const int* galaxy_star_counts,
    const int* galaxy_habitable_counts,
    const int* boost_inclination_ns,
    const int* compact_type_cases,
    const float* star_orbit_scalers,
    const double* star_masses,
    const float* star_habitable_radiuses,
    const float* star_light_balance_radiuses,
    const float* orbit_around_planet_real_radiuses,
    const float* orbit_around_planet_orbit_radiuses,
    const double* orbit_around_planet_orbital_periods,
    int batch_count,
    dsp_planet_core_f32_out_t* out_results)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 0 || idx >= batch_count || out_results == nullptr)
        return;

    EvalPlanetCoreF32One(
        info_seeds[idx],
        orbit_arounds[idx],
        orbit_indexes[idx],
        gas_giants[idx],
        star_indexes[idx],
        galaxy_star_counts[idx],
        galaxy_habitable_counts[idx],
        boost_inclination_ns[idx],
        compact_type_cases[idx],
        star_orbit_scalers[idx],
        star_masses[idx],
        star_habitable_radiuses[idx],
        star_light_balance_radiuses[idx],
        orbit_around_planet_real_radiuses[idx],
        orbit_around_planet_orbit_radiuses[idx],
        orbit_around_planet_orbital_periods[idx],
        out_results + idx);
}

__global__ void EvalPlanetGasDetailsF32Kernel(
    int theme_seed,
    float gas_coef,
    float resource_coef,
    const float* in_gas_speeds,
    const float* in_gas_heat_values,
    int gas_len,
    float* out_gas_speeds,
    double* out_total_heat)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    DotNet35RandomDevice rng(theme_seed);
    float resource_scale = powf(resource_coef, 0.3f);
    double total_heat = 0.0;
    for (int i = 0; i < gas_len; ++i)
    {
        float rand_factor = static_cast<float>(rng.NextDouble() * 0.190909147262573 + 0.909090876579285);
        float speed = in_gas_speeds[i] * rand_factor * gas_coef;
        float scaled_speed = speed * resource_scale;
        out_gas_speeds[i] = scaled_speed;
        total_heat += static_cast<double>(in_gas_heat_values[i] * scaled_speed);
    }
    *out_total_heat = total_heat;
}

__global__ void RefreshPlanetVeinSpotsBatchKernel(
    const int* planet_seeds,
    const float* p_values,
    const int* bonus_cases,
    const int* is_birth_stars,
    const int* vein_spot_lens,
    const int* rare_vein_lens,
    const int* vein_spot_values,
    int vein_spot_stride,
    const int* rare_vein_values,
    int rare_vein_stride,
    const float* rare_settings_values,
    int rare_settings_stride,
    int planet_count,
    int out_vein_len,
    int use_fp32_prob_compare,
    int* out_counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= planet_count)
        return;

    int* out_seg = out_counts + static_cast<long long>(idx) * out_vein_len;
    for (int i = 0; i < out_vein_len; ++i)
        out_seg[i] = 0;

    const int vlen = vein_spot_lens[idx];
    const int* vspot = vein_spot_values + static_cast<long long>(idx) * vein_spot_stride;
    for (int i = 0; i < vlen; ++i)
    {
        int out_idx = i + 1;
        if (out_idx >= 0 && out_idx < out_vein_len)
            out_seg[out_idx] = vspot[i];
    }

    DotNet35RandomDevice rng(planet_seeds[idx]);
    rng.InternalSample();
    rng.InternalSample();
    rng.InternalSample();
    rng.InternalSample();
    rng.InternalSample();
    DotNet35RandomDevice rng2(rng.InternalSample());
    (void)rng2;

    const bool use_fp32 = use_fp32_prob_compare != 0;
    const int bonus_case = bonus_cases[idx];
    if (bonus_case == 1)
    {
        if (9 < out_vein_len)
        {
            ++out_seg[9];
            ++out_seg[9];
            for (int i = 1; i < 12 && ProbHit(rng, 0.449999988079071f, use_fp32); ++i)
                ++out_seg[9];
        }
        if (10 < out_vein_len)
        {
            ++out_seg[10];
            ++out_seg[10];
            for (int i = 1; i < 12 && ProbHit(rng, 0.449999988079071f, use_fp32); ++i)
                ++out_seg[10];
        }
        if (12 < out_vein_len)
        {
            ++out_seg[12];
            for (int i = 1; i < 12 && ProbHit(rng, 0.5f, use_fp32); ++i)
                ++out_seg[12];
        }
    }
    else if (bonus_case == 2)
    {
        if (14 < out_vein_len)
        {
            ++out_seg[14];
            for (int i = 1; i < 12 && ProbHit(rng, 0.649999976158142f, use_fp32); ++i)
                ++out_seg[14];
        }
    }

    const int rare_len = rare_vein_lens[idx];
    const int* rveins = rare_vein_values + static_cast<long long>(idx) * rare_vein_stride;
    const float* rsettings = rare_settings_values + static_cast<long long>(idx) * rare_settings_stride;
    const bool is_birth = is_birth_stars[idx] != 0;
    const float p = p_values[idx];
    for (int ri = 0; ri < rare_len; ++ri)
    {
        int vein_id = rveins[ri];
        int sbase = ri * 4;
        float appear_base = is_birth ? rsettings[sbase] : rsettings[sbase + 1];
        float chain_prob = rsettings[sbase + 2];
        float appear_prob = 1.0f - powf(1.0f - appear_base, p);

        if (ProbHit(rng, appear_prob, use_fp32))
        {
            if (vein_id > 0 && vein_id < out_vein_len)
                ++out_seg[vein_id];
            for (int i = 1; i < 12 && ProbHit(rng, chain_prob, use_fp32); ++i)
            {
                if (vein_id > 0 && vein_id < out_vein_len)
                    ++out_seg[vein_id];
            }
        }
    }
}

__global__ void RefreshPlanetVeinSpotsByThemeBatchKernel(
    const int* planet_seeds,
    const float* p_values,
    const int* bonus_cases,
    const int* is_birth_stars,
    const int* theme_indexes,
    int planet_count,
    int out_vein_len,
    int use_fp32_prob_compare,
    const int* theme_vein_spot_offsets,
    const int* theme_vein_spot_values,
    const int* theme_rare_vein_offsets,
    const int* theme_rare_vein_values,
    const int* theme_rare_settings_offsets,
    const float* theme_rare_settings_values,
    int theme_count,
    int* out_counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= planet_count)
        return;

    int* out_seg = out_counts + static_cast<long long>(idx) * out_vein_len;
    for (int i = 0; i < out_vein_len; ++i)
        out_seg[i] = 0;

    int theme_idx = theme_indexes[idx];
    if (theme_idx < 0 || theme_idx >= theme_count)
        return;

    int vein_begin = theme_vein_spot_offsets[theme_idx];
    int vein_end = theme_vein_spot_offsets[theme_idx + 1];
    int vlen = vein_end - vein_begin;
    if (vlen < 0)
        vlen = 0;
    for (int i = 0; i < vlen; ++i)
    {
        int out_idx = i + 1;
        if (out_idx >= 0 && out_idx < out_vein_len)
            out_seg[out_idx] = theme_vein_spot_values[vein_begin + i];
    }

    DotNet35RandomDevice rng(planet_seeds[idx]);
    rng.InternalSample();
    rng.InternalSample();
    rng.InternalSample();
    rng.InternalSample();
    rng.InternalSample();
    DotNet35RandomDevice rng2(rng.InternalSample());
    (void)rng2;

    const bool use_fp32 = use_fp32_prob_compare != 0;
    const int bonus_case = bonus_cases[idx];
    if (bonus_case == 1)
    {
        if (9 < out_vein_len)
        {
            ++out_seg[9];
            ++out_seg[9];
            for (int i = 1; i < 12 && ProbHit(rng, 0.449999988079071f, use_fp32); ++i)
                ++out_seg[9];
        }
        if (10 < out_vein_len)
        {
            ++out_seg[10];
            ++out_seg[10];
            for (int i = 1; i < 12 && ProbHit(rng, 0.449999988079071f, use_fp32); ++i)
                ++out_seg[10];
        }
        if (12 < out_vein_len)
        {
            ++out_seg[12];
            for (int i = 1; i < 12 && ProbHit(rng, 0.5f, use_fp32); ++i)
                ++out_seg[12];
        }
    }
    else if (bonus_case == 2)
    {
        if (14 < out_vein_len)
        {
            ++out_seg[14];
            for (int i = 1; i < 12 && ProbHit(rng, 0.649999976158142f, use_fp32); ++i)
                ++out_seg[14];
        }
    }

    int rare_begin = theme_rare_vein_offsets[theme_idx];
    int rare_end = theme_rare_vein_offsets[theme_idx + 1];
    int rare_len = rare_end - rare_begin;
    if (rare_len < 0)
        rare_len = 0;

    int settings_begin = theme_rare_settings_offsets[theme_idx];
    int settings_end = theme_rare_settings_offsets[theme_idx + 1];
    int settings_len = settings_end - settings_begin;
    if (settings_len < 0)
        settings_len = 0;

    const bool is_birth = is_birth_stars[idx] != 0;
    const float p = p_values[idx];
    for (int ri = 0; ri < rare_len; ++ri)
    {
        int vein_id = theme_rare_vein_values[rare_begin + ri];
        int local_sbase = ri * 4;
        float s0 = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        if (local_sbase + 0 < settings_len) s0 = theme_rare_settings_values[settings_begin + local_sbase + 0];
        if (local_sbase + 1 < settings_len) s1 = theme_rare_settings_values[settings_begin + local_sbase + 1];
        if (local_sbase + 2 < settings_len) s2 = theme_rare_settings_values[settings_begin + local_sbase + 2];

        float appear_base = is_birth ? s0 : s1;
        float chain_prob = s2;
        float appear_prob = 1.0f - powf(1.0f - appear_base, p);

        if (ProbHit(rng, appear_prob, use_fp32))
        {
            if (vein_id > 0 && vein_id < out_vein_len)
                ++out_seg[vein_id];
            for (int i = 1; i < 12 && ProbHit(rng, chain_prob, use_fp32); ++i)
            {
                if (vein_id > 0 && vein_id < out_vein_len)
                    ++out_seg[vein_id];
            }
        }
    }
}

__global__ void GatherTempPosesHeadKernel(
    const dsp_vec3d_t* in_poses,
    int seed_count,
    int in_stride,
    int head_count,
    int sample_step,
    dsp_vec3d_t* out_poses,
    int out_stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seed_count * head_count;
    if (idx >= total)
        return;
    int seed_idx = idx / head_count;
    int local_idx = idx - seed_idx * head_count;
    int src_idx = local_idx * sample_step;
    out_poses[seed_idx * out_stride + local_idx] = in_poses[seed_idx * in_stride + src_idx];
}
} // namespace

extern "C" int dsp_cuda_generate_temp_poses_params_fp64_batch(
    const int* seeds,
    int seed_count,
    int max_count,
    double min_dist,
    double min_step_len,
    double max_step_len,
    double flatten,
    int collision_fp64,
    int device_id,
    dsp_vec3d_t* out_poses,
    int out_stride,
    int* out_counts)
{
    bool debug_enter = false;
    if (const char* de = std::getenv("DSP_NATIVE_SIG_DEBUG_ENTER"))
        debug_enter = std::atoi(de) != 0;
    if (seeds == nullptr || out_poses == nullptr || out_counts == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (seed_count <= 0 || max_count <= 0 || out_stride < max_count)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;

    if (device_id >= 0)
    {
        cudaError_t set_device_rc = cudaSetDevice(device_id);
        if (set_device_rc != cudaSuccess)
        {
            if (debug_enter)
                std::fprintf(stderr, "[cuda-galaxy-error] pose_batch cudaSetDevice rc=%d %s\n", static_cast<int>(set_device_rc), cudaGetErrorString(set_device_rc));
            return DSP_CUDA_ERR_CUDA;
        }
    }

    const size_t seeds_bytes = static_cast<size_t>(seed_count) * sizeof(int);
    const size_t counts_bytes = static_cast<size_t>(seed_count) * sizeof(int);
    const size_t poses_bytes = static_cast<size_t>(seed_count) * static_cast<size_t>(out_stride) * sizeof(dsp_vec3d_t);

    int* d_seeds = nullptr;
    int* d_counts = nullptr;
    dsp_vec3d_t* d_poses = nullptr;
    dsp_vec3d_t* d_drunk = nullptr;
    cudaStream_t stream = nullptr;

    cudaError_t rc = EnsureBatchBuffers(
        device_id,
        seeds_bytes,
        counts_bytes,
        poses_bytes,
        0,
        0,
        &d_seeds,
        &d_counts,
        &d_poses,
        &d_drunk,
        nullptr,
        nullptr,
        &stream);
    if (rc != cudaSuccess)
    {
        if (debug_enter)
            std::fprintf(stderr, "[cuda-galaxy-error] pose_batch EnsureBatchBuffers rc=%d %s\n", static_cast<int>(rc), cudaGetErrorString(rc));
        return DSP_CUDA_ERR_CUDA;
    }

    rc = cudaMemcpyAsync(d_seeds, seeds, seeds_bytes, cudaMemcpyHostToDevice, stream);
    if (rc != cudaSuccess)
    {
        if (debug_enter)
            std::fprintf(stderr, "[cuda-galaxy-error] pose_batch H2D seeds rc=%d %s\n", static_cast<int>(rc), cudaGetErrorString(rc));
        return DSP_CUDA_ERR_CUDA;
    }

    int block_size = 128;
    int grid_size = (seed_count + block_size - 1) / block_size;
    GenerateTempPosesParamsFp64BatchKernel<<<grid_size, block_size, 0, stream>>>(
        d_seeds,
        seed_count,
        max_count,
        min_dist,
        min_step_len,
        max_step_len,
        flatten,
        collision_fp64,
        d_poses,
        d_drunk,
        out_stride,
        d_counts,
        nullptr);

    rc = cudaGetLastError();
    if (rc == cudaSuccess)
        rc = cudaMemcpyAsync(out_counts, d_counts, counts_bytes, cudaMemcpyDeviceToHost, stream);
    if (rc == cudaSuccess)
        rc = cudaMemcpyAsync(out_poses, d_poses, poses_bytes, cudaMemcpyDeviceToHost, stream);
    if (rc == cudaSuccess)
        rc = cudaStreamSynchronize(stream);

    if (rc != cudaSuccess)
    {
        if (debug_enter)
            std::fprintf(stderr, "[cuda-galaxy-error] pose_batch post-kernel rc=%d %s\n", static_cast<int>(rc), cudaGetErrorString(rc));
        return DSP_CUDA_ERR_CUDA;
    }
    return DSP_CUDA_OK;
}

extern "C" int dsp_cuda_generate_temp_poses_params_fp64_batch_head(
    const int* seeds,
    int seed_count,
    int max_count,
    int head_count,
    int sample_step,
    double min_dist,
    double min_step_len,
    double max_step_len,
    double flatten,
    int collision_fp64,
    int device_id,
    dsp_vec3d_t* out_poses,
    int out_stride,
    int* out_counts)
{
    g_last_pose_batch_head_timing =
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    bool debug_enter = false;
    if (const char* de = std::getenv("DSP_NATIVE_SIG_DEBUG_ENTER"))
        debug_enter = std::atoi(de) != 0;
    if (seeds == nullptr || out_poses == nullptr || out_counts == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (seed_count <= 0 || max_count <= 0 || head_count <= 0 || head_count > max_count || out_stride < head_count)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (sample_step <= 0)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if ((head_count - 1) * sample_step >= max_count)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    bool op_timing = false;
    if (const char* st = std::getenv("DSP_NATIVE_SIG_STAGE_TIMING"))
        op_timing = std::atoi(st) != 0;
    if (const char* ot = std::getenv("DSP_NATIVE_OP_TIMING"))
        op_timing = op_timing || (std::atoi(ot) != 0);
    auto now_ms_host = []() -> double {
        using clock = std::chrono::steady_clock;
        return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
    };

    if (device_id >= 0)
    {
        cudaError_t set_device_rc = cudaSetDevice(device_id);
        if (set_device_rc != cudaSuccess)
        {
            if (debug_enter)
                std::fprintf(stderr, "[cuda-galaxy-error] pose_batch_head cudaSetDevice rc=%d %s\n", static_cast<int>(set_device_rc), cudaGetErrorString(set_device_rc));
            return DSP_CUDA_ERR_CUDA;
        }
    }

    const int device_stride = max_count;
    const size_t seeds_bytes = static_cast<size_t>(seed_count) * sizeof(int);
    const size_t counts_bytes = static_cast<size_t>(seed_count) * sizeof(int);
    const size_t poses_bytes = static_cast<size_t>(seed_count) * static_cast<size_t>(device_stride) * sizeof(dsp_vec3d_t);
    const size_t head_bytes = static_cast<size_t>(seed_count) * static_cast<size_t>(out_stride) * sizeof(dsp_vec3d_t);
    const size_t gen_profiles_bytes = static_cast<size_t>(seed_count) * sizeof(PoseGenSeedProfile);

    int* d_seeds = nullptr;
    int* d_counts = nullptr;
    dsp_vec3d_t* d_poses = nullptr;
    dsp_vec3d_t* d_drunk = nullptr;
    dsp_vec3d_t* d_head_poses = nullptr;
    PoseGenSeedProfile* d_gen_profiles = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr, ev2 = nullptr, ev3 = nullptr, ev4 = nullptr, ev5 = nullptr;
    auto destroy_events = [&]() {
        if (ev5 != nullptr) cudaEventDestroy(ev5);
        if (ev4 != nullptr) cudaEventDestroy(ev4);
        if (ev3 != nullptr) cudaEventDestroy(ev3);
        if (ev2 != nullptr) cudaEventDestroy(ev2);
        if (ev1 != nullptr) cudaEventDestroy(ev1);
        if (ev0 != nullptr) cudaEventDestroy(ev0);
    };

    cudaError_t rc = EnsureBatchBuffers(
        device_id,
        seeds_bytes,
        counts_bytes,
        poses_bytes,
        head_bytes,
        op_timing ? gen_profiles_bytes : 0,
        &d_seeds,
        &d_counts,
        &d_poses,
        &d_drunk,
        &d_head_poses,
        &d_gen_profiles,
        &stream);
    if (rc != cudaSuccess)
    {
        if (debug_enter)
            std::fprintf(stderr, "[cuda-galaxy-error] pose_batch_head EnsureBatchBuffers rc=%d %s\n", static_cast<int>(rc), cudaGetErrorString(rc));
        return DSP_CUDA_ERR_CUDA;
    }
    if (op_timing)
    {
        rc = cudaEventCreate(&ev0);
        if (rc == cudaSuccess) rc = cudaEventCreate(&ev1);
        if (rc == cudaSuccess) rc = cudaEventCreate(&ev2);
        if (rc == cudaSuccess) rc = cudaEventCreate(&ev3);
        if (rc == cudaSuccess) rc = cudaEventCreate(&ev4);
        if (rc == cudaSuccess) rc = cudaEventCreate(&ev5);
        if (rc != cudaSuccess)
        {
            destroy_events();
            ev0 = ev1 = ev2 = ev3 = ev4 = ev5 = nullptr;
            op_timing = false;
        }
    }
    if (op_timing)
        cudaEventRecord(ev0, stream);

    rc = cudaMemcpyAsync(d_seeds, seeds, seeds_bytes, cudaMemcpyHostToDevice, stream);
    if (rc != cudaSuccess)
    {
        if (debug_enter)
            std::fprintf(stderr, "[cuda-galaxy-error] pose_batch_head H2D seeds rc=%d %s\n", static_cast<int>(rc), cudaGetErrorString(rc));
        return DSP_CUDA_ERR_CUDA;
    }
    if (op_timing)
        cudaEventRecord(ev1, stream);

    int block_size = 128;
    int grid_size = (seed_count + block_size - 1) / block_size;
    GenerateTempPosesParamsFp64BatchKernel<<<grid_size, block_size, 0, stream>>>(
        d_seeds,
        seed_count,
        max_count,
        min_dist,
        min_step_len,
        max_step_len,
        flatten,
        collision_fp64,
        d_poses,
        d_drunk,
        device_stride,
        d_counts,
        op_timing ? d_gen_profiles : nullptr);
    if (op_timing)
        cudaEventRecord(ev2, stream);

    rc = cudaGetLastError();
    if (rc == cudaSuccess)
        rc = cudaMemcpyAsync(out_counts, d_counts, counts_bytes, cudaMemcpyDeviceToHost, stream);
    if (rc == cudaSuccess && op_timing)
        cudaEventRecord(ev3, stream);
    if (rc == cudaSuccess)
    {
        int block = 128;
        int grid = (seed_count * head_count + block - 1) / block;
        GatherTempPosesHeadKernel<<<grid, block, 0, stream>>>(
            d_poses,
            seed_count,
            device_stride,
            head_count,
            sample_step,
            d_head_poses,
            out_stride);
        if (op_timing)
            cudaEventRecord(ev4, stream);
        rc = cudaGetLastError();
        if (rc == cudaSuccess)
        {
            double t_submit_begin = op_timing ? now_ms_host() : 0.0;
            rc = cudaMemcpyAsync(out_poses, d_head_poses, head_bytes, cudaMemcpyDeviceToHost, stream);
            if (rc == cudaSuccess && op_timing)
                g_last_pose_batch_head_timing.d2h_head_submit_ms += (now_ms_host() - t_submit_begin);
        }
        if (rc == cudaSuccess && op_timing)
            cudaEventRecord(ev5, stream);
        if (rc == cudaSuccess)
        {
            double t_sync_begin = op_timing ? now_ms_host() : 0.0;
            rc = cudaStreamSynchronize(stream);
            if (rc == cudaSuccess && op_timing)
                g_last_pose_batch_head_timing.d2h_head_sync_wait_ms += (now_ms_host() - t_sync_begin);
        }
    }
    if (rc == cudaSuccess && op_timing)
    {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, ev0, ev1) == cudaSuccess) g_last_pose_batch_head_timing.h2d_seeds_ms = static_cast<double>(ms);
        if (cudaEventElapsedTime(&ms, ev1, ev2) == cudaSuccess) g_last_pose_batch_head_timing.gen_kernel_ms = static_cast<double>(ms);
        if (cudaEventElapsedTime(&ms, ev2, ev3) == cudaSuccess) g_last_pose_batch_head_timing.d2h_counts_ms = static_cast<double>(ms);
        if (cudaEventElapsedTime(&ms, ev3, ev4) == cudaSuccess) g_last_pose_batch_head_timing.gather_kernel_ms = static_cast<double>(ms);
        if (cudaEventElapsedTime(&ms, ev4, ev5) == cudaSuccess) g_last_pose_batch_head_timing.d2h_head_ms = static_cast<double>(ms);
        if (cudaEventElapsedTime(&ms, ev0, ev5) == cudaSuccess) g_last_pose_batch_head_timing.total_ms = static_cast<double>(ms);
        g_last_pose_batch_head_timing.d2h_head_bytes_mb = static_cast<double>(head_bytes) / (1024.0 * 1024.0);
        if (g_last_pose_batch_head_timing.d2h_head_ms > 1e-9)
            g_last_pose_batch_head_timing.d2h_head_bw_gbps =
                static_cast<double>(head_bytes) / (g_last_pose_batch_head_timing.d2h_head_ms * 1.0e6);
    }
    if (rc == cudaSuccess && op_timing && d_gen_profiles != nullptr)
    {
        std::vector<PoseGenSeedProfile> host_profiles(seed_count);
        cudaError_t rc_prof = cudaMemcpy(
            host_profiles.data(),
            d_gen_profiles,
            gen_profiles_bytes,
            cudaMemcpyDeviceToHost);
        if (rc_prof == cudaSuccess)
        {
            int profile_device = device_id;
            if (profile_device < 0)
            {
                if (cudaGetDevice(&profile_device) != cudaSuccess)
                    profile_device = 0;
            }
            int clock_khz = 0;
            if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, profile_device) == cudaSuccess && clock_khz > 0)
            {
                std::vector<double> seed_ms;
                seed_ms.reserve(seed_count);
                double phase1_ms = 0.0;
                double phase2_ms = 0.0;
                double attempts_total = 0.0;
                double collision_total = 0.0;
                double sphere_total = 0.0;
                double gate_total = 0.0;
                for (int i = 0; i < seed_count; ++i)
                {
                    const PoseGenSeedProfile& p = host_profiles[i];
                    phase1_ms += static_cast<double>(p.phase1_cycles) / static_cast<double>(clock_khz);
                    phase2_ms += static_cast<double>(p.phase2_cycles) / static_cast<double>(clock_khz);
                    seed_ms.push_back(static_cast<double>(p.phase1_cycles + p.phase2_cycles) / static_cast<double>(clock_khz));
                    attempts_total += static_cast<double>(p.attempts);
                    collision_total += static_cast<double>(p.collision_rejects);
                    sphere_total += static_cast<double>(p.sphere_rejects);
                    gate_total += static_cast<double>(p.gate_skips);
                }
                std::sort(seed_ms.begin(), seed_ms.end());
                auto percentile = [&](double q) -> double {
                    if (seed_ms.empty())
                        return 0.0;
                    double pos = q * static_cast<double>(seed_ms.size() - 1);
                    size_t idx = static_cast<size_t>(pos);
                    size_t idx2 = std::min(idx + 1, seed_ms.size() - 1);
                    double frac = pos - static_cast<double>(idx);
                    return seed_ms[idx] * (1.0 - frac) + seed_ms[idx2] * frac;
                };
                g_last_pose_batch_head_timing.gen_phase1_ms = phase1_ms;
                g_last_pose_batch_head_timing.gen_phase2_ms = phase2_ms;
                g_last_pose_batch_head_timing.gen_seed_p50_ms = percentile(0.50);
                g_last_pose_batch_head_timing.gen_seed_p95_ms = percentile(0.95);
                g_last_pose_batch_head_timing.gen_seed_max_ms = seed_ms.empty() ? 0.0 : seed_ms.back();
                g_last_pose_batch_head_timing.gen_attempts_total = attempts_total;
                g_last_pose_batch_head_timing.gen_collision_rejects_total = collision_total;
                g_last_pose_batch_head_timing.gen_sphere_rejects_total = sphere_total;
                g_last_pose_batch_head_timing.gen_gate_skips_total = gate_total;
                if (seed_count > 0)
                {
                    g_last_pose_batch_head_timing.gen_phase1_ms /= static_cast<double>(seed_count);
                    g_last_pose_batch_head_timing.gen_phase2_ms /= static_cast<double>(seed_count);
                }
            }
        }
    }
    destroy_events();

    if (rc != cudaSuccess)
    {
        if (debug_enter)
            std::fprintf(stderr, "[cuda-galaxy-error] pose_batch_head post-kernel rc=%d %s\n", static_cast<int>(rc), cudaGetErrorString(rc));
        return DSP_CUDA_ERR_CUDA;
    }
    return DSP_CUDA_OK;
}

extern "C" int dsp_cuda_get_last_pose_batch_head_timing(
    dsp_cuda_pose_batch_head_timing_t* out_timing)
{
    if (out_timing == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    *out_timing = g_last_pose_batch_head_timing;
    return DSP_CUDA_OK;
}

extern "C" int dsp_cuda_generate_temp_poses_params_fp64(
    int seed,
    int max_count,
    double min_dist,
    double min_step_len,
    double max_step_len,
    double flatten,
    int collision_fp64,
    int device_id,
    dsp_vec3d_t* out_poses,
    int out_capacity,
    int* out_count)
{
    if (out_capacity < max_count || out_count == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;

    int seed_value = seed;
    int count_value = 0;
    int rc = dsp_cuda_generate_temp_poses_params_fp64_batch(
        &seed_value,
        1,
        max_count,
        min_dist,
        min_step_len,
        max_step_len,
        flatten,
        collision_fp64,
        device_id,
        out_poses,
        out_capacity,
        &count_value);
    *out_count = count_value;
    return rc;
}

extern "C" int dsp_cuda_debug_rng_nextdouble(
    int seed,
    int count,
    int device_id,
    double* out_values)
{
    if (count <= 0 || out_values == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;

    if (device_id >= 0)
    {
        cudaError_t set_device_rc = cudaSetDevice(device_id);
        if (set_device_rc != cudaSuccess)
            return DSP_CUDA_ERR_CUDA;
    }

    double* d_values = nullptr;
    size_t values_bytes = static_cast<size_t>(count) * sizeof(double);
    cudaError_t rc = cudaMalloc(reinterpret_cast<void**>(&d_values), values_bytes);
    if (rc != cudaSuccess)
        return DSP_CUDA_ERR_CUDA;

    DebugRngNextDoubleKernel<<<1, 1>>>(seed, count, d_values);
    rc = cudaGetLastError();
    if (rc == cudaSuccess)
        rc = cudaDeviceSynchronize();
    if (rc == cudaSuccess)
        rc = cudaMemcpy(out_values, d_values, values_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    if (rc != cudaSuccess)
        return DSP_CUDA_ERR_CUDA;
    return DSP_CUDA_OK;
}

extern "C" int dsp_cuda_debug_rng_state_after_ctor(
    int seed,
    int device_id,
    int* out_seed_array_56,
    int* out_inext,
    int* out_inextp)
{
    if (out_seed_array_56 == nullptr || out_inext == nullptr || out_inextp == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;

    if (device_id >= 0)
    {
        cudaError_t set_device_rc = cudaSetDevice(device_id);
        if (set_device_rc != cudaSuccess)
            return DSP_CUDA_ERR_CUDA;
    }

    int* d_seed_array = nullptr;
    int* d_inext = nullptr;
    int* d_inextp = nullptr;
    cudaError_t rc = cudaMalloc(reinterpret_cast<void**>(&d_seed_array), sizeof(int) * 56);
    if (rc != cudaSuccess)
        return DSP_CUDA_ERR_CUDA;
    rc = cudaMalloc(reinterpret_cast<void**>(&d_inext), sizeof(int));
    if (rc != cudaSuccess)
    {
        cudaFree(d_seed_array);
        return DSP_CUDA_ERR_CUDA;
    }
    rc = cudaMalloc(reinterpret_cast<void**>(&d_inextp), sizeof(int));
    if (rc != cudaSuccess)
    {
        cudaFree(d_inext);
        cudaFree(d_seed_array);
        return DSP_CUDA_ERR_CUDA;
    }

    DebugRngStateAfterCtorKernel<<<1, 1>>>(seed, d_seed_array, d_inext, d_inextp);
    rc = cudaGetLastError();
    if (rc == cudaSuccess)
        rc = cudaDeviceSynchronize();
    if (rc == cudaSuccess)
        rc = cudaMemcpy(out_seed_array_56, d_seed_array, sizeof(int) * 56, cudaMemcpyDeviceToHost);
    if (rc == cudaSuccess)
        rc = cudaMemcpy(out_inext, d_inext, sizeof(int), cudaMemcpyDeviceToHost);
    if (rc == cudaSuccess)
        rc = cudaMemcpy(out_inextp, d_inextp, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_inextp);
    cudaFree(d_inext);
    cudaFree(d_seed_array);

    if (rc != cudaSuccess)
        return DSP_CUDA_ERR_CUDA;
    return DSP_CUDA_OK;
}

extern "C" int dsp_cuda_refresh_planet_vein_spots_batch(
    const int* planet_seeds,
    const float* p_values,
    const int* bonus_cases,
    const int* is_birth_stars,
    const int* vein_spot_lens,
    const int* rare_vein_lens,
    const int* vein_spot_values,
    int vein_spot_stride,
    const int* rare_vein_values,
    int rare_vein_stride,
    const float* rare_settings_values,
    int rare_settings_stride,
    int planet_count,
    int out_vein_len,
    int use_fp32_prob_compare,
    int device_id,
    int* out_counts)
{
    if (planet_seeds == nullptr || p_values == nullptr || bonus_cases == nullptr || is_birth_stars == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (vein_spot_lens == nullptr || rare_vein_lens == nullptr || vein_spot_values == nullptr || rare_vein_values == nullptr || rare_settings_values == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (out_counts == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (planet_count <= 0 || out_vein_len <= 0)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (vein_spot_stride <= 0 || rare_vein_stride <= 0 || rare_settings_stride <= 0)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;

    if (device_id >= 0)
    {
        cudaError_t set_device_rc = cudaSetDevice(device_id);
        if (set_device_rc != cudaSuccess)
            return DSP_CUDA_ERR_CUDA;
    }

    const size_t n = static_cast<size_t>(planet_count);
    const size_t seeds_bytes = n * sizeof(int);
    const size_t p_values_bytes = n * sizeof(float);
    const size_t bonus_bytes = n * sizeof(int);
    const size_t birth_bytes = n * sizeof(int);
    const size_t vlen_bytes = n * sizeof(int);
    const size_t rlen_bytes = n * sizeof(int);
    const size_t vspot_bytes = n * static_cast<size_t>(vein_spot_stride) * sizeof(int);
    const size_t rvein_bytes = n * static_cast<size_t>(rare_vein_stride) * sizeof(int);
    const size_t rset_bytes = n * static_cast<size_t>(rare_settings_stride) * sizeof(float);
    const size_t out_bytes = n * static_cast<size_t>(out_vein_len) * sizeof(int);

    int* d_planet_seeds = nullptr;
    float* d_p_values = nullptr;
    int* d_bonus_cases = nullptr;
    int* d_is_birth_stars = nullptr;
    int* d_vein_spot_lens = nullptr;
    int* d_rare_vein_lens = nullptr;
    int* d_vein_spot_values = nullptr;
    int* d_rare_vein_values = nullptr;
    float* d_rare_settings_values = nullptr;
    int* d_out_counts = nullptr;

    auto fail = [&]() {
        cudaFree(d_out_counts);
        cudaFree(d_rare_settings_values);
        cudaFree(d_rare_vein_values);
        cudaFree(d_vein_spot_values);
        cudaFree(d_rare_vein_lens);
        cudaFree(d_vein_spot_lens);
        cudaFree(d_is_birth_stars);
        cudaFree(d_bonus_cases);
        cudaFree(d_p_values);
        cudaFree(d_planet_seeds);
        return DSP_CUDA_ERR_CUDA;
    };

    cudaError_t rc = cudaMalloc(reinterpret_cast<void**>(&d_planet_seeds), seeds_bytes);
    if (rc != cudaSuccess)
        return DSP_CUDA_ERR_CUDA;
    rc = cudaMalloc(reinterpret_cast<void**>(&d_p_values), p_values_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_bonus_cases), bonus_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_is_birth_stars), birth_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_vein_spot_lens), vlen_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_rare_vein_lens), rlen_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_vein_spot_values), vspot_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_rare_vein_values), rvein_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_rare_settings_values), rset_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_out_counts), out_bytes);
    if (rc != cudaSuccess)
        return fail();

    rc = cudaMemcpy(d_planet_seeds, planet_seeds, seeds_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_p_values, p_values, p_values_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_bonus_cases, bonus_cases, bonus_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_is_birth_stars, is_birth_stars, birth_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_vein_spot_lens, vein_spot_lens, vlen_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_rare_vein_lens, rare_vein_lens, rlen_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_vein_spot_values, vein_spot_values, vspot_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_rare_vein_values, rare_vein_values, rvein_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_rare_settings_values, rare_settings_values, rset_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();

    int block_size = 128;
    int grid_size = (planet_count + block_size - 1) / block_size;
    RefreshPlanetVeinSpotsBatchKernel<<<grid_size, block_size>>>(
        d_planet_seeds,
        d_p_values,
        d_bonus_cases,
        d_is_birth_stars,
        d_vein_spot_lens,
        d_rare_vein_lens,
        d_vein_spot_values,
        vein_spot_stride,
        d_rare_vein_values,
        rare_vein_stride,
        d_rare_settings_values,
        rare_settings_stride,
        planet_count,
        out_vein_len,
        use_fp32_prob_compare,
        d_out_counts);

    rc = cudaGetLastError();
    if (rc == cudaSuccess)
        rc = cudaMemcpy(out_counts, d_out_counts, out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_out_counts);
    cudaFree(d_rare_settings_values);
    cudaFree(d_rare_vein_values);
    cudaFree(d_vein_spot_values);
    cudaFree(d_rare_vein_lens);
    cudaFree(d_vein_spot_lens);
    cudaFree(d_is_birth_stars);
    cudaFree(d_bonus_cases);
    cudaFree(d_p_values);
    cudaFree(d_planet_seeds);

    if (rc != cudaSuccess)
        return DSP_CUDA_ERR_CUDA;
    return DSP_CUDA_OK;
}

extern "C" int dsp_cuda_refresh_planet_vein_spots_by_theme_batch(
    const int* planet_seeds,
    const float* p_values,
    const int* bonus_cases,
    const int* is_birth_stars,
    const int* theme_indexes,
    int planet_count,
    int out_vein_len,
    int use_fp32_prob_compare,
    int device_id,
    const int* theme_vein_spot_offsets,
    const int* theme_vein_spot_values,
    const int* theme_rare_vein_offsets,
    const int* theme_rare_vein_values,
    const int* theme_rare_settings_offsets,
    const float* theme_rare_settings_values,
    int theme_count,
    int* out_counts)
{
    if (planet_seeds == nullptr || p_values == nullptr || bonus_cases == nullptr || is_birth_stars == nullptr || theme_indexes == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (theme_vein_spot_offsets == nullptr ||
        theme_rare_vein_offsets == nullptr ||
        theme_rare_settings_offsets == nullptr)
    {
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    }
    if (out_counts == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (planet_count <= 0 || out_vein_len <= 0 || theme_count <= 0)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;

    const int theme_vein_total = theme_vein_spot_offsets[theme_count];
    const int theme_rare_total = theme_rare_vein_offsets[theme_count];
    const int theme_settings_total = theme_rare_settings_offsets[theme_count];
    if (theme_vein_total < 0 || theme_rare_total < 0 || theme_settings_total < 0)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (theme_vein_total > 0 && theme_vein_spot_values == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (theme_rare_total > 0 && theme_rare_vein_values == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (theme_settings_total > 0 && theme_rare_settings_values == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;

    if (device_id >= 0)
    {
        cudaError_t set_device_rc = cudaSetDevice(device_id);
        if (set_device_rc != cudaSuccess)
            return DSP_CUDA_ERR_CUDA;
    }

    const size_t n = static_cast<size_t>(planet_count);
    const size_t seeds_bytes = n * sizeof(int);
    const size_t p_values_bytes = n * sizeof(float);
    const size_t bonus_bytes = n * sizeof(int);
    const size_t birth_bytes = n * sizeof(int);
    const size_t theme_idx_bytes = n * sizeof(int);
    const size_t out_bytes = n * static_cast<size_t>(out_vein_len) * sizeof(int);

    const size_t theme_offsets_bytes = static_cast<size_t>(theme_count + 1) * sizeof(int);
    const size_t theme_vein_values_bytes = static_cast<size_t>(theme_vein_total) * sizeof(int);
    const size_t theme_rare_values_bytes = static_cast<size_t>(theme_rare_total) * sizeof(int);
    const size_t theme_settings_values_bytes = static_cast<size_t>(theme_settings_total) * sizeof(float);

    int* d_planet_seeds = nullptr;
    float* d_p_values = nullptr;
    int* d_bonus_cases = nullptr;
    int* d_is_birth_stars = nullptr;
    int* d_theme_indexes = nullptr;
    int* d_theme_vein_offsets = nullptr;
    int* d_theme_vein_values = nullptr;
    int* d_theme_rare_offsets = nullptr;
    int* d_theme_rare_values = nullptr;
    int* d_theme_settings_offsets = nullptr;
    float* d_theme_settings_values = nullptr;
    int* d_out_counts = nullptr;

    auto fail = [&]() {
        cudaFree(d_out_counts);
        cudaFree(d_theme_settings_values);
        cudaFree(d_theme_settings_offsets);
        cudaFree(d_theme_rare_values);
        cudaFree(d_theme_rare_offsets);
        cudaFree(d_theme_vein_values);
        cudaFree(d_theme_vein_offsets);
        cudaFree(d_theme_indexes);
        cudaFree(d_is_birth_stars);
        cudaFree(d_bonus_cases);
        cudaFree(d_p_values);
        cudaFree(d_planet_seeds);
        return DSP_CUDA_ERR_CUDA;
    };

    cudaError_t rc = cudaMalloc(reinterpret_cast<void**>(&d_planet_seeds), seeds_bytes);
    if (rc != cudaSuccess) return DSP_CUDA_ERR_CUDA;
    rc = cudaMalloc(reinterpret_cast<void**>(&d_p_values), p_values_bytes);
    if (rc != cudaSuccess) return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_bonus_cases), bonus_bytes);
    if (rc != cudaSuccess) return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_is_birth_stars), birth_bytes);
    if (rc != cudaSuccess) return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_theme_indexes), theme_idx_bytes);
    if (rc != cudaSuccess) return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_theme_vein_offsets), theme_offsets_bytes);
    if (rc != cudaSuccess) return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_theme_vein_values), theme_vein_values_bytes > 0 ? theme_vein_values_bytes : sizeof(int));
    if (rc != cudaSuccess) return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_theme_rare_offsets), theme_offsets_bytes);
    if (rc != cudaSuccess) return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_theme_rare_values), theme_rare_values_bytes > 0 ? theme_rare_values_bytes : sizeof(int));
    if (rc != cudaSuccess) return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_theme_settings_offsets), theme_offsets_bytes);
    if (rc != cudaSuccess) return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_theme_settings_values), theme_settings_values_bytes > 0 ? theme_settings_values_bytes : sizeof(float));
    if (rc != cudaSuccess) return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_out_counts), out_bytes);
    if (rc != cudaSuccess) return fail();

    rc = cudaMemcpy(d_planet_seeds, planet_seeds, seeds_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return fail();
    rc = cudaMemcpy(d_p_values, p_values, p_values_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return fail();
    rc = cudaMemcpy(d_bonus_cases, bonus_cases, bonus_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return fail();
    rc = cudaMemcpy(d_is_birth_stars, is_birth_stars, birth_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return fail();
    rc = cudaMemcpy(d_theme_indexes, theme_indexes, theme_idx_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return fail();
    rc = cudaMemcpy(d_theme_vein_offsets, theme_vein_spot_offsets, theme_offsets_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return fail();
    if (theme_vein_values_bytes > 0)
    {
        rc = cudaMemcpy(d_theme_vein_values, theme_vein_spot_values, theme_vein_values_bytes, cudaMemcpyHostToDevice);
        if (rc != cudaSuccess) return fail();
    }
    rc = cudaMemcpy(d_theme_rare_offsets, theme_rare_vein_offsets, theme_offsets_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return fail();
    if (theme_rare_values_bytes > 0)
    {
        rc = cudaMemcpy(d_theme_rare_values, theme_rare_vein_values, theme_rare_values_bytes, cudaMemcpyHostToDevice);
        if (rc != cudaSuccess) return fail();
    }
    rc = cudaMemcpy(d_theme_settings_offsets, theme_rare_settings_offsets, theme_offsets_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return fail();
    if (theme_settings_values_bytes > 0)
    {
        rc = cudaMemcpy(d_theme_settings_values, theme_rare_settings_values, theme_settings_values_bytes, cudaMemcpyHostToDevice);
        if (rc != cudaSuccess) return fail();
    }

    int block_size = 128;
    int grid_size = (planet_count + block_size - 1) / block_size;
    RefreshPlanetVeinSpotsByThemeBatchKernel<<<grid_size, block_size>>>(
        d_planet_seeds,
        d_p_values,
        d_bonus_cases,
        d_is_birth_stars,
        d_theme_indexes,
        planet_count,
        out_vein_len,
        use_fp32_prob_compare,
        d_theme_vein_offsets,
        d_theme_vein_values,
        d_theme_rare_offsets,
        d_theme_rare_values,
        d_theme_settings_offsets,
        d_theme_settings_values,
        theme_count,
        d_out_counts);

    rc = cudaGetLastError();
    if (rc == cudaSuccess)
        rc = cudaMemcpy(out_counts, d_out_counts, out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_out_counts);
    cudaFree(d_theme_settings_values);
    cudaFree(d_theme_settings_offsets);
    cudaFree(d_theme_rare_values);
    cudaFree(d_theme_rare_offsets);
    cudaFree(d_theme_vein_values);
    cudaFree(d_theme_vein_offsets);
    cudaFree(d_theme_indexes);
    cudaFree(d_is_birth_stars);
    cudaFree(d_bonus_cases);
    cudaFree(d_p_values);
    cudaFree(d_planet_seeds);

    if (rc != cudaSuccess)
        return DSP_CUDA_ERR_CUDA;
    return DSP_CUDA_OK;
}

extern "C" int dsp_cuda_planet_eval_core_f32(
    int info_seed,
    int orbit_around,
    int orbit_index,
    int gas_giant,
    int star_index,
    int galaxy_star_count,
    int galaxy_habitable_count,
    int boost_inclination_ns,
    int compact_type_case,
    float star_orbit_scaler,
    double star_mass,
    float star_habitable_radius,
    float star_light_balance_radius,
    float orbit_around_planet_real_radius,
    float orbit_around_planet_orbit_radius,
    double orbit_around_planet_orbital_period,
    int device_id,
    dsp_planet_core_f32_out_t* out_result)
{
    if (out_result == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;

    if (device_id >= 0)
    {
        cudaError_t set_device_rc = cudaSetDevice(device_id);
        if (set_device_rc != cudaSuccess)
            return DSP_CUDA_ERR_CUDA;
    }

    dsp_planet_core_f32_out_t* d_out = nullptr;
    cudaError_t rc = cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(dsp_planet_core_f32_out_t));
    if (rc != cudaSuccess)
        return DSP_CUDA_ERR_CUDA;

    EvalPlanetCoreF32Kernel<<<1, 1>>>(
        info_seed,
        orbit_around,
        orbit_index,
        gas_giant,
        star_index,
        galaxy_star_count,
        galaxy_habitable_count,
        boost_inclination_ns,
        compact_type_case,
        star_orbit_scaler,
        star_mass,
        star_habitable_radius,
        star_light_balance_radius,
        orbit_around_planet_real_radius,
        orbit_around_planet_orbit_radius,
        orbit_around_planet_orbital_period,
        d_out);

    rc = cudaGetLastError();
    if (rc == cudaSuccess)
        rc = cudaMemcpy(out_result, d_out, sizeof(dsp_planet_core_f32_out_t), cudaMemcpyDeviceToHost);

    cudaFree(d_out);

    if (rc != cudaSuccess)
        return DSP_CUDA_ERR_CUDA;
    return DSP_CUDA_OK;
}

extern "C" int dsp_cuda_planet_eval_core_f32_batch(
    const int* info_seeds,
    const int* orbit_arounds,
    const int* orbit_indexes,
    const int* gas_giants,
    const int* star_indexes,
    const int* galaxy_star_counts,
    const int* galaxy_habitable_counts,
    const int* boost_inclination_ns,
    const int* compact_type_cases,
    const float* star_orbit_scalers,
    const double* star_masses,
    const float* star_habitable_radiuses,
    const float* star_light_balance_radiuses,
    const float* orbit_around_planet_real_radiuses,
    const float* orbit_around_planet_orbit_radiuses,
    const double* orbit_around_planet_orbital_periods,
    int batch_count,
    int device_id,
    dsp_planet_core_f32_out_t* out_results)
{
    if (info_seeds == nullptr || orbit_arounds == nullptr || orbit_indexes == nullptr || gas_giants == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (star_indexes == nullptr || galaxy_star_counts == nullptr || galaxy_habitable_counts == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (boost_inclination_ns == nullptr || compact_type_cases == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (star_orbit_scalers == nullptr || star_masses == nullptr || star_habitable_radiuses == nullptr || star_light_balance_radiuses == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (orbit_around_planet_real_radiuses == nullptr || orbit_around_planet_orbit_radiuses == nullptr || orbit_around_planet_orbital_periods == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (out_results == nullptr || batch_count <= 0)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;

    if (device_id >= 0)
    {
        cudaError_t set_device_rc = cudaSetDevice(device_id);
        if (set_device_rc != cudaSuccess)
            return DSP_CUDA_ERR_CUDA;
    }

    const size_t n = static_cast<size_t>(batch_count);
    const size_t ints_bytes = n * sizeof(int);
    const size_t floats_bytes = n * sizeof(float);
    const size_t doubles_bytes = n * sizeof(double);
    const size_t out_bytes = n * sizeof(dsp_planet_core_f32_out_t);

    PlanetCoreBatchDeviceBuffers* buffers = nullptr;
    cudaError_t rc = EnsurePlanetCoreBatchBuffers(
        device_id,
        ints_bytes,
        floats_bytes,
        doubles_bytes,
        out_bytes,
        &buffers);
    if (rc != cudaSuccess || buffers == nullptr)
        return DSP_CUDA_ERR_CUDA;

    rc = cudaMemcpy(buffers->d_info_seeds, info_seeds, ints_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return DSP_CUDA_ERR_CUDA;
    rc = cudaMemcpy(buffers->d_orbit_arounds, orbit_arounds, ints_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return DSP_CUDA_ERR_CUDA;
    rc = cudaMemcpy(buffers->d_orbit_indexes, orbit_indexes, ints_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return DSP_CUDA_ERR_CUDA;
    rc = cudaMemcpy(buffers->d_gas_giants, gas_giants, ints_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return DSP_CUDA_ERR_CUDA;
    rc = cudaMemcpy(buffers->d_star_indexes, star_indexes, ints_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return DSP_CUDA_ERR_CUDA;
    rc = cudaMemcpy(buffers->d_galaxy_star_counts, galaxy_star_counts, ints_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return DSP_CUDA_ERR_CUDA;
    rc = cudaMemcpy(buffers->d_galaxy_habitable_counts, galaxy_habitable_counts, ints_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return DSP_CUDA_ERR_CUDA;
    rc = cudaMemcpy(buffers->d_boost_inclination_ns, boost_inclination_ns, ints_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return DSP_CUDA_ERR_CUDA;
    rc = cudaMemcpy(buffers->d_compact_type_cases, compact_type_cases, ints_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return DSP_CUDA_ERR_CUDA;
    rc = cudaMemcpy(buffers->d_star_orbit_scalers, star_orbit_scalers, floats_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return DSP_CUDA_ERR_CUDA;
    rc = cudaMemcpy(buffers->d_star_masses, star_masses, doubles_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return DSP_CUDA_ERR_CUDA;
    rc = cudaMemcpy(buffers->d_star_habitable_radiuses, star_habitable_radiuses, floats_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return DSP_CUDA_ERR_CUDA;
    rc = cudaMemcpy(buffers->d_star_light_balance_radiuses, star_light_balance_radiuses, floats_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return DSP_CUDA_ERR_CUDA;
    rc = cudaMemcpy(buffers->d_orbit_around_planet_real_radiuses, orbit_around_planet_real_radiuses, floats_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return DSP_CUDA_ERR_CUDA;
    rc = cudaMemcpy(buffers->d_orbit_around_planet_orbit_radiuses, orbit_around_planet_orbit_radiuses, floats_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return DSP_CUDA_ERR_CUDA;
    rc = cudaMemcpy(buffers->d_orbit_around_planet_orbital_periods, orbit_around_planet_orbital_periods, doubles_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess) return DSP_CUDA_ERR_CUDA;

    int block_size = 128;
    int grid_size = (batch_count + block_size - 1) / block_size;
    EvalPlanetCoreF32BatchKernel<<<grid_size, block_size>>>(
        buffers->d_info_seeds,
        buffers->d_orbit_arounds,
        buffers->d_orbit_indexes,
        buffers->d_gas_giants,
        buffers->d_star_indexes,
        buffers->d_galaxy_star_counts,
        buffers->d_galaxy_habitable_counts,
        buffers->d_boost_inclination_ns,
        buffers->d_compact_type_cases,
        buffers->d_star_orbit_scalers,
        buffers->d_star_masses,
        buffers->d_star_habitable_radiuses,
        buffers->d_star_light_balance_radiuses,
        buffers->d_orbit_around_planet_real_radiuses,
        buffers->d_orbit_around_planet_orbit_radiuses,
        buffers->d_orbit_around_planet_orbital_periods,
        batch_count,
        buffers->d_out_results);

    rc = cudaGetLastError();
    if (rc == cudaSuccess)
        rc = cudaMemcpy(out_results, buffers->d_out_results, out_bytes, cudaMemcpyDeviceToHost);

    if (rc != cudaSuccess)
        return DSP_CUDA_ERR_CUDA;
    return DSP_CUDA_OK;
}

extern "C" int dsp_cuda_planet_eval_gas_details_f32(
    int theme_seed,
    float gas_coef,
    float resource_coef,
    const float* in_gas_speeds,
    const float* in_gas_heat_values,
    int gas_len,
    int device_id,
    float* out_gas_speeds,
    double* out_total_heat)
{
    if (in_gas_speeds == nullptr || in_gas_heat_values == nullptr || out_gas_speeds == nullptr || out_total_heat == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (gas_len <= 0)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;

    if (device_id >= 0)
    {
        cudaError_t set_device_rc = cudaSetDevice(device_id);
        if (set_device_rc != cudaSuccess)
            return DSP_CUDA_ERR_CUDA;
    }

    size_t bytes = static_cast<size_t>(gas_len) * sizeof(float);
    float* d_in_speeds = nullptr;
    float* d_in_heat = nullptr;
    float* d_out_speeds = nullptr;
    double* d_out_total = nullptr;

    auto fail = [&]() {
        cudaFree(d_out_total);
        cudaFree(d_out_speeds);
        cudaFree(d_in_heat);
        cudaFree(d_in_speeds);
        return DSP_CUDA_ERR_CUDA;
    };

    cudaError_t rc = cudaMalloc(reinterpret_cast<void**>(&d_in_speeds), bytes);
    if (rc != cudaSuccess)
        return DSP_CUDA_ERR_CUDA;
    rc = cudaMalloc(reinterpret_cast<void**>(&d_in_heat), bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_out_speeds), bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_out_total), sizeof(double));
    if (rc != cudaSuccess)
        return fail();

    rc = cudaMemcpy(d_in_speeds, in_gas_speeds, bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_in_heat, in_gas_heat_values, bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();

    EvalPlanetGasDetailsF32Kernel<<<1, 1>>>(
        theme_seed,
        gas_coef,
        resource_coef,
        d_in_speeds,
        d_in_heat,
        gas_len,
        d_out_speeds,
        d_out_total);

    rc = cudaGetLastError();
    if (rc == cudaSuccess)
        rc = cudaMemcpy(out_gas_speeds, d_out_speeds, bytes, cudaMemcpyDeviceToHost);
    if (rc == cudaSuccess)
        rc = cudaMemcpy(out_total_heat, d_out_total, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_out_total);
    cudaFree(d_out_speeds);
    cudaFree(d_in_heat);
    cudaFree(d_in_speeds);

    if (rc != cudaSuccess)
        return DSP_CUDA_ERR_CUDA;
    return DSP_CUDA_OK;
}

extern "C" int dsp_cuda_mix_chunk_eval_planets_f32(
    const int* info_seeds,
    const int* orbit_arounds,
    const int* orbit_indexes,
    const int* gas_giants,
    const int* star_indexes,
    const int* galaxy_star_counts,
    const int* galaxy_habitable_counts,
    const int* boost_inclination_ns,
    const int* compact_type_cases,
    const float* star_orbit_scalers,
    const double* star_masses,
    const float* star_habitable_radiuses,
    const float* star_light_balance_radiuses,
    const float* orbit_around_planet_real_radiuses,
    const float* orbit_around_planet_orbit_radiuses,
    const double* orbit_around_planet_orbital_periods,
    int batch_count,
    int device_id,
    dsp_planet_core_f32_out_t* out_results)
{
    return dsp_cuda_planet_eval_core_f32_batch(
        info_seeds,
        orbit_arounds,
        orbit_indexes,
        gas_giants,
        star_indexes,
        galaxy_star_counts,
        galaxy_habitable_counts,
        boost_inclination_ns,
        compact_type_cases,
        star_orbit_scalers,
        star_masses,
        star_habitable_radiuses,
        star_light_balance_radiuses,
        orbit_around_planet_real_radiuses,
        orbit_around_planet_orbit_radiuses,
        orbit_around_planet_orbital_periods,
        batch_count,
        device_id,
        out_results);
}

extern "C" int dsp_cuda_mix_chunk_eval_veins_f32(
    const int* planet_seeds,
    const float* p_values,
    const int* bonus_cases,
    const int* is_birth_stars,
    const int* vein_spot_lens,
    const int* rare_vein_lens,
    const int* vein_spot_values,
    int vein_spot_stride,
    const int* rare_vein_values,
    int rare_vein_stride,
    const float* rare_settings_values,
    int rare_settings_stride,
    int planet_count,
    int out_vein_len,
    int use_fp32_prob_compare,
    int device_id,
    int* out_counts)
{
    return dsp_cuda_refresh_planet_vein_spots_batch(
        planet_seeds,
        p_values,
        bonus_cases,
        is_birth_stars,
        vein_spot_lens,
        rare_vein_lens,
        vein_spot_values,
        vein_spot_stride,
        rare_vein_values,
        rare_vein_stride,
        rare_settings_values,
        rare_settings_stride,
        planet_count,
        out_vein_len,
        use_fp32_prob_compare,
        device_id,
        out_counts);
}

extern "C" int dsp_cuda_mix_chunk_eval_veins_by_theme_f32(
    const int* planet_seeds,
    const float* p_values,
    const int* bonus_cases,
    const int* is_birth_stars,
    const int* theme_indexes,
    int planet_count,
    int out_vein_len,
    int use_fp32_prob_compare,
    int device_id,
    const int* theme_vein_spot_offsets,
    const int* theme_vein_spot_values,
    const int* theme_rare_vein_offsets,
    const int* theme_rare_vein_values,
    const int* theme_rare_settings_offsets,
    const float* theme_rare_settings_values,
    int theme_count,
    int* out_counts)
{
    return dsp_cuda_refresh_planet_vein_spots_by_theme_batch(
        planet_seeds,
        p_values,
        bonus_cases,
        is_birth_stars,
        theme_indexes,
        planet_count,
        out_vein_len,
        use_fp32_prob_compare,
        device_id,
        theme_vein_spot_offsets,
        theme_vein_spot_values,
        theme_rare_vein_offsets,
        theme_rare_vein_values,
        theme_rare_settings_offsets,
        theme_rare_settings_values,
        theme_count,
        out_counts);
}

extern "C" int dsp_cuda_mix_chunk_reduce_signatures(
    int seed_count,
    const int* seed_star_offsets,
    const int* seed_planet_offsets,
    const int* galaxy_star_counts,
    const int* birth_star_ids,
    const int* birth_planet_ids,
    const int* star_ids,
    const int* star_types,
    const int* star_spectrs,
    const int* star_planet_counts,
    const int* star_planet_loop_counts,
    const int* star_pos_x,
    const int* star_pos_y,
    const int* star_pos_z,
    const int* planet_ids,
    const int* planet_types,
    const int* planet_themes,
    const int* planet_water_item_ids,
    const int* planet_orbit_indexes,
    const int* planet_orbit_arounds,
    const int* planet_is_null,
    const int* planet_is_gas,
    const int* planet_vein_offsets,
    const int* vein_counts_flat,
    int vein_stride,
    unsigned long long* out_galaxy_sigs,
    unsigned long long* out_planet_sigs,
    unsigned long long* out_vein_sigs,
    unsigned long long* out_pipeline_sigs)
{
    if (seed_count <= 0)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (seed_star_offsets == nullptr || seed_planet_offsets == nullptr ||
        galaxy_star_counts == nullptr || birth_star_ids == nullptr || birth_planet_ids == nullptr ||
        star_ids == nullptr || star_types == nullptr || star_spectrs == nullptr || star_planet_counts == nullptr || star_planet_loop_counts == nullptr ||
        star_pos_x == nullptr || star_pos_y == nullptr || star_pos_z == nullptr ||
        planet_ids == nullptr || planet_types == nullptr || planet_themes == nullptr ||
        planet_water_item_ids == nullptr || planet_orbit_indexes == nullptr || planet_orbit_arounds == nullptr ||
        planet_is_null == nullptr || planet_is_gas == nullptr || planet_vein_offsets == nullptr ||
        out_galaxy_sigs == nullptr || out_planet_sigs == nullptr || out_vein_sigs == nullptr || out_pipeline_sigs == nullptr)
    {
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    }

    constexpr unsigned long long kFnvOffset = 14695981039346656037ull;
    constexpr unsigned long long kFnvPrime = 1099511628211ull;

    auto mix = [&](unsigned long long h, int v) -> unsigned long long
    {
        h ^= static_cast<unsigned int>(v);
        h *= kFnvPrime;
        return h;
    };

    for (int si = 0; si < seed_count; ++si)
    {
        int star_begin = seed_star_offsets[si];
        int star_end = seed_star_offsets[si + 1];
        int planet_begin = seed_planet_offsets[si];
        int planet_end = seed_planet_offsets[si + 1];
        if (star_end < star_begin || planet_end < planet_begin)
            return DSP_CUDA_ERR_INVALID_ARGUMENT;

        unsigned long long h_galaxy = kFnvOffset;
        unsigned long long h_planet = kFnvOffset;
        unsigned long long h_vein = kFnvOffset;
        unsigned long long h_pipeline = kFnvOffset;

        int star_count = galaxy_star_counts[si];
        int birth_star_id = birth_star_ids[si];
        int birth_planet_id = birth_planet_ids[si];

        h_galaxy = mix(h_galaxy, star_count);

        h_planet = mix(h_planet, star_count);
        h_planet = mix(h_planet, birth_star_id);
        h_planet = mix(h_planet, birth_planet_id);

        h_vein = mix(h_vein, star_count);

        h_pipeline = mix(h_pipeline, star_count);
        h_pipeline = mix(h_pipeline, birth_star_id);
        h_pipeline = mix(h_pipeline, birth_planet_id);

        int p_cur = planet_begin;
        for (int st = star_begin; st < star_end; ++st)
        {
            const int sid = star_ids[st];
            const int stype = star_types[st];
            const int sspectr = star_spectrs[st];
            const int spcnt = star_planet_counts[st];
            const int spcnt_loop = star_planet_loop_counts[st];
            const int sx = star_pos_x[st];
            const int sy = star_pos_y[st];
            const int sz = star_pos_z[st];

            h_galaxy = mix(h_galaxy, sid);
            h_galaxy = mix(h_galaxy, stype);
            h_galaxy = mix(h_galaxy, sspectr);
            h_galaxy = mix(h_galaxy, spcnt);
            h_galaxy = mix(h_galaxy, sx);
            h_galaxy = mix(h_galaxy, sy);
            h_galaxy = mix(h_galaxy, sz);

            h_planet = mix(h_planet, sid);
            h_planet = mix(h_planet, stype);
            h_planet = mix(h_planet, sspectr);
            h_planet = mix(h_planet, spcnt);
            h_planet = mix(h_planet, sx);
            h_planet = mix(h_planet, sy);
            h_planet = mix(h_planet, sz);

            h_vein = mix(h_vein, sid);

            h_pipeline = mix(h_pipeline, sid);
            h_pipeline = mix(h_pipeline, stype);
            h_pipeline = mix(h_pipeline, sspectr);
            h_pipeline = mix(h_pipeline, spcnt);
            h_pipeline = mix(h_pipeline, sx);
            h_pipeline = mix(h_pipeline, sy);
            h_pipeline = mix(h_pipeline, sz);

            int loop_pcnt = spcnt_loop;
            if (loop_pcnt < 0)
                loop_pcnt = 0;
            for (int pi = 0; pi < loop_pcnt; ++pi)
            {
                if (p_cur >= planet_end)
                    return DSP_CUDA_ERR_INVALID_ARGUMENT;

                if (planet_is_null[p_cur] != 0)
                {
                    h_planet = mix(h_planet, -2);
                    h_vein = mix(h_vein, -2);
                    h_pipeline = mix(h_pipeline, -2);
                    ++p_cur;
                    continue;
                }

                const int pid = planet_ids[p_cur];
                const int ptype = planet_types[p_cur];
                const int ptheme = planet_themes[p_cur];
                const int pwater = planet_water_item_ids[p_cur];
                const int porbit_index = planet_orbit_indexes[p_cur];
                const int porbit_around = planet_orbit_arounds[p_cur];

                h_planet = mix(h_planet, ptype);
                h_planet = mix(h_planet, ptheme);
                h_planet = mix(h_planet, pwater);
                h_planet = mix(h_planet, porbit_index);
                h_planet = mix(h_planet, porbit_around);

                h_vein = mix(h_vein, pid);

                h_pipeline = mix(h_pipeline, ptype);
                h_pipeline = mix(h_pipeline, ptheme);
                h_pipeline = mix(h_pipeline, pwater);
                h_pipeline = mix(h_pipeline, porbit_index);
                h_pipeline = mix(h_pipeline, porbit_around);

                if (planet_is_gas[p_cur] != 0)
                {
                    h_vein = mix(h_vein, 0);
                    ++p_cur;
                    continue;
                }

                int vbase = planet_vein_offsets[p_cur];
                if (vbase < 0 || vein_counts_flat == nullptr || vein_stride <= 1)
                {
                    h_vein = mix(h_vein, -4);
                    h_pipeline = mix(h_pipeline, -4);
                    ++p_cur;
                    continue;
                }

                int vmax = vein_stride < 32 ? vein_stride : 32;
                for (int vid = 1; vid < vmax; ++vid)
                {
                    int vv = vein_counts_flat[vbase * vein_stride + vid];
                    h_vein = mix(h_vein, vv);
                    h_pipeline = mix(h_pipeline, vv);
                }

                ++p_cur;
            }
        }

        if (p_cur != planet_end)
            return DSP_CUDA_ERR_INVALID_ARGUMENT;

        out_galaxy_sigs[si] = h_galaxy;
        out_planet_sigs[si] = h_planet;
        out_vein_sigs[si] = h_vein;
        out_pipeline_sigs[si] = h_pipeline;
    }

    return DSP_CUDA_OK;
}
