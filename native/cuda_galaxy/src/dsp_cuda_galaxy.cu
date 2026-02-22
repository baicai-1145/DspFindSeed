#include "dsp_cuda_galaxy.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace
{
constexpr int kMBig = 2147483647;
constexpr int kMSeed = 161803398;

struct BatchDeviceBuffers
{
    int device_id;
    int* d_seeds;
    int* d_counts;
    dsp_vec3d_t* d_poses;
    dsp_vec3d_t* d_drunk;
    size_t seeds_capacity_bytes;
    size_t counts_capacity_bytes;
    size_t poses_capacity_bytes;
    size_t drunk_capacity_bytes;

    BatchDeviceBuffers()
        : device_id(-2),
          d_seeds(nullptr),
          d_counts(nullptr),
          d_poses(nullptr),
          d_drunk(nullptr),
          seeds_capacity_bytes(0),
          counts_capacity_bytes(0),
          poses_capacity_bytes(0),
          drunk_capacity_bytes(0)
    {
    }
};

thread_local BatchDeviceBuffers g_batch_buffers;

void ReleaseBatchDeviceBuffers(BatchDeviceBuffers& buffers)
{
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
    int** d_seeds,
    int** d_counts,
    dsp_vec3d_t** d_poses,
    dsp_vec3d_t** d_drunk)
{
    BatchDeviceBuffers& buffers = g_batch_buffers;
    if (buffers.device_id != device_id)
    {
        ReleaseBatchDeviceBuffers(buffers);
        buffers.device_id = device_id;
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

    *d_seeds = buffers.d_seeds;
    *d_counts = buffers.d_counts;
    *d_poses = buffers.d_poses;
    *d_drunk = buffers.d_drunk;
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

__device__ bool CheckCollisionF32(const dsp_vec3d_t* pts, int count, const dsp_vec3d_t& pt, double min_dist)
{
    float min_dist_f = static_cast<float>(min_dist);
    double min2 = static_cast<double>(min_dist_f) * static_cast<double>(min_dist_f);
    for (int i = 0; i < count; ++i)
    {
        const dsp_vec3d_t& p = pts[i];
        float dx = static_cast<float>(pt.x - p.x);
        float dy = static_cast<float>(pt.y - p.y);
        float dz = static_cast<float>(pt.z - p.z);
        double dist2 = static_cast<double>(dx) * static_cast<double>(dx) +
                       static_cast<double>(dy) * static_cast<double>(dy) +
                       static_cast<double>(dz) * static_cast<double>(dz);
        if (dist2 < min2)
            return true;
    }
    return false;
}

__device__ bool CheckCollisionFp64(const dsp_vec3d_t* pts, int count, const dsp_vec3d_t& pt, double min_dist)
{
    double min2 = min_dist * min_dist;
    for (int i = 0; i < count; ++i)
    {
        const dsp_vec3d_t& p = pts[i];
        double dx = pt.x - p.x;
        double dy = pt.y - p.y;
        double dz = pt.z - p.z;
        if (dx * dx + dy * dy + dz * dz < min2)
            return true;
    }
    return false;
}

__device__ bool CheckCollision(const dsp_vec3d_t* pts, int count, const dsp_vec3d_t& pt, double min_dist, bool collision_fp64)
{
    if (collision_fp64)
        return CheckCollisionFp64(pts, count, pt, min_dist);
    return CheckCollisionF32(pts, count, pt, min_dist);
}

__device__ int GenerateRandomPosesParamsFp64(
    int seed,
    int max_count,
    double min_dist,
    double min_step_len,
    double max_step_len,
    double flatten,
    bool collision_fp64,
    dsp_vec3d_t* poses,
    dsp_vec3d_t* drunk)
{
    DotNet35RandomDevice rng(seed);

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
            double xd = rng.NextDouble() * 2.0 - 1.0;
            double yd = (rng.NextDouble() * 2.0 - 1.0) * flatten;
            double zd = rng.NextDouble() * 2.0 - 1.0;
            double num10 = rng.NextDouble();

            double dd = xd * xd + yd * yd + zd * zd;
            if (dd <= 1.0 && dd >= 1e-8)
            {
                double inv_len = 1.0 / sqrt(dd);
                double scale = (num10 * (max_step_len - min_step_len) + min_dist) * inv_len;
                dsp_vec3d_t pt{xd * scale, yd * scale, zd * scale};

                if (!CheckCollision(poses, poses_count, pt, min_dist, collision_fp64))
                {
                    drunk[drunk_count++] = pt;
                    poses[poses_count++] = pt;
                    if (poses_count >= max_count)
                        return poses_count;
                    break;
                }
            }
        }
    }

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
                    double xd = rng.NextDouble() * 2.0 - 1.0;
                    double yd = (rng.NextDouble() * 2.0 - 1.0) * flatten;
                    double zd = rng.NextDouble() * 2.0 - 1.0;
                    double num18 = rng.NextDouble();

                    double dd = xd * xd + yd * yd + zd * zd;
                    if (dd <= 1.0 && dd >= 1e-8)
                    {
                        double inv_len = 1.0 / sqrt(dd);
                        double scale = (num18 * (max_step_len - min_step_len) + min_dist) * inv_len;
                        const dsp_vec3d_t base_pt = drunk[i];
                        dsp_vec3d_t pt{
                            base_pt.x + xd * scale,
                            base_pt.y + yd * scale,
                            base_pt.z + zd * scale};

                        if (!CheckCollision(poses, poses_count, pt, min_dist, collision_fp64))
                        {
                            drunk[i] = pt;
                            poses[poses_count++] = pt;
                            if (poses_count >= max_count)
                                return poses_count;
                            break;
                        }
                    }
                }
            }
        }
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
    int* out_counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seed_count)
        return;

    dsp_vec3d_t* poses_seg = out_poses + static_cast<long long>(idx) * out_stride;
    dsp_vec3d_t* drunk_seg = out_drunk + static_cast<long long>(idx) * out_stride;
    int count = GenerateRandomPosesParamsFp64(
        seeds[idx],
        max_count,
        min_dist,
        min_step_len,
        max_step_len,
        flatten,
        collision_fp64 != 0,
        poses_seg,
        drunk_seg);
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
    if (seeds == nullptr || out_poses == nullptr || out_counts == nullptr)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    if (seed_count <= 0 || max_count <= 0 || out_stride < max_count)
        return DSP_CUDA_ERR_INVALID_ARGUMENT;

    if (device_id >= 0)
    {
        cudaError_t set_device_rc = cudaSetDevice(device_id);
        if (set_device_rc != cudaSuccess)
            return DSP_CUDA_ERR_CUDA;
    }

    const size_t seeds_bytes = static_cast<size_t>(seed_count) * sizeof(int);
    const size_t counts_bytes = static_cast<size_t>(seed_count) * sizeof(int);
    const size_t poses_bytes = static_cast<size_t>(seed_count) * static_cast<size_t>(out_stride) * sizeof(dsp_vec3d_t);

    int* d_seeds = nullptr;
    int* d_counts = nullptr;
    dsp_vec3d_t* d_poses = nullptr;
    dsp_vec3d_t* d_drunk = nullptr;

    cudaError_t rc = EnsureBatchBuffers(
        device_id,
        seeds_bytes,
        counts_bytes,
        poses_bytes,
        &d_seeds,
        &d_counts,
        &d_poses,
        &d_drunk);
    if (rc != cudaSuccess)
        return DSP_CUDA_ERR_CUDA;

    rc = cudaMemcpy(d_seeds, seeds, seeds_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return DSP_CUDA_ERR_CUDA;

    int block_size = 128;
    int grid_size = (seed_count + block_size - 1) / block_size;
    GenerateTempPosesParamsFp64BatchKernel<<<grid_size, block_size>>>(
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
        d_counts);

    rc = cudaGetLastError();
    if (rc == cudaSuccess)
        rc = cudaMemcpy(out_counts, d_counts, counts_bytes, cudaMemcpyDeviceToHost);
    if (rc == cudaSuccess)
        rc = cudaMemcpy(out_poses, d_poses, poses_bytes, cudaMemcpyDeviceToHost);

    if (rc != cudaSuccess)
        return DSP_CUDA_ERR_CUDA;
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

    int* d_info_seeds = nullptr;
    int* d_orbit_arounds = nullptr;
    int* d_orbit_indexes = nullptr;
    int* d_gas_giants = nullptr;
    int* d_star_indexes = nullptr;
    int* d_galaxy_star_counts = nullptr;
    int* d_galaxy_habitable_counts = nullptr;
    int* d_boost_inclination_ns = nullptr;
    int* d_compact_type_cases = nullptr;
    float* d_star_orbit_scalers = nullptr;
    double* d_star_masses = nullptr;
    float* d_star_habitable_radiuses = nullptr;
    float* d_star_light_balance_radiuses = nullptr;
    float* d_orbit_around_planet_real_radiuses = nullptr;
    float* d_orbit_around_planet_orbit_radiuses = nullptr;
    double* d_orbit_around_planet_orbital_periods = nullptr;
    dsp_planet_core_f32_out_t* d_out_results = nullptr;

    auto fail = [&]() {
        cudaFree(d_out_results);
        cudaFree(d_orbit_around_planet_orbital_periods);
        cudaFree(d_orbit_around_planet_orbit_radiuses);
        cudaFree(d_orbit_around_planet_real_radiuses);
        cudaFree(d_star_light_balance_radiuses);
        cudaFree(d_star_habitable_radiuses);
        cudaFree(d_star_masses);
        cudaFree(d_star_orbit_scalers);
        cudaFree(d_compact_type_cases);
        cudaFree(d_boost_inclination_ns);
        cudaFree(d_galaxy_habitable_counts);
        cudaFree(d_galaxy_star_counts);
        cudaFree(d_star_indexes);
        cudaFree(d_gas_giants);
        cudaFree(d_orbit_indexes);
        cudaFree(d_orbit_arounds);
        cudaFree(d_info_seeds);
        return DSP_CUDA_ERR_CUDA;
    };

    cudaError_t rc = cudaMalloc(reinterpret_cast<void**>(&d_info_seeds), ints_bytes);
    if (rc != cudaSuccess)
        return DSP_CUDA_ERR_CUDA;
    rc = cudaMalloc(reinterpret_cast<void**>(&d_orbit_arounds), ints_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_orbit_indexes), ints_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_gas_giants), ints_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_star_indexes), ints_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_galaxy_star_counts), ints_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_galaxy_habitable_counts), ints_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_boost_inclination_ns), ints_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_compact_type_cases), ints_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_star_orbit_scalers), floats_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_star_masses), doubles_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_star_habitable_radiuses), floats_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_star_light_balance_radiuses), floats_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_orbit_around_planet_real_radiuses), floats_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_orbit_around_planet_orbit_radiuses), floats_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_orbit_around_planet_orbital_periods), doubles_bytes);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMalloc(reinterpret_cast<void**>(&d_out_results), out_bytes);
    if (rc != cudaSuccess)
        return fail();

    rc = cudaMemcpy(d_info_seeds, info_seeds, ints_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_orbit_arounds, orbit_arounds, ints_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_orbit_indexes, orbit_indexes, ints_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_gas_giants, gas_giants, ints_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_star_indexes, star_indexes, ints_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_galaxy_star_counts, galaxy_star_counts, ints_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_galaxy_habitable_counts, galaxy_habitable_counts, ints_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_boost_inclination_ns, boost_inclination_ns, ints_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_compact_type_cases, compact_type_cases, ints_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_star_orbit_scalers, star_orbit_scalers, floats_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_star_masses, star_masses, doubles_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_star_habitable_radiuses, star_habitable_radiuses, floats_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_star_light_balance_radiuses, star_light_balance_radiuses, floats_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_orbit_around_planet_real_radiuses, orbit_around_planet_real_radiuses, floats_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_orbit_around_planet_orbit_radiuses, orbit_around_planet_orbit_radiuses, floats_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();
    rc = cudaMemcpy(d_orbit_around_planet_orbital_periods, orbit_around_planet_orbital_periods, doubles_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
        return fail();

    int block_size = 128;
    int grid_size = (batch_count + block_size - 1) / block_size;
    EvalPlanetCoreF32BatchKernel<<<grid_size, block_size>>>(
        d_info_seeds,
        d_orbit_arounds,
        d_orbit_indexes,
        d_gas_giants,
        d_star_indexes,
        d_galaxy_star_counts,
        d_galaxy_habitable_counts,
        d_boost_inclination_ns,
        d_compact_type_cases,
        d_star_orbit_scalers,
        d_star_masses,
        d_star_habitable_radiuses,
        d_star_light_balance_radiuses,
        d_orbit_around_planet_real_radiuses,
        d_orbit_around_planet_orbit_radiuses,
        d_orbit_around_planet_orbital_periods,
        batch_count,
        d_out_results);

    rc = cudaGetLastError();
    if (rc == cudaSuccess)
        rc = cudaMemcpy(out_results, d_out_results, out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_out_results);
    cudaFree(d_orbit_around_planet_orbital_periods);
    cudaFree(d_orbit_around_planet_orbit_radiuses);
    cudaFree(d_orbit_around_planet_real_radiuses);
    cudaFree(d_star_light_balance_radiuses);
    cudaFree(d_star_habitable_radiuses);
    cudaFree(d_star_masses);
    cudaFree(d_star_orbit_scalers);
    cudaFree(d_compact_type_cases);
    cudaFree(d_boost_inclination_ns);
    cudaFree(d_galaxy_habitable_counts);
    cudaFree(d_galaxy_star_counts);
    cudaFree(d_star_indexes);
    cudaFree(d_gas_giants);
    cudaFree(d_orbit_indexes);
    cudaFree(d_orbit_arounds);
    cudaFree(d_info_seeds);

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
