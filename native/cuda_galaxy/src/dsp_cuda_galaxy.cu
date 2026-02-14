#include "dsp_cuda_galaxy.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace
{
constexpr int kMBig = 2147483647;
constexpr int kMSeed = 161803398;

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
    float min2 = static_cast<float>(min_dist) * static_cast<float>(min_dist);
    for (int i = 0; i < count; ++i)
    {
        const dsp_vec3d_t& p = pts[i];
        float dx = static_cast<float>(pt.x - p.x);
        float dy = static_cast<float>(pt.y - p.y);
        float dz = static_cast<float>(pt.z - p.z);
        if (dx * dx + dy * dy + dz * dz < min2)
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

    cudaError_t rc = cudaMalloc(reinterpret_cast<void**>(&d_seeds), seeds_bytes);
    if (rc != cudaSuccess)
        return DSP_CUDA_ERR_CUDA;
    rc = cudaMalloc(reinterpret_cast<void**>(&d_counts), counts_bytes);
    if (rc != cudaSuccess)
    {
        cudaFree(d_seeds);
        return DSP_CUDA_ERR_CUDA;
    }
    rc = cudaMalloc(reinterpret_cast<void**>(&d_poses), poses_bytes);
    if (rc != cudaSuccess)
    {
        cudaFree(d_counts);
        cudaFree(d_seeds);
        return DSP_CUDA_ERR_CUDA;
    }
    rc = cudaMalloc(reinterpret_cast<void**>(&d_drunk), poses_bytes);
    if (rc != cudaSuccess)
    {
        cudaFree(d_poses);
        cudaFree(d_counts);
        cudaFree(d_seeds);
        return DSP_CUDA_ERR_CUDA;
    }

    rc = cudaMemcpy(d_seeds, seeds, seeds_bytes, cudaMemcpyHostToDevice);
    if (rc != cudaSuccess)
    {
        cudaFree(d_drunk);
        cudaFree(d_poses);
        cudaFree(d_counts);
        cudaFree(d_seeds);
        return DSP_CUDA_ERR_CUDA;
    }

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
        rc = cudaDeviceSynchronize();
    if (rc == cudaSuccess)
        rc = cudaMemcpy(out_counts, d_counts, counts_bytes, cudaMemcpyDeviceToHost);
    if (rc == cudaSuccess)
        rc = cudaMemcpy(out_poses, d_poses, poses_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_drunk);
    cudaFree(d_poses);
    cudaFree(d_counts);
    cudaFree(d_seeds);

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
