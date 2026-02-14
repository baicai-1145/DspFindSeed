#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dsp_vec3d_t {
    double x;
    double y;
    double z;
} dsp_vec3d_t;

enum {
    DSP_CUDA_OK = 0,
    DSP_CUDA_ERR_INVALID_ARGUMENT = -1,
    DSP_CUDA_ERR_CUDA = -2
};

int dsp_cuda_generate_temp_poses_params_fp64(
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
    int* out_count);

int dsp_cuda_generate_temp_poses_params_fp64_batch(
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
    int* out_counts);

int dsp_cuda_debug_rng_nextdouble(
    int seed,
    int count,
    int device_id,
    double* out_values);

int dsp_cuda_debug_rng_state_after_ctor(
    int seed,
    int device_id,
    int* out_seed_array_56,
    int* out_inext,
    int* out_inextp);

#ifdef __cplusplus
}
#endif
