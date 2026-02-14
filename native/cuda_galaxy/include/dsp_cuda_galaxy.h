#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dsp_vec3d_t {
    double x;
    double y;
    double z;
} dsp_vec3d_t;

typedef struct dsp_planet_core_f32_out_t {
    float orbit_radius;
    float orbit_inclination;
    float orbit_longitude;
    double orbital_period;
    float orbit_phase;
    float obliquity;
    double rotation_period;
    float rotation_phase;
    float sun_distance;
    float scale;
    float habitable_bias;
    float temperature_bias;
    float radius;
    float luminosity;
    float rand1;
    float rand2;
    float rand3;
    float rand4;
    int theme_seed;
    int type_case;
    int singularity_flags;
    int habitable_count_delta;
    int precision;
    int segment;
} dsp_planet_core_f32_out_t;

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

int dsp_cuda_refresh_planet_vein_spots_batch(
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
    int* out_counts);

int dsp_cuda_planet_eval_core_f32(
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
    dsp_planet_core_f32_out_t* out_result);

int dsp_cuda_planet_eval_gas_details_f32(
    int theme_seed,
    float gas_coef,
    float resource_coef,
    const float* in_gas_speeds,
    const float* in_gas_heat_values,
    int gas_len,
    int device_id,
    float* out_gas_speeds,
    double* out_total_heat);

#ifdef __cplusplus
}
#endif
