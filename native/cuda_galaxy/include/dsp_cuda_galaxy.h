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
    double rand1;
    double rand2;
    double rand3;
    double rand4;
    double num13;
    double num14;
    int theme_seed;
    int type_case;
    int singularity_flags;
    int habitable_count_delta;
    int precision;
    int segment;
} dsp_planet_core_f32_out_t;

typedef struct dsp_cuda_pose_batch_head_timing_t {
    double h2d_seeds_ms;
    double gen_kernel_ms;
    double d2h_counts_ms;
    double gather_kernel_ms;
    double d2h_head_ms;
    double gen_phase1_ms;
    double gen_phase2_ms;
    double gen_seed_p50_ms;
    double gen_seed_p95_ms;
    double gen_seed_max_ms;
    double gen_attempts_total;
    double gen_collision_rejects_total;
    double gen_sphere_rejects_total;
    double gen_gate_skips_total;
    double d2h_head_submit_ms;
    double d2h_head_sync_wait_ms;
    double d2h_head_bytes_mb;
    double d2h_head_bw_gbps;
    double total_ms;
    double api_pre_ms;
    double api_post_ms;
    double api_total_host_ms;
    double set_device_ms;
    double ensure_buffers_ms;
    double event_setup_ms;
} dsp_cuda_pose_batch_head_timing_t;

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

int dsp_cuda_generate_temp_poses_params_fp64_batch_head(
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
    int* out_counts);

int dsp_cuda_generate_temp_poses_params_fp64_batch_head_device(
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
    dsp_vec3d_t* out_poses_device,
    int out_stride,
    int* out_counts);

int dsp_cuda_get_last_pose_batch_head_timing(
    dsp_cuda_pose_batch_head_timing_t* out_timing);

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

int dsp_cuda_planet_eval_core_f32_batch(
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
    dsp_planet_core_f32_out_t* out_results);

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

// Chunk-level wrappers for pipeline orchestration.
// Current implementation reuses existing batch kernels; ABI is stabilized for future fused kernels.
int dsp_cuda_mix_chunk_eval_planets_f32(
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
    dsp_planet_core_f32_out_t* out_results);

int dsp_cuda_mix_chunk_eval_veins_f32(
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

// Variant: theme arrays are passed once and each planet only references theme index.
// This avoids host-side per-planet expansion of vein/rare settings.
int dsp_cuda_mix_chunk_eval_veins_by_theme_f32(
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
    int* out_counts);

// Reduce per-seed signatures from flattened chunk data.
// This entry is host-side deterministic reduction (no RNG) and is used by compare pipeline fast-path.
int dsp_cuda_mix_chunk_reduce_signatures(
    int seed_count,
    const int* seed_star_offsets,   // len = seed_count + 1
    const int* seed_planet_offsets, // len = seed_count + 1
    const int* galaxy_star_counts,  // len = seed_count
    const int* birth_star_ids,      // len = seed_count
    const int* birth_planet_ids,    // len = seed_count
    const int* star_ids,            // len = total_stars
    const int* star_types,          // len = total_stars
    const int* star_spectrs,        // len = total_stars
    const int* star_planet_counts,  // len = total_stars (signature field: star.planetCount)
    const int* star_planet_loop_counts, // len = total_stars (actual traversed count: star.planets.Length)
    const int* star_pos_x,          // len = total_stars (already quantized to round(uPos.x*0.001))
    const int* star_pos_y,          // len = total_stars
    const int* star_pos_z,          // len = total_stars
    const int* planet_ids,          // len = total_planets
    const int* planet_types,        // len = total_planets
    const int* planet_themes,       // len = total_planets
    const int* planet_water_item_ids, // len = total_planets
    const int* planet_orbit_indexes,  // len = total_planets
    const int* planet_orbit_arounds,  // len = total_planets
    const int* planet_is_null,      // len = total_planets (1=true)
    const int* planet_is_gas,       // len = total_planets (1=true, ignored when is_null=1)
    const int* planet_vein_offsets, // len = total_planets, -1 for gas/null
    const int* vein_counts_flat,    // len = total_solid_planets * vein_stride
    int vein_stride,
    unsigned long long* out_galaxy_sigs,   // len = seed_count
    unsigned long long* out_planet_sigs,   // len = seed_count
    unsigned long long* out_vein_sigs,     // len = seed_count
    unsigned long long* out_pipeline_sigs  // len = seed_count
);

// Direct seed->signature pipeline entry.
// This path evaluates stars/planets/themes/veins from seeds and directly returns 4 signatures,
// so the host can bypass building GalaxyData/StarData/PlanetData object graphs.
int dsp_cuda_mix_signatures_from_seeds_f32(
    const int* galaxy_seeds,
    int seed_count,
    int star_count,
    int collision_fp64,
    int use_fp32_prob_compare,
    int vein_len,
    int device_id,
    int star_type_main_seq,
    int star_type_giant,
    int star_type_white_dwarf,
    int star_type_neutron_star,
    int star_type_black_hole,
    int spectr_m,
    int spectr_k,
    int spectr_g,
    int spectr_f,
    int spectr_a,
    int spectr_b,
    int spectr_o,
    int spectr_x,
    int planet_type_gas,
    int planet_type_ocean,
    int planet_type_vocano,
    int planet_type_desert,
    int planet_type_ice,
    int theme_distribute_default,
    int theme_distribute_birth,
    int theme_distribute_interstellar,
    const int* theme_ids,
    const int* theme_planet_types,
    const float* theme_temperatures,
    const int* theme_distributes,
    const int* theme_water_item_ids,
    const int* theme_vein_spot_offsets,   // len = theme_count + 1
    const int* theme_vein_spot_values,
    const int* theme_rare_vein_offsets,   // len = theme_count + 1
    const int* theme_rare_vein_values,
    const int* theme_rare_settings_offsets, // len = theme_count + 1
    const float* theme_rare_settings_values,
    int theme_count,
    unsigned long long* out_galaxy_sigs,
    unsigned long long* out_planet_sigs,
    unsigned long long* out_vein_sigs,
    unsigned long long* out_pipeline_sigs);

#ifdef __cplusplus
}
#endif
