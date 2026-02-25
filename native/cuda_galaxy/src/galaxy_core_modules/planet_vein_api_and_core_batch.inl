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

