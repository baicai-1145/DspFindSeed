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
