inline bool TryEvalThemeVeinHashGpu(
    int device_id,
    int seed_count,
    int vein_len,
    int use_fp32_prob_compare,
    const EnumMap& em,
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
    const SigGpuFlatSoA& flat,
    bool speed_only,
    unsigned long long* out_galaxy_sigs,
    unsigned long long* out_planet_sigs,
    unsigned long long* out_vein_sigs,
    unsigned long long* out_pipeline_sigs,
    int* out_debug_birth_star_ids,
    int* out_debug_birth_planet_ids,
    int* out_debug_theme_indexes)
{
    if (seed_count <= 0)
        return false;
    if (device_id >= 0)
    {
        cudaError_t rc_set = cudaSetDevice(device_id);
        if (rc_set != cudaSuccess)
            return false;
    }

    const int total_stars = flat.seed_star_offsets.empty() ? 0 : flat.seed_star_offsets.back();
    const int total_planets = flat.star_planet_offsets.empty() ? 0 : flat.star_planet_offsets.back();

    int* d_seed_star_offsets = nullptr;
    int* d_star_planet_offsets = nullptr;
    int* d_star_ids = nullptr;
    int* d_star_types = nullptr;
    int* d_star_spectrs = nullptr;
    int* d_star_planet_counts = nullptr;
    int* d_star_pos_qx = nullptr;
    int* d_star_pos_qy = nullptr;
    int* d_star_pos_qz = nullptr;
    int* d_star_indexes = nullptr;
    float* d_star_habitable_radiuses = nullptr;
    int* d_planet_ids = nullptr;
    int* d_planet_indexes = nullptr;
    int* d_planet_orbit_indexes = nullptr;
    int* d_planet_orbit_arounds = nullptr;
    int* d_planet_gen_seeds = nullptr;
    int* d_planet_gas_giants = nullptr;
    float* d_planet_core_habitable_bias = nullptr;
    float* d_planet_core_sun_distance = nullptr;
    float* d_planet_core_temperature_bias = nullptr;
    double* d_planet_core_num13 = nullptr;
    double* d_planet_core_num14 = nullptr;
    double* d_planet_core_rand1 = nullptr;
    const ThemeDeviceCacheEntry* theme_cache = nullptr;
    unsigned long long* d_out_galaxy_sigs = nullptr;
    unsigned long long* d_out_planet_sigs = nullptr;
    unsigned long long* d_out_vein_sigs = nullptr;
    unsigned long long* d_out_pipeline_sigs = nullptr;
    int* d_out_birth_star_ids = nullptr;
    int* d_out_birth_planet_ids = nullptr;
    int* d_out_theme_indexes = nullptr;
    cudaStream_t stream = nullptr;

    auto cleanup = [&]() {
        cudaFree(d_out_theme_indexes);
        cudaFree(d_out_birth_planet_ids);
        cudaFree(d_out_birth_star_ids);
        cudaFree(d_out_pipeline_sigs);
        cudaFree(d_out_vein_sigs);
        cudaFree(d_out_planet_sigs);
        cudaFree(d_out_galaxy_sigs);
        cudaFree(d_planet_core_rand1);
        cudaFree(d_planet_core_num14);
        cudaFree(d_planet_core_num13);
        cudaFree(d_planet_core_temperature_bias);
        cudaFree(d_planet_core_sun_distance);
        cudaFree(d_planet_core_habitable_bias);
        cudaFree(d_planet_gas_giants);
        cudaFree(d_planet_gen_seeds);
        cudaFree(d_planet_orbit_arounds);
        cudaFree(d_planet_orbit_indexes);
        cudaFree(d_planet_indexes);
        cudaFree(d_planet_ids);
        cudaFree(d_star_habitable_radiuses);
        cudaFree(d_star_indexes);
        cudaFree(d_star_pos_qz);
        cudaFree(d_star_pos_qy);
        cudaFree(d_star_pos_qx);
        cudaFree(d_star_planet_counts);
        cudaFree(d_star_spectrs);
        cudaFree(d_star_types);
        cudaFree(d_star_ids);
        cudaFree(d_star_planet_offsets);
        cudaFree(d_seed_star_offsets);
    };

    auto alloc_int = [&](int** p, size_t count) -> bool {
        size_t bytes = count > 0 ? count * sizeof(int) : sizeof(int);
        return cudaMalloc(reinterpret_cast<void**>(p), bytes) == cudaSuccess;
    };
    auto alloc_float = [&](float** p, size_t count) -> bool {
        size_t bytes = count > 0 ? count * sizeof(float) : sizeof(float);
        return cudaMalloc(reinterpret_cast<void**>(p), bytes) == cudaSuccess;
    };
    auto alloc_double = [&](double** p, size_t count) -> bool {
        size_t bytes = count > 0 ? count * sizeof(double) : sizeof(double);
        return cudaMalloc(reinterpret_cast<void**>(p), bytes) == cudaSuccess;
    };
    auto alloc_u64 = [&](unsigned long long** p, size_t count) -> bool {
        size_t bytes = count > 0 ? count * sizeof(unsigned long long) : sizeof(unsigned long long);
        return cudaMalloc(reinterpret_cast<void**>(p), bytes) == cudaSuccess;
    };
    auto h2d_int = [&](int* dst, const std::vector<int>& src) -> bool {
        if (src.empty())
            return true;
        return cudaMemcpyAsync(dst, src.data(), src.size() * sizeof(int), cudaMemcpyHostToDevice, stream) == cudaSuccess;
    };
    auto h2d_float = [&](float* dst, const std::vector<float>& src) -> bool {
        if (src.empty())
            return true;
        return cudaMemcpyAsync(dst, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess;
    };
    auto h2d_double = [&](double* dst, const std::vector<double>& src) -> bool {
        if (src.empty())
            return true;
        return cudaMemcpyAsync(dst, src.data(), src.size() * sizeof(double), cudaMemcpyHostToDevice, stream) == cudaSuccess;
    };
    if (!AcquireSigThreadStream(device_id, &stream))
    {
        cleanup();
        return false;
    }
    if (!EnsureThemeDeviceCache(
            device_id,
            theme_count,
            max_planet_type,
            theme_ids,
            theme_planet_types,
            theme_temperatures,
            theme_distributes,
            theme_water_item_ids,
            theme_vein_spot_offsets,
            theme_vein_spot_values,
            theme_rare_vein_offsets,
            theme_rare_vein_values,
            theme_rare_settings_offsets,
            theme_rare_settings_values,
            type_theme_offsets,
            type_theme_values,
            &theme_cache))
    {
        cleanup();
        return false;
    }

    if (!alloc_int(&d_seed_star_offsets, flat.seed_star_offsets.size()) ||
        !alloc_int(&d_star_planet_offsets, flat.star_planet_offsets.size()) ||
        !alloc_int(&d_star_ids, flat.star_ids.size()) ||
        !alloc_int(&d_star_types, flat.star_types.size()) ||
        !alloc_int(&d_star_spectrs, flat.star_spectrs.size()) ||
        !alloc_int(&d_star_planet_counts, flat.star_planet_counts.size()) ||
        !alloc_int(&d_star_pos_qx, flat.star_pos_qx.size()) ||
        !alloc_int(&d_star_pos_qy, flat.star_pos_qy.size()) ||
        !alloc_int(&d_star_pos_qz, flat.star_pos_qz.size()) ||
        !alloc_int(&d_star_indexes, flat.star_indexes.size()) ||
        !alloc_float(&d_star_habitable_radiuses, flat.star_habitable_radiuses.size()) ||
        !alloc_int(&d_planet_ids, flat.planet_ids.size()) ||
        !alloc_int(&d_planet_indexes, flat.planet_indexes.size()) ||
        !alloc_int(&d_planet_orbit_indexes, flat.planet_orbit_indexes.size()) ||
        !alloc_int(&d_planet_orbit_arounds, flat.planet_orbit_arounds.size()) ||
        !alloc_int(&d_planet_gen_seeds, flat.planet_gen_seeds.size()) ||
        !alloc_int(&d_planet_gas_giants, flat.planet_gas_giants.size()) ||
        !alloc_float(&d_planet_core_habitable_bias, flat.planet_core_habitable_bias.size()) ||
        !alloc_float(&d_planet_core_sun_distance, flat.planet_core_sun_distance.size()) ||
        !alloc_float(&d_planet_core_temperature_bias, flat.planet_core_temperature_bias.size()) ||
        !alloc_double(&d_planet_core_num13, flat.planet_core_num13.size()) ||
        !alloc_double(&d_planet_core_num14, flat.planet_core_num14.size()) ||
        !alloc_double(&d_planet_core_rand1, flat.planet_core_rand1.size()) ||
        !alloc_u64(&d_out_galaxy_sigs, static_cast<size_t>(seed_count)) ||
        !alloc_u64(&d_out_planet_sigs, static_cast<size_t>(seed_count)) ||
        !alloc_u64(&d_out_vein_sigs, static_cast<size_t>(seed_count)) ||
        !alloc_u64(&d_out_pipeline_sigs, static_cast<size_t>(seed_count)) ||
        (out_debug_birth_star_ids != nullptr && !alloc_int(&d_out_birth_star_ids, static_cast<size_t>(seed_count))) ||
        (out_debug_birth_planet_ids != nullptr && !alloc_int(&d_out_birth_planet_ids, static_cast<size_t>(seed_count))) ||
        (out_debug_theme_indexes != nullptr && !alloc_int(&d_out_theme_indexes, static_cast<size_t>(total_planets))))
    {
        cleanup();
        return false;
    }

    if (!h2d_int(d_seed_star_offsets, flat.seed_star_offsets) ||
        !h2d_int(d_star_planet_offsets, flat.star_planet_offsets) ||
        !h2d_int(d_star_ids, flat.star_ids) ||
        !h2d_int(d_star_types, flat.star_types) ||
        !h2d_int(d_star_spectrs, flat.star_spectrs) ||
        !h2d_int(d_star_planet_counts, flat.star_planet_counts) ||
        !h2d_int(d_star_pos_qx, flat.star_pos_qx) ||
        !h2d_int(d_star_pos_qy, flat.star_pos_qy) ||
        !h2d_int(d_star_pos_qz, flat.star_pos_qz) ||
        !h2d_int(d_star_indexes, flat.star_indexes) ||
        !h2d_float(d_star_habitable_radiuses, flat.star_habitable_radiuses) ||
        !h2d_int(d_planet_ids, flat.planet_ids) ||
        !h2d_int(d_planet_indexes, flat.planet_indexes) ||
        !h2d_int(d_planet_orbit_indexes, flat.planet_orbit_indexes) ||
        !h2d_int(d_planet_orbit_arounds, flat.planet_orbit_arounds) ||
        !h2d_int(d_planet_gen_seeds, flat.planet_gen_seeds) ||
        !h2d_int(d_planet_gas_giants, flat.planet_gas_giants) ||
        !h2d_float(d_planet_core_habitable_bias, flat.planet_core_habitable_bias) ||
        !h2d_float(d_planet_core_sun_distance, flat.planet_core_sun_distance) ||
        !h2d_float(d_planet_core_temperature_bias, flat.planet_core_temperature_bias) ||
        !h2d_double(d_planet_core_num13, flat.planet_core_num13) ||
        !h2d_double(d_planet_core_num14, flat.planet_core_num14) ||
        !h2d_double(d_planet_core_rand1, flat.planet_core_rand1))
    {
        cleanup();
        return false;
    }

    const int block = ResolveSigBlockSize();
    const int grid = (seed_count + block - 1) / block;
    if (out_debug_theme_indexes != nullptr)
    {
        std::fprintf(stderr,
            "[native-sig-theme-debug] em planet(g/o/v/d/i)=%d/%d/%d/%d/%d distribute(def/birth/inter)=%d/%d/%d maxPlanetType=%d themeCount=%d\n",
            em.planet_type_gas, em.planet_type_ocean, em.planet_type_vocano, em.planet_type_desert, em.planet_type_ice,
            em.theme_distribute_default, em.theme_distribute_birth, em.theme_distribute_interstellar,
            max_planet_type, theme_count);
        int dbg_n = std::min(6, static_cast<int>(flat.star_planet_offsets.size()));
        std::fprintf(stderr, "[native-sig-theme-debug] starPlanetOffsetsHost:");
        for (int i = 0; i < dbg_n; ++i)
            std::fprintf(stderr, " %d", flat.star_planet_offsets[static_cast<size_t>(i)]);
        std::fprintf(stderr, "\n");
        std::vector<int> dbg_offsets(static_cast<size_t>(dbg_n), 0);
        if (dbg_n > 0 &&
            cudaMemcpy(dbg_offsets.data(), d_star_planet_offsets, static_cast<size_t>(dbg_n) * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess)
        {
            std::fprintf(stderr, "[native-sig-theme-debug] starPlanetOffsetsDev:");
            for (int i = 0; i < dbg_n; ++i)
                std::fprintf(stderr, " %d", dbg_offsets[static_cast<size_t>(i)]);
            std::fprintf(stderr, "\n");
        }
        std::fflush(stderr);
    }
    EvalThemeVeinHashKernel<<<grid, block, 0, stream>>>(
        seed_count,
        vein_len,
        use_fp32_prob_compare,
        em,
        d_seed_star_offsets,
        d_star_planet_offsets,
        d_star_ids,
        d_star_types,
        d_star_spectrs,
        d_star_planet_counts,
        d_star_pos_qx,
        d_star_pos_qy,
        d_star_pos_qz,
        d_star_indexes,
        d_star_habitable_radiuses,
        d_planet_ids,
        d_planet_indexes,
        d_planet_orbit_indexes,
        d_planet_orbit_arounds,
        d_planet_gen_seeds,
        d_planet_gas_giants,
        d_planet_core_habitable_bias,
        d_planet_core_sun_distance,
        d_planet_core_temperature_bias,
        d_planet_core_num13,
        d_planet_core_num14,
        d_planet_core_rand1,
        theme_count,
        theme_cache->d_theme_ids,
        theme_cache->d_theme_planet_types,
        theme_cache->d_theme_temperatures,
        theme_cache->d_theme_distributes,
        theme_cache->d_theme_water_item_ids,
        theme_cache->d_theme_vein_spot_offsets,
        theme_cache->d_theme_vein_spot_values,
        theme_cache->d_theme_rare_vein_offsets,
        theme_cache->d_theme_rare_vein_values,
        theme_cache->d_theme_rare_settings_offsets,
        theme_cache->d_theme_rare_settings_values,
        max_planet_type,
        theme_cache->d_type_theme_offsets,
        theme_cache->d_type_theme_values,
        d_out_galaxy_sigs,
        d_out_planet_sigs,
        d_out_vein_sigs,
        d_out_pipeline_sigs,
        d_out_birth_star_ids,
        d_out_birth_planet_ids,
        d_out_theme_indexes);

    cudaError_t rc = cudaGetLastError();
    if (rc == cudaSuccess && speed_only)
        rc = cudaStreamSynchronize(stream);
    if (rc == cudaSuccess && speed_only)
    {
        cleanup();
        return true;
    }
    if (rc == cudaSuccess)
        rc = cudaMemcpyAsync(out_galaxy_sigs, d_out_galaxy_sigs, static_cast<size_t>(seed_count) * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream);
    if (rc == cudaSuccess)
        rc = cudaMemcpyAsync(out_planet_sigs, d_out_planet_sigs, static_cast<size_t>(seed_count) * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream);
    if (rc == cudaSuccess)
        rc = cudaMemcpyAsync(out_vein_sigs, d_out_vein_sigs, static_cast<size_t>(seed_count) * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream);
    if (rc == cudaSuccess)
        rc = cudaMemcpyAsync(out_pipeline_sigs, d_out_pipeline_sigs, static_cast<size_t>(seed_count) * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream);
    if (rc == cudaSuccess && out_debug_birth_star_ids != nullptr && d_out_birth_star_ids != nullptr)
        rc = cudaMemcpyAsync(out_debug_birth_star_ids, d_out_birth_star_ids, static_cast<size_t>(seed_count) * sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (rc == cudaSuccess && out_debug_birth_planet_ids != nullptr && d_out_birth_planet_ids != nullptr)
        rc = cudaMemcpyAsync(out_debug_birth_planet_ids, d_out_birth_planet_ids, static_cast<size_t>(seed_count) * sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (rc == cudaSuccess && out_debug_theme_indexes != nullptr && d_out_theme_indexes != nullptr)
        rc = cudaMemcpyAsync(out_debug_theme_indexes, d_out_theme_indexes, static_cast<size_t>(total_planets) * sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (rc == cudaSuccess)
        rc = cudaStreamSynchronize(stream);

    cleanup();
    return rc == cudaSuccess;
}

inline unsigned long long MixHash(unsigned long long h, int v)
{
    h ^= static_cast<unsigned int>(v);
    h *= kFnvPrime;
    return h;
}

inline int ResolveGroupSize(int seed_count, int star_count, int iter_count)
{
    const char* env = std::getenv("DSP_NATIVE_SIG_GROUP_SEEDS");
    if (env != nullptr && env[0] != '\0')
    {
        int v = std::atoi(env);
        if (v < 1)
            v = 1;
        if (v > 65536)
            v = 65536;
        return v;
    }

    // Prefer one native group per submitted seed chunk while keeping host memory bounded.
    long long denom = static_cast<long long>(sizeof(dsp_vec3d_t)) *
                      static_cast<long long>(std::max(1, star_count)) *
                      static_cast<long long>(std::max(1, iter_count));
    long long max_by_pose = denom > 0 ? (256LL * 1024LL * 1024LL) / denom : 1LL; // ~256 MiB pose buffer budget
    if (max_by_pose < 1LL)
        max_by_pose = 1LL;

    int v = seed_count;
    if (v > 20000)
        v = 20000;
    if (v > static_cast<int>(max_by_pose))
        v = static_cast<int>(max_by_pose);
    if (v < 1)
        v = 1;
    return v;
}

inline int ResolveSigBlockSize()
{
    const char* env = std::getenv("DSP_NATIVE_SIG_BLOCK_SIZE");
    if (env != nullptr && env[0] != '\0')
    {
        int v = std::atoi(env);
        if (v < 64)
            v = 64;
        if (v > 512)
            v = 512;
        v = (v / 32) * 32;
        if (v < 32)
            v = 32;
        return v;
    }
    return 256;
}
} // namespace
