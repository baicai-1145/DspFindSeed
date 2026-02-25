inline bool TryEvalThemeVeinHashGpuDirectFromPose(
    int device_id,
    const EnumMap& em,
    const int* galaxy_seeds,
    int seed_count,
    int star_count,
    int iter_count,
    const std::vector<int>& pose_raw_counts,
    const std::vector<dsp_vec3d_t>& pose_raw,
    const dsp_vec3d_t* pose_raw_device,
    int vein_len,
    int use_fp32_prob_compare,
    int star_type_white_dwarf,
    int star_type_neutron_star,
    int star_type_black_hole,
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
    bool speed_only,
    unsigned long long* out_galaxy_sigs,
    unsigned long long* out_planet_sigs,
    unsigned long long* out_vein_sigs,
    unsigned long long* out_pipeline_sigs)
{
    if (galaxy_seeds == nullptr || seed_count <= 0 || star_count <= 0 || iter_count <= 0)
        return false;
    if (static_cast<int>(pose_raw_counts.size()) < seed_count)
        return false;
    if (pose_raw_device == nullptr &&
        pose_raw.size() < static_cast<size_t>(seed_count) * static_cast<size_t>(star_count))
        return false;
    if (device_id >= 0)
    {
        cudaError_t rc_set = cudaSetDevice(device_id);
        if (rc_set != cudaSuccess)
            return false;
    }

    std::vector<int> seed_star_counts(seed_count, 0);
    std::vector<int> seed_star_offsets(seed_count + 1, 0);
    int total_stars = 0;
    for (int i = 0; i < seed_count; ++i)
    {
        int pc = pose_raw_counts[i] / iter_count;
        if (pc < 0)
            pc = 0;
        if (pc > star_count)
            pc = star_count;
        seed_star_counts[i] = pc;
        seed_star_offsets[i] = total_stars;
        total_stars += pc;
    }
    seed_star_offsets[seed_count] = total_stars;
    if (total_stars <= 0)
        return false;

    const size_t plans_n = static_cast<size_t>(total_stars) * static_cast<size_t>(kStarPlanMaxPlanets);

    int* d_galaxy_seeds = nullptr;
    int* d_seed_star_counts = nullptr;
    int* d_seed_star_offsets = nullptr;
    bool owns_d_pose_raw = pose_raw_device == nullptr;
    dsp_vec3d_t* d_pose_raw = owns_d_pose_raw ? nullptr : const_cast<dsp_vec3d_t*>(pose_raw_device);

    int* d_star_ids = nullptr;
    int* d_star_seeds = nullptr;
    int* d_star_types = nullptr;
    int* d_star_spectrs = nullptr;
    int* d_star_indexes = nullptr;
    int* d_star_counts_in_galaxy = nullptr;
    int* d_star_pos_qx = nullptr;
    int* d_star_pos_qy = nullptr;
    int* d_star_pos_qz = nullptr;
    float* d_star_orbit_scalers = nullptr;
    double* d_star_masses = nullptr;
    float* d_star_habitable_radiuses = nullptr;
    float* d_star_light_balance_radiuses = nullptr;

    int* d_out_planet_counts = nullptr;
    int* d_out_orbit_arounds = nullptr;
    int* d_out_orbit_indexes = nullptr;
    int* d_out_numbers = nullptr;
    int* d_out_gas_giants = nullptr;
    int* d_out_info_seeds = nullptr;
    int* d_out_gen_seeds = nullptr;
    CoreLite* d_out_core = nullptr;
    int* d_star_planet_offsets = nullptr;

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
    int* d_total_planets_dev = nullptr;

    const ThemeDeviceCacheEntry* theme_cache = nullptr;

    unsigned long long* d_out_galaxy_sigs = nullptr;
    unsigned long long* d_out_planet_sigs = nullptr;
    unsigned long long* d_out_vein_sigs = nullptr;
    unsigned long long* d_out_pipeline_sigs = nullptr;

    cudaStream_t stream = nullptr;
    auto cleanup = [&]() {};

    if (!AcquireSigThreadStream(device_id, &stream))
        return false;
    DirectSigWorkspace& workspace = AcquireDirectSigWorkspace(device_id, stream);

    bool debug_direct_star = false;
    if (const char* ds = std::getenv("DSP_NATIVE_SIG_DEBUG_DIRECT_STAR"))
        debug_direct_star = std::atoi(ds) != 0;
    bool debug_direct_plan = false;
    if (const char* dp = std::getenv("DSP_NATIVE_SIG_DEBUG_DIRECT_PLAN"))
        debug_direct_plan = std::atoi(dp) != 0;
    bool debug_direct_core = false;
    if (const char* dc = std::getenv("DSP_NATIVE_SIG_DEBUG_DIRECT_CORE"))
        debug_direct_core = std::atoi(dc) != 0;
    bool strict_direct_star_override = !speed_only;
    if (const char* ss = std::getenv("DSP_NATIVE_SIG_DIRECT_STRICT_STARS"))
        strict_direct_star_override = std::atoi(ss) != 0;
    const bool need_host_star_reference = strict_direct_star_override || debug_direct_star || debug_direct_plan || debug_direct_core;
    const bool use_fused_plan_core_pack = !(debug_direct_plan || debug_direct_core);

    auto alloc_int = [&](DirectSigBufferId id, int** p, size_t count) -> bool {
        size_t bytes = count > 0 ? count * sizeof(int) : sizeof(int);
        return workspace.Ensure(id, bytes, reinterpret_cast<void**>(p));
    };
    auto alloc_float = [&](DirectSigBufferId id, float** p, size_t count) -> bool {
        size_t bytes = count > 0 ? count * sizeof(float) : sizeof(float);
        return workspace.Ensure(id, bytes, reinterpret_cast<void**>(p));
    };
    auto alloc_double = [&](DirectSigBufferId id, double** p, size_t count) -> bool {
        size_t bytes = count > 0 ? count * sizeof(double) : sizeof(double);
        return workspace.Ensure(id, bytes, reinterpret_cast<void**>(p));
    };
    auto alloc_u64 = [&](DirectSigBufferId id, unsigned long long** p, size_t count) -> bool {
        size_t bytes = count > 0 ? count * sizeof(unsigned long long) : sizeof(unsigned long long);
        return workspace.Ensure(id, bytes, reinterpret_cast<void**>(p));
    };
    auto alloc_core = [&](DirectSigBufferId id, CoreLite** p, size_t count) -> bool {
        size_t bytes = count > 0 ? count * sizeof(CoreLite) : sizeof(CoreLite);
        return workspace.Ensure(id, bytes, reinterpret_cast<void**>(p));
    };
    auto alloc_pose = [&](DirectSigBufferId id, dsp_vec3d_t** p, size_t count) -> bool {
        size_t bytes = count > 0 ? count * sizeof(dsp_vec3d_t) : sizeof(dsp_vec3d_t);
        return workspace.Ensure(id, bytes, reinterpret_cast<void**>(p));
    };
    auto h2d_raw = [&](void* dst, const void* src, size_t bytes) -> bool {
        if (bytes == 0)
            return true;
        return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream) == cudaSuccess;
    };
    auto d2h_raw = [&](void* dst, const void* src, size_t bytes) -> bool {
        if (bytes == 0)
            return true;
        return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream) == cudaSuccess;
    };

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

    bool alloc_ok =
        alloc_int(DirectSigBufferId::GalaxySeeds, &d_galaxy_seeds, static_cast<size_t>(seed_count)) &&
        alloc_int(DirectSigBufferId::SeedStarCounts, &d_seed_star_counts, static_cast<size_t>(seed_count)) &&
        alloc_int(DirectSigBufferId::SeedStarOffsets, &d_seed_star_offsets, static_cast<size_t>(seed_count + 1)) &&
        (!owns_d_pose_raw || alloc_pose(DirectSigBufferId::PoseRaw, &d_pose_raw, static_cast<size_t>(seed_count) * static_cast<size_t>(star_count))) &&
        alloc_int(DirectSigBufferId::StarIds, &d_star_ids, static_cast<size_t>(total_stars)) &&
        alloc_int(DirectSigBufferId::StarSeeds, &d_star_seeds, static_cast<size_t>(total_stars)) &&
        alloc_int(DirectSigBufferId::StarTypes, &d_star_types, static_cast<size_t>(total_stars)) &&
        alloc_int(DirectSigBufferId::StarSpectrs, &d_star_spectrs, static_cast<size_t>(total_stars)) &&
        alloc_int(DirectSigBufferId::StarIndexes, &d_star_indexes, static_cast<size_t>(total_stars)) &&
        alloc_int(DirectSigBufferId::StarCountsInGalaxy, &d_star_counts_in_galaxy, static_cast<size_t>(total_stars)) &&
        alloc_int(DirectSigBufferId::StarPosQx, &d_star_pos_qx, static_cast<size_t>(total_stars)) &&
        alloc_int(DirectSigBufferId::StarPosQy, &d_star_pos_qy, static_cast<size_t>(total_stars)) &&
        alloc_int(DirectSigBufferId::StarPosQz, &d_star_pos_qz, static_cast<size_t>(total_stars)) &&
        alloc_float(DirectSigBufferId::StarOrbitScalers, &d_star_orbit_scalers, static_cast<size_t>(total_stars)) &&
        alloc_double(DirectSigBufferId::StarMasses, &d_star_masses, static_cast<size_t>(total_stars)) &&
        alloc_float(DirectSigBufferId::StarHabitableRadiuses, &d_star_habitable_radiuses, static_cast<size_t>(total_stars)) &&
        alloc_float(DirectSigBufferId::StarLightBalanceRadiuses, &d_star_light_balance_radiuses, static_cast<size_t>(total_stars)) &&
        alloc_int(DirectSigBufferId::OutPlanetCounts, &d_out_planet_counts, static_cast<size_t>(total_stars)) &&
        alloc_int(DirectSigBufferId::StarPlanetOffsets, &d_star_planet_offsets, static_cast<size_t>(total_stars + 1)) &&
        alloc_int(DirectSigBufferId::PlanetIds, &d_planet_ids, plans_n) &&
        alloc_int(DirectSigBufferId::PlanetIndexes, &d_planet_indexes, plans_n) &&
        alloc_int(DirectSigBufferId::PlanetOrbitIndexes, &d_planet_orbit_indexes, plans_n) &&
        alloc_int(DirectSigBufferId::PlanetOrbitArounds, &d_planet_orbit_arounds, plans_n) &&
        alloc_int(DirectSigBufferId::PlanetGenSeeds, &d_planet_gen_seeds, plans_n) &&
        alloc_int(DirectSigBufferId::PlanetGasGiants, &d_planet_gas_giants, plans_n) &&
        alloc_float(DirectSigBufferId::PlanetCoreHabitableBias, &d_planet_core_habitable_bias, plans_n) &&
        alloc_float(DirectSigBufferId::PlanetCoreSunDistance, &d_planet_core_sun_distance, plans_n) &&
        alloc_float(DirectSigBufferId::PlanetCoreTemperatureBias, &d_planet_core_temperature_bias, plans_n) &&
        alloc_double(DirectSigBufferId::PlanetCoreNum13, &d_planet_core_num13, plans_n) &&
        alloc_double(DirectSigBufferId::PlanetCoreNum14, &d_planet_core_num14, plans_n) &&
        alloc_double(DirectSigBufferId::PlanetCoreRand1, &d_planet_core_rand1, plans_n) &&
        alloc_int(DirectSigBufferId::TotalPlanetsDev, &d_total_planets_dev, 1) &&
        alloc_u64(DirectSigBufferId::OutGalaxySigs, &d_out_galaxy_sigs, static_cast<size_t>(seed_count)) &&
        alloc_u64(DirectSigBufferId::OutPlanetSigs, &d_out_planet_sigs, static_cast<size_t>(seed_count)) &&
        alloc_u64(DirectSigBufferId::OutVeinSigs, &d_out_vein_sigs, static_cast<size_t>(seed_count)) &&
        alloc_u64(DirectSigBufferId::OutPipelineSigs, &d_out_pipeline_sigs, static_cast<size_t>(seed_count));
    if (alloc_ok && !use_fused_plan_core_pack)
    {
        alloc_ok =
            alloc_int(DirectSigBufferId::DebugPlanOrbitArounds, &d_out_orbit_arounds, plans_n) &&
            alloc_int(DirectSigBufferId::DebugPlanOrbitIndexes, &d_out_orbit_indexes, plans_n) &&
            alloc_int(DirectSigBufferId::DebugPlanNumbers, &d_out_numbers, plans_n) &&
            alloc_int(DirectSigBufferId::DebugPlanGasGiants, &d_out_gas_giants, plans_n) &&
            alloc_int(DirectSigBufferId::DebugPlanInfoSeeds, &d_out_info_seeds, plans_n) &&
            alloc_int(DirectSigBufferId::DebugPlanGenSeeds, &d_out_gen_seeds, plans_n) &&
            alloc_core(DirectSigBufferId::DebugCoreFlat, &d_out_core, plans_n);
    }
    if (!alloc_ok)
    {
        cleanup();
        return false;
    }

    if (!h2d_raw(d_galaxy_seeds, galaxy_seeds, static_cast<size_t>(seed_count) * sizeof(int)) ||
        !h2d_raw(d_seed_star_counts, seed_star_counts.data(), static_cast<size_t>(seed_count) * sizeof(int)) ||
        !h2d_raw(d_seed_star_offsets, seed_star_offsets.data(), static_cast<size_t>(seed_count + 1) * sizeof(int)) ||
        (owns_d_pose_raw &&
         !h2d_raw(d_pose_raw, pose_raw.data(), static_cast<size_t>(seed_count) * static_cast<size_t>(star_count) * sizeof(dsp_vec3d_t))))
    {
        cleanup();
        return false;
    }

    const int block_seed = ResolveSigBlockSize();
    const int grid_seed = (seed_count + block_seed - 1) / block_seed;
    BuildStarsFromPoseKernel<<<grid_seed, block_seed, 0, stream>>>(
        d_galaxy_seeds,
        d_seed_star_counts,
        d_seed_star_offsets,
        d_pose_raw,
        star_count,
        seed_count,
        em,
        d_star_ids,
        d_star_seeds,
        d_star_types,
        d_star_spectrs,
        d_star_indexes,
        d_star_counts_in_galaxy,
        d_star_pos_qx,
        d_star_pos_qy,
        d_star_pos_qz,
        d_star_orbit_scalers,
        d_star_masses,
        d_star_habitable_radiuses,
        d_star_light_balance_radiuses);
    if (cudaGetLastError() != cudaSuccess)
    {
        cleanup();
        return false;
    }
    if (need_host_star_reference &&
        pose_raw.size() < static_cast<size_t>(seed_count) * static_cast<size_t>(star_count))
    {
        cleanup();
        return false;
    }

    // NOTE: GPU-star build introduces tiny numeric deltas vs host std:: math that can amplify in theme branches.
    // Keep host-precise override for correctness path; speed-only can skip it to reduce host bottleneck.
    std::vector<int> h_star_ids;
    std::vector<int> h_star_seeds;
    std::vector<int> h_star_types_host;
    std::vector<int> h_star_spectrs_host;
    std::vector<int> h_star_indexes_host;
    std::vector<int> h_star_counts_host;
    std::vector<int> h_star_pos_qx_host;
    std::vector<int> h_star_pos_qy_host;
    std::vector<int> h_star_pos_qz_host;
    std::vector<float> h_star_orbit_host;
    std::vector<double> h_star_mass_host;
    std::vector<float> h_star_hab_host;
    std::vector<float> h_star_light_host;
    if (need_host_star_reference)
    {
        h_star_ids.assign(static_cast<size_t>(total_stars), 0);
        h_star_seeds.assign(static_cast<size_t>(total_stars), 0);
        h_star_types_host.assign(static_cast<size_t>(total_stars), 0);
        h_star_spectrs_host.assign(static_cast<size_t>(total_stars), 0);
        h_star_indexes_host.assign(static_cast<size_t>(total_stars), 0);
        h_star_counts_host.assign(static_cast<size_t>(total_stars), 0);
        h_star_pos_qx_host.assign(static_cast<size_t>(total_stars), 0);
        h_star_pos_qy_host.assign(static_cast<size_t>(total_stars), 0);
        h_star_pos_qz_host.assign(static_cast<size_t>(total_stars), 0);
        h_star_orbit_host.assign(static_cast<size_t>(total_stars), 0.0f);
        h_star_mass_host.assign(static_cast<size_t>(total_stars), 0.0);
        h_star_hab_host.assign(static_cast<size_t>(total_stars), 0.0f);
        h_star_light_host.assign(static_cast<size_t>(total_stars), 0.0f);
        for (int seed_idx = 0; seed_idx < seed_count; ++seed_idx)
        {
            const int cnt = seed_star_counts[seed_idx];
            const int base = seed_star_offsets[seed_idx];
            if (cnt <= 0)
                continue;
            DotNet35RandomHost galaxy_rng(galaxy_seeds[seed_idx]);
            galaxy_rng.Next(); // consumed by pose seed
            float num1 = static_cast<float>(galaxy_rng.NextDouble());
            float num2 = static_cast<float>(galaxy_rng.NextDouble());
            float num3 = static_cast<float>(galaxy_rng.NextDouble());
            float num4 = static_cast<float>(galaxy_rng.NextDouble());
            int num5 = static_cast<int>(std::ceil(0.01 * cnt + num1 * 0.300000011920929));
            int num6 = static_cast<int>(std::ceil(0.01 * cnt + num2 * 0.300000011920929));
            int num7 = static_cast<int>(std::ceil(0.0160000007599592 * cnt + num3 * 0.400000005960464));
            int num8 = static_cast<int>(std::ceil(0.0130000002682209 * cnt + num4 * 1.39999997615814));
            int num9 = cnt - num5;
            int num10 = num9 - num6;
            int num11 = num10 - num7;
            int num12 = (num11 - 1) / num8;
            int num13 = num12 / 2;

            for (int si = 0; si < cnt; ++si)
            {
                const int flat = base + si;
                const int star_seed = galaxy_rng.Next();
                StarLite st{};
                if (si == 0)
                {
                    st = CreateBirthStarLite(cnt, star_seed, em);
                }
                else
                {
                    int need_spectr = em.spectr_x;
                    if (si == 3) need_spectr = em.spectr_m;
                    else if (si == num11 - 1) need_spectr = em.spectr_o;

                    int need_type = em.star_type_main_seq;
                    if (num12 != 0 && si % num12 == num13)
                        need_type = em.star_type_giant;
                    if (si >= num9)
                        need_type = em.star_type_black_hole;
                    else if (si >= num10)
                        need_type = em.star_type_neutron_star;
                    else if (si >= num11)
                        need_type = em.star_type_white_dwarf;

                    const dsp_vec3d_t pos = pose_raw[static_cast<size_t>(seed_idx) * static_cast<size_t>(star_count) + static_cast<size_t>(si)];
                    st = CreateStarLite(cnt, pos, si + 1, star_seed, need_type, need_spectr, em);
                }
                h_star_ids[flat] = si + 1;
                h_star_seeds[flat] = star_seed;
                h_star_types_host[flat] = st.type;
                h_star_spectrs_host[flat] = st.spectr;
                h_star_indexes_host[flat] = si;
                h_star_counts_host[flat] = cnt;
                h_star_pos_qx_host[flat] = st.pos_qx;
                h_star_pos_qy_host[flat] = st.pos_qy;
                h_star_pos_qz_host[flat] = st.pos_qz;
                h_star_orbit_host[flat] = st.orbit_scaler;
                h_star_mass_host[flat] = static_cast<double>(st.mass);
                h_star_hab_host[flat] = st.habitable_radius;
                h_star_light_host[flat] = st.light_balance_radius;
            }
        }
    }
    if (strict_direct_star_override &&
        (!h2d_raw(d_star_ids, h_star_ids.data(), static_cast<size_t>(total_stars) * sizeof(int)) ||
         !h2d_raw(d_star_seeds, h_star_seeds.data(), static_cast<size_t>(total_stars) * sizeof(int)) ||
         !h2d_raw(d_star_types, h_star_types_host.data(), static_cast<size_t>(total_stars) * sizeof(int)) ||
         !h2d_raw(d_star_spectrs, h_star_spectrs_host.data(), static_cast<size_t>(total_stars) * sizeof(int)) ||
         !h2d_raw(d_star_indexes, h_star_indexes_host.data(), static_cast<size_t>(total_stars) * sizeof(int)) ||
         !h2d_raw(d_star_counts_in_galaxy, h_star_counts_host.data(), static_cast<size_t>(total_stars) * sizeof(int)) ||
         !h2d_raw(d_star_pos_qx, h_star_pos_qx_host.data(), static_cast<size_t>(total_stars) * sizeof(int)) ||
         !h2d_raw(d_star_pos_qy, h_star_pos_qy_host.data(), static_cast<size_t>(total_stars) * sizeof(int)) ||
         !h2d_raw(d_star_pos_qz, h_star_pos_qz_host.data(), static_cast<size_t>(total_stars) * sizeof(int)) ||
         !h2d_raw(d_star_orbit_scalers, h_star_orbit_host.data(), static_cast<size_t>(total_stars) * sizeof(float)) ||
         !h2d_raw(d_star_masses, h_star_mass_host.data(), static_cast<size_t>(total_stars) * sizeof(double)) ||
         !h2d_raw(d_star_habitable_radiuses, h_star_hab_host.data(), static_cast<size_t>(total_stars) * sizeof(float)) ||
         !h2d_raw(d_star_light_balance_radiuses, h_star_light_host.data(), static_cast<size_t>(total_stars) * sizeof(float))))
    {
        cleanup();
        return false;
    }

    if (debug_direct_star && seed_count > 0 && need_host_star_reference)
    {
        std::vector<int> h_star_types(static_cast<size_t>(total_stars));
        std::vector<int> h_star_spectrs(static_cast<size_t>(total_stars));
        std::vector<float> h_star_orbit(static_cast<size_t>(total_stars));
        std::vector<double> h_star_mass(static_cast<size_t>(total_stars));
        std::vector<float> h_star_hab(static_cast<size_t>(total_stars));
        std::vector<float> h_star_light(static_cast<size_t>(total_stars));
        if (!d2h_raw(h_star_types.data(), d_star_types, static_cast<size_t>(total_stars) * sizeof(int)) ||
            !d2h_raw(h_star_spectrs.data(), d_star_spectrs, static_cast<size_t>(total_stars) * sizeof(int)) ||
            !d2h_raw(h_star_orbit.data(), d_star_orbit_scalers, static_cast<size_t>(total_stars) * sizeof(float)) ||
            !d2h_raw(h_star_mass.data(), d_star_masses, static_cast<size_t>(total_stars) * sizeof(double)) ||
            !d2h_raw(h_star_hab.data(), d_star_habitable_radiuses, static_cast<size_t>(total_stars) * sizeof(float)) ||
            !d2h_raw(h_star_light.data(), d_star_light_balance_radiuses, static_cast<size_t>(total_stars) * sizeof(float)) ||
            cudaStreamSynchronize(stream) != cudaSuccess)
        {
            cleanup();
            return false;
        }

        const int seed0_count = seed_star_counts[0];
        const int seed0_base = seed_star_offsets[0];
        DotNet35RandomHost galaxy_rng(galaxy_seeds[0]);
        galaxy_rng.Next();
        float num1 = static_cast<float>(galaxy_rng.NextDouble());
        float num2 = static_cast<float>(galaxy_rng.NextDouble());
        float num3 = static_cast<float>(galaxy_rng.NextDouble());
        float num4 = static_cast<float>(galaxy_rng.NextDouble());
        int num5 = static_cast<int>(std::ceil(0.01 * seed0_count + num1 * 0.300000011920929));
        int num6 = static_cast<int>(std::ceil(0.01 * seed0_count + num2 * 0.300000011920929));
        int num7 = static_cast<int>(std::ceil(0.0160000007599592 * seed0_count + num3 * 0.400000005960464));
        int num8 = static_cast<int>(std::ceil(0.0130000002682209 * seed0_count + num4 * 1.39999997615814));
        int num9 = seed0_count - num5;
        int num10 = num9 - num6;
        int num11 = num10 - num7;
        int num12 = (num11 - 1) / num8;
        int num13 = num12 / 2;
        bool printed = false;
        int diff_count = 0;
        double max_mass_diff = 0.0;
        double max_orbit_diff = 0.0;
        double max_hab_diff = 0.0;
        double max_light_diff = 0.0;
        for (int si = 0; si < seed0_count; ++si)
        {
            const int star_seed = galaxy_rng.Next();
            StarLite st{};
            if (si == 0)
            {
                st = CreateBirthStarLite(seed0_count, star_seed, em);
            }
            else
            {
                int need_spectr = em.spectr_x;
                if (si == 3) need_spectr = em.spectr_m;
                else if (si == num11 - 1) need_spectr = em.spectr_o;
                int need_type = em.star_type_main_seq;
                if (num12 != 0 && si % num12 == num13)
                    need_type = em.star_type_giant;
                if (si >= num9)
                    need_type = em.star_type_black_hole;
                else if (si >= num10)
                    need_type = em.star_type_neutron_star;
                else if (si >= num11)
                    need_type = em.star_type_white_dwarf;
                const dsp_vec3d_t pos = pose_raw[static_cast<size_t>(si)];
                st = CreateStarLite(seed0_count, pos, si + 1, star_seed, need_type, need_spectr, em);
            }
            const int flat = seed0_base + si;
            const bool diff =
                h_star_types[flat] != st.type ||
                h_star_spectrs[flat] != st.spectr ||
                std::fabs(h_star_orbit[flat] - st.orbit_scaler) > 1e-6f ||
                std::fabs(h_star_mass[flat] - static_cast<double>(st.mass)) > 1e-9 ||
                std::fabs(h_star_hab[flat] - st.habitable_radius) > 1e-6f ||
                std::fabs(h_star_light[flat] - st.light_balance_radius) > 1e-6f;
            max_mass_diff = std::max(max_mass_diff, std::fabs(h_star_mass[flat] - static_cast<double>(st.mass)));
            max_orbit_diff = std::max(max_orbit_diff, static_cast<double>(std::fabs(h_star_orbit[flat] - st.orbit_scaler)));
            max_hab_diff = std::max(max_hab_diff, static_cast<double>(std::fabs(h_star_hab[flat] - st.habitable_radius)));
            max_light_diff = std::max(max_light_diff, static_cast<double>(std::fabs(h_star_light[flat] - st.light_balance_radius)));
            if (diff && !printed)
            {
                std::fprintf(stderr,
                    "[direct-star-diff] si=%d type gpu/host=%d/%d spectr=%d/%d orbit=%.9f/%.9f mass=%.12f/%.12f hab=%.9f/%.9f light=%.9f/%.9f\n",
                    si,
                    h_star_types[flat], st.type,
                    h_star_spectrs[flat], st.spectr,
                    static_cast<double>(h_star_orbit[flat]), static_cast<double>(st.orbit_scaler),
                    h_star_mass[flat], static_cast<double>(st.mass),
                    static_cast<double>(h_star_hab[flat]), static_cast<double>(st.habitable_radius),
                    static_cast<double>(h_star_light[flat]), static_cast<double>(st.light_balance_radius));
                std::fflush(stderr);
                printed = true;
            }
            if (diff)
                ++diff_count;
        }
        std::fprintf(stderr,
            "[direct-star-diff] diffCount=%d/%d maxMassDiff=%.12e maxOrbitDiff=%.12e maxHabDiff=%.12e maxLightDiff=%.12e\n",
            diff_count, seed0_count, max_mass_diff, max_orbit_diff, max_hab_diff, max_light_diff);
        std::fflush(stderr);
    }

    const int block_star = ResolveSigBlockSize();
    const int grid_star = (total_stars + block_star - 1) / block_star;
    if (!use_fused_plan_core_pack)
    {
        BuildStarPlanetPlansKernel<<<grid_star, block_star, 0, stream>>>(
        d_star_seeds,
        d_star_types,
        d_star_spectrs,
        d_star_indexes,
        total_stars,
        em.star_type_main_seq,
        em.star_type_giant,
        em.star_type_white_dwarf,
        em.star_type_neutron_star,
        em.star_type_black_hole,
        em.spectr_m,
        em.spectr_k,
        em.spectr_g,
        em.spectr_f,
        em.spectr_a,
        em.spectr_b,
        em.spectr_o,
        d_out_planet_counts,
        d_out_orbit_arounds,
        d_out_orbit_indexes,
        d_out_numbers,
        d_out_gas_giants,
        d_out_info_seeds,
        d_out_gen_seeds);
        if (cudaGetLastError() != cudaSuccess)
        {
            cleanup();
            return false;
        }

        if (debug_direct_plan && seed_count > 0)
        {
        std::vector<int> h_planet_counts_dbg(static_cast<size_t>(total_stars), 0);
        std::vector<int> h_orbit_arounds_dbg(plans_n, 0);
        std::vector<int> h_orbit_indexes_dbg(plans_n, 0);
        std::vector<int> h_numbers_dbg(plans_n, 0);
        std::vector<int> h_gas_dbg(plans_n, 0);
        std::vector<int> h_info_dbg(plans_n, 0);
        std::vector<int> h_gen_dbg(plans_n, 0);
        if (!d2h_raw(h_planet_counts_dbg.data(), d_out_planet_counts, static_cast<size_t>(total_stars) * sizeof(int)) ||
            !d2h_raw(h_orbit_arounds_dbg.data(), d_out_orbit_arounds, plans_n * sizeof(int)) ||
            !d2h_raw(h_orbit_indexes_dbg.data(), d_out_orbit_indexes, plans_n * sizeof(int)) ||
            !d2h_raw(h_numbers_dbg.data(), d_out_numbers, plans_n * sizeof(int)) ||
            !d2h_raw(h_gas_dbg.data(), d_out_gas_giants, plans_n * sizeof(int)) ||
            !d2h_raw(h_info_dbg.data(), d_out_info_seeds, plans_n * sizeof(int)) ||
            !d2h_raw(h_gen_dbg.data(), d_out_gen_seeds, plans_n * sizeof(int)) ||
            cudaStreamSynchronize(stream) != cudaSuccess)
        {
            cleanup();
            return false;
        }

        int plan_diff_count = 0;
        bool plan_first_printed = false;
        const int seed0_count = seed_star_counts[0];
        const int seed0_base = seed_star_offsets[0];
        for (int si = 0; si < seed0_count; ++si)
        {
            const int flat = seed0_base + si;
            StarCtx sc{};
            sc.star.id = h_star_ids[flat];
            sc.star.index = h_star_indexes_host[flat];
            sc.star.seed = h_star_seeds[flat];
            sc.star.type = h_star_types_host[flat];
            sc.star.spectr = h_star_spectrs_host[flat];
            BuildStarPlanetPlans(sc, em);
            const int gpu_pc = h_planet_counts_dbg[flat] < 0 ? 0 : (h_planet_counts_dbg[flat] > kStarPlanMaxPlanets ? kStarPlanMaxPlanets : h_planet_counts_dbg[flat]);
            const int host_pc = static_cast<int>(sc.planets.size());
            if (gpu_pc != host_pc)
            {
                ++plan_diff_count;
                if (!plan_first_printed)
                {
                    std::fprintf(stderr, "[direct-plan-diff] star=%d pc gpu/host=%d/%d\n", si, gpu_pc, host_pc);
                    std::fflush(stderr);
                    plan_first_printed = true;
                }
                continue;
            }
            const int base = flat * kStarPlanMaxPlanets;
            for (int pi = 0; pi < host_pc; ++pi)
            {
                const PlanetPlanLite& hp = sc.planets[pi];
                bool diff =
                    h_orbit_arounds_dbg[base + pi] != hp.orbit_around ||
                    h_orbit_indexes_dbg[base + pi] != hp.orbit_index ||
                    h_numbers_dbg[base + pi] != hp.number ||
                    h_gas_dbg[base + pi] != (hp.gas_giant ? 1 : 0) ||
                    h_info_dbg[base + pi] != hp.info_seed ||
                    h_gen_dbg[base + pi] != hp.gen_seed;
                if (diff)
                {
                    ++plan_diff_count;
                    if (!plan_first_printed)
                    {
                        std::fprintf(stderr,
                            "[direct-plan-diff] star=%d pi=%d orbitAround gpu/host=%d/%d orbitIndex=%d/%d number=%d/%d gas=%d/%d info=%d/%d gen=%d/%d\n",
                            si, pi,
                            h_orbit_arounds_dbg[base + pi], hp.orbit_around,
                            h_orbit_indexes_dbg[base + pi], hp.orbit_index,
                            h_numbers_dbg[base + pi], hp.number,
                            h_gas_dbg[base + pi], hp.gas_giant ? 1 : 0,
                            h_info_dbg[base + pi], hp.info_seed,
                            h_gen_dbg[base + pi], hp.gen_seed);
                        std::fflush(stderr);
                        plan_first_printed = true;
                    }
                }
            }
        }
        std::fprintf(stderr, "[direct-plan-diff] diffCount=%d\n", plan_diff_count);
        std::fflush(stderr);
    }

        if (debug_direct_core && seed_count > 0)
        {
        EvalPlanetCoreFromPlansKernel<<<grid_star, block_star, 0, stream>>>(
            d_star_types,
            d_star_indexes,
            d_star_counts_in_galaxy,
            d_star_orbit_scalers,
            d_star_masses,
            d_star_habitable_radiuses,
            d_star_light_balance_radiuses,
            d_out_planet_counts,
            d_out_orbit_arounds,
            d_out_orbit_indexes,
            d_out_numbers,
            d_out_gas_giants,
            d_out_info_seeds,
            total_stars,
            star_type_white_dwarf,
            star_type_neutron_star,
            star_type_black_hole,
            d_out_core);
        if (cudaGetLastError() != cudaSuccess)
        {
            cleanup();
            return false;
        }

        std::vector<int> h_planet_counts_dbg(static_cast<size_t>(total_stars), 0);
        std::vector<int> h_orbit_arounds_dbg(plans_n, 0);
        std::vector<int> h_orbit_indexes_dbg(plans_n, 0);
        std::vector<int> h_gas_dbg(plans_n, 0);
        std::vector<int> h_gen_dbg(plans_n, 0);
        std::vector<CoreLite> h_core_dbg(plans_n);
        if (!d2h_raw(h_planet_counts_dbg.data(), d_out_planet_counts, static_cast<size_t>(total_stars) * sizeof(int)) ||
            !d2h_raw(h_orbit_arounds_dbg.data(), d_out_orbit_arounds, plans_n * sizeof(int)) ||
            !d2h_raw(h_orbit_indexes_dbg.data(), d_out_orbit_indexes, plans_n * sizeof(int)) ||
            !d2h_raw(h_gas_dbg.data(), d_out_gas_giants, plans_n * sizeof(int)) ||
            !d2h_raw(h_gen_dbg.data(), d_out_gen_seeds, plans_n * sizeof(int)) ||
            !d2h_raw(h_core_dbg.data(), d_out_core, plans_n * sizeof(CoreLite)) ||
            cudaStreamSynchronize(stream) != cudaSuccess)
        {
            cleanup();
            return false;
        }

        std::vector<SeedCtx> seeds_ref(static_cast<size_t>(seed_count));
        for (int seed_idx = 0; seed_idx < seed_count; ++seed_idx)
        {
            SeedCtx& sctx = seeds_ref[seed_idx];
            sctx.galaxy_seed = galaxy_seeds[seed_idx];
            sctx.birth_star_id = 0;
            sctx.birth_planet_id = 0;
            sctx.star_count = seed_star_counts[seed_idx];
            sctx.stars.resize(static_cast<size_t>(sctx.star_count));
            const int base = seed_star_offsets[seed_idx];
            for (int si = 0; si < sctx.star_count; ++si)
            {
                const int flat = base + si;
                StarLite st{};
                st.id = h_star_ids[flat];
                st.index = h_star_indexes_host[flat];
                st.seed = h_star_seeds[flat];
                st.type = h_star_types_host[flat];
                st.spectr = h_star_spectrs_host[flat];
                st.planet_count = 0;
                st.mass = static_cast<float>(h_star_mass_host[flat]);
                st.orbit_scaler = h_star_orbit_host[flat];
                st.habitable_radius = h_star_hab_host[flat];
                st.light_balance_radius = h_star_light_host[flat];
                st.pos_qx = h_star_pos_qx_host[flat];
                st.pos_qy = h_star_pos_qy_host[flat];
                st.pos_qz = h_star_pos_qz_host[flat];
                sctx.stars[static_cast<size_t>(si)].star = st;
            }
        }

        std::vector<PlanetRef> pri_ref;
        std::vector<PlanetRef> sec_ref;
        std::vector<CoreLite> core_ref;
        std::vector<int> ref_planet_counts;
        std::vector<int> ref_orbit_arounds;
        std::vector<int> ref_orbit_indexes;
        std::vector<int> ref_gas;
        std::vector<int> ref_gen;
        bool ref_ok = TryBuildStarPlanetPlansCuda(
            device_id,
            em,
            seeds_ref,
            star_type_white_dwarf,
            star_type_neutron_star,
            star_type_black_hole,
            false,
            false,
            1,
            pri_ref,
            sec_ref,
            core_ref,
            nullptr,
            &ref_planet_counts,
            &ref_orbit_arounds,
            &ref_orbit_indexes,
            &ref_gas,
            &ref_gen,
            true);
        int core_diff = 0;
        if (ref_ok &&
            ref_planet_counts.size() == h_planet_counts_dbg.size() &&
            ref_orbit_arounds.size() == h_orbit_arounds_dbg.size() &&
            ref_orbit_indexes.size() == h_orbit_indexes_dbg.size() &&
            ref_gas.size() == h_gas_dbg.size() &&
            ref_gen.size() == h_gen_dbg.size() &&
            core_ref.size() == h_core_dbg.size())
        {
            for (size_t i = 0; i < h_planet_counts_dbg.size(); ++i)
            {
                if (ref_planet_counts[i] != h_planet_counts_dbg[i])
                    ++core_diff;
            }
            for (size_t i = 0; i < h_orbit_arounds_dbg.size(); ++i)
            {
                bool d = ref_orbit_arounds[i] != h_orbit_arounds_dbg[i] ||
                         ref_orbit_indexes[i] != h_orbit_indexes_dbg[i] ||
                         ref_gas[i] != h_gas_dbg[i] ||
                         ref_gen[i] != h_gen_dbg[i];
                const CoreLite& a = core_ref[i];
                const CoreLite& b = h_core_dbg[i];
                d = d ||
                    std::fabs(a.habitable_bias - b.habitable_bias) > 1e-6f ||
                    std::fabs(a.sun_distance - b.sun_distance) > 1e-6f ||
                    std::fabs(a.temperature_bias - b.temperature_bias) > 1e-6f ||
                    std::fabs(a.num13 - b.num13) > 1e-12 ||
                    std::fabs(a.num14 - b.num14) > 1e-12 ||
                    std::fabs(a.rand1 - b.rand1) > 1e-12;
                if (d)
                {
                    ++core_diff;
                    if (core_diff == 1)
                    {
                        std::fprintf(stderr,
                            "[direct-core-diff] idx=%zu ref(hab/sun/temp/num13/num14/rand1)=%.9f/%.9f/%.9f/%.12f/%.12f/%.12f gpu=%.9f/%.9f/%.9f/%.12f/%.12f/%.12f\n",
                            i,
                            static_cast<double>(a.habitable_bias),
                            static_cast<double>(a.sun_distance),
                            static_cast<double>(a.temperature_bias),
                            a.num13, a.num14, a.rand1,
                            static_cast<double>(b.habitable_bias),
                            static_cast<double>(b.sun_distance),
                            static_cast<double>(b.temperature_bias),
                            b.num13, b.num14, b.rand1);
                        std::fflush(stderr);
                    }
                }
            }
            std::fprintf(stderr, "[direct-core-diff] diffCount=%d\n", core_diff);
            std::fflush(stderr);

            SigGpuFlatSoA flat_ref{};
            bool flat_ok = BuildSigGpuFlatSoAFromPlan(
                seeds_ref,
                ref_planet_counts,
                ref_orbit_arounds,
                ref_orbit_indexes,
                ref_gas,
                ref_gen,
                core_ref,
                flat_ref);
            if (!flat_ok)
            {
                std::fprintf(stderr, "[direct-flat-diff] ref-flat-build-failed\n");
                std::fflush(stderr);
            }
            else
            {
                std::vector<int> h_star_planet_offsets_dbg(static_cast<size_t>(total_stars + 1), 0);
                std::vector<int> h_planet_ids_dbg(static_cast<size_t>(flat_ref.planet_ids.size()), 0);
                std::vector<int> h_planet_indexes_dbg(static_cast<size_t>(flat_ref.planet_indexes.size()), 0);
                std::vector<int> h_planet_orbit_indexes_dbg(static_cast<size_t>(flat_ref.planet_orbit_indexes.size()), 0);
                std::vector<int> h_planet_orbit_arounds_dbg(static_cast<size_t>(flat_ref.planet_orbit_arounds.size()), 0);
                std::vector<int> h_planet_gen_dbg2(static_cast<size_t>(flat_ref.planet_gen_seeds.size()), 0);
                std::vector<int> h_planet_gas_dbg2(static_cast<size_t>(flat_ref.planet_gas_giants.size()), 0);
                std::vector<float> h_planet_hab_dbg(static_cast<size_t>(flat_ref.planet_core_habitable_bias.size()), 0.0f);
                std::vector<float> h_planet_sun_dbg(static_cast<size_t>(flat_ref.planet_core_sun_distance.size()), 0.0f);
                std::vector<float> h_planet_temp_dbg(static_cast<size_t>(flat_ref.planet_core_temperature_bias.size()), 0.0f);
                std::vector<double> h_planet_num13_dbg(static_cast<size_t>(flat_ref.planet_core_num13.size()), 0.0);
                std::vector<double> h_planet_num14_dbg(static_cast<size_t>(flat_ref.planet_core_num14.size()), 0.0);
                std::vector<double> h_planet_rand1_dbg(static_cast<size_t>(flat_ref.planet_core_rand1.size()), 0.0);
                if (!d2h_raw(h_star_planet_offsets_dbg.data(), d_star_planet_offsets, static_cast<size_t>(total_stars + 1) * sizeof(int)) ||
                    !d2h_raw(h_planet_ids_dbg.data(), d_planet_ids, h_planet_ids_dbg.size() * sizeof(int)) ||
                    !d2h_raw(h_planet_indexes_dbg.data(), d_planet_indexes, h_planet_indexes_dbg.size() * sizeof(int)) ||
                    !d2h_raw(h_planet_orbit_indexes_dbg.data(), d_planet_orbit_indexes, h_planet_orbit_indexes_dbg.size() * sizeof(int)) ||
                    !d2h_raw(h_planet_orbit_arounds_dbg.data(), d_planet_orbit_arounds, h_planet_orbit_arounds_dbg.size() * sizeof(int)) ||
                    !d2h_raw(h_planet_gen_dbg2.data(), d_planet_gen_seeds, h_planet_gen_dbg2.size() * sizeof(int)) ||
                    !d2h_raw(h_planet_gas_dbg2.data(), d_planet_gas_giants, h_planet_gas_dbg2.size() * sizeof(int)) ||
                    !d2h_raw(h_planet_hab_dbg.data(), d_planet_core_habitable_bias, h_planet_hab_dbg.size() * sizeof(float)) ||
                    !d2h_raw(h_planet_sun_dbg.data(), d_planet_core_sun_distance, h_planet_sun_dbg.size() * sizeof(float)) ||
                    !d2h_raw(h_planet_temp_dbg.data(), d_planet_core_temperature_bias, h_planet_temp_dbg.size() * sizeof(float)) ||
                    !d2h_raw(h_planet_num13_dbg.data(), d_planet_core_num13, h_planet_num13_dbg.size() * sizeof(double)) ||
                    !d2h_raw(h_planet_num14_dbg.data(), d_planet_core_num14, h_planet_num14_dbg.size() * sizeof(double)) ||
                    !d2h_raw(h_planet_rand1_dbg.data(), d_planet_core_rand1, h_planet_rand1_dbg.size() * sizeof(double)) ||
                    cudaStreamSynchronize(stream) != cudaSuccess)
                {
                    cleanup();
                    return false;
                }
                int flat_diff = 0;
                for (size_t i = 0; i < flat_ref.star_planet_offsets.size(); ++i)
                {
                    if (flat_ref.star_planet_offsets[i] != h_star_planet_offsets_dbg[i])
                        ++flat_diff;
                }
                for (size_t i = 0; i < flat_ref.planet_ids.size(); ++i)
                {
                    bool d = flat_ref.planet_ids[i] != h_planet_ids_dbg[i] ||
                             flat_ref.planet_indexes[i] != h_planet_indexes_dbg[i] ||
                             flat_ref.planet_orbit_indexes[i] != h_planet_orbit_indexes_dbg[i] ||
                             flat_ref.planet_orbit_arounds[i] != h_planet_orbit_arounds_dbg[i] ||
                             flat_ref.planet_gen_seeds[i] != h_planet_gen_dbg2[i] ||
                             flat_ref.planet_gas_giants[i] != h_planet_gas_dbg2[i] ||
                             std::fabs(flat_ref.planet_core_habitable_bias[i] - h_planet_hab_dbg[i]) > 1e-6f ||
                             std::fabs(flat_ref.planet_core_sun_distance[i] - h_planet_sun_dbg[i]) > 1e-6f ||
                             std::fabs(flat_ref.planet_core_temperature_bias[i] - h_planet_temp_dbg[i]) > 1e-6f ||
                             std::fabs(flat_ref.planet_core_num13[i] - h_planet_num13_dbg[i]) > 1e-12 ||
                             std::fabs(flat_ref.planet_core_num14[i] - h_planet_num14_dbg[i]) > 1e-12 ||
                             std::fabs(flat_ref.planet_core_rand1[i] - h_planet_rand1_dbg[i]) > 1e-12;
                    if (d)
                    {
                        ++flat_diff;
                        if (flat_diff == 1)
                        {
                            std::fprintf(stderr,
                                "[direct-flat-diff] idx=%zu id gpu/ref=%d/%d orbitIdx=%d/%d orbitAround=%d/%d gen=%d/%d gas=%d/%d\n",
                                i,
                                h_planet_ids_dbg[i], flat_ref.planet_ids[i],
                                h_planet_orbit_indexes_dbg[i], flat_ref.planet_orbit_indexes[i],
                                h_planet_orbit_arounds_dbg[i], flat_ref.planet_orbit_arounds[i],
                                h_planet_gen_dbg2[i], flat_ref.planet_gen_seeds[i],
                                h_planet_gas_dbg2[i], flat_ref.planet_gas_giants[i]);
                            std::fflush(stderr);
                        }
                    }
                }
                std::fprintf(stderr, "[direct-flat-diff] diffCount=%d\n", flat_diff);
                std::fflush(stderr);
            }
        }
        else
        {
            std::fprintf(stderr, "[direct-core-diff] ref-build-failed-or-size-mismatch refOk=%d\n", ref_ok ? 1 : 0);
            std::fflush(stderr);
        }
    }

        ClampPlanetCountsKernel<<<grid_star, block_star, 0, stream>>>(d_out_planet_counts, total_stars);
        if (cudaGetLastError() != cudaSuccess)
        {
            cleanup();
            return false;
        }

        if (cudaMemsetAsync(d_total_planets_dev, 0, sizeof(int), stream) != cudaSuccess)
        {
            cleanup();
            return false;
        }
        BuildStarPlanetOffsetsAtomicKernel<<<grid_star, block_star, 0, stream>>>(
            d_out_planet_counts,
            total_stars,
            d_star_planet_offsets,
            d_total_planets_dev);
        if (cudaGetLastError() != cudaSuccess)
        {
            cleanup();
            return false;
        }
        FinalizeStarPlanetOffsetsKernel<<<1, 1, 0, stream>>>(
            total_stars,
            d_total_planets_dev,
            d_star_planet_offsets);
        if (cudaGetLastError() != cudaSuccess)
        {
            cleanup();
            return false;
        }

        EvalPlanetCoreAndPackCompactFromPlansKernel<<<grid_star, block_star, 0, stream>>>(
            d_star_ids,
            d_star_types,
            d_star_indexes,
            d_star_counts_in_galaxy,
            d_star_orbit_scalers,
            d_star_masses,
            d_star_habitable_radiuses,
            d_star_light_balance_radiuses,
            d_out_planet_counts,
            d_star_planet_offsets,
            d_out_orbit_arounds,
            d_out_orbit_indexes,
            d_out_numbers,
            d_out_gas_giants,
            d_out_info_seeds,
            d_out_gen_seeds,
            total_stars,
            star_type_white_dwarf,
            star_type_neutron_star,
            star_type_black_hole,
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
            d_planet_core_rand1);
        if (cudaGetLastError() != cudaSuccess)
        {
            cleanup();
            return false;
        }
    }
    else
    {
        if (cudaMemsetAsync(d_total_planets_dev, 0, sizeof(int), stream) != cudaSuccess)
        {
            cleanup();
            return false;
        }
        BuildPlanCorePackCompactKernel<<<grid_star, block_star, 0, stream>>>(
            d_star_ids,
            d_star_seeds,
            d_star_types,
            d_star_spectrs,
            d_star_indexes,
            d_star_counts_in_galaxy,
            d_star_orbit_scalers,
            d_star_masses,
            d_star_habitable_radiuses,
            d_star_light_balance_radiuses,
            total_stars,
            em.star_type_main_seq,
            em.star_type_giant,
            em.star_type_white_dwarf,
            em.star_type_neutron_star,
            em.star_type_black_hole,
            em.spectr_m,
            em.spectr_k,
            em.spectr_g,
            em.spectr_f,
            em.spectr_a,
            em.spectr_b,
            em.spectr_o,
            d_out_planet_counts,
            d_star_planet_offsets,
            d_total_planets_dev,
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
            d_planet_core_rand1);
        if (cudaGetLastError() != cudaSuccess)
        {
            cleanup();
            return false;
        }
        FinalizeStarPlanetOffsetsKernel<<<1, 1, 0, stream>>>(
            total_stars,
            d_total_planets_dev,
            d_star_planet_offsets);
        if (cudaGetLastError() != cudaSuccess)
        {
            cleanup();
            return false;
        }
    }

    EvalThemeVeinHashKernel<<<grid_seed, block_seed, 0, stream>>>(
        seed_count,
        vein_len,
        use_fp32_prob_compare,
        em,
        d_seed_star_offsets,
        d_star_planet_offsets,
        d_star_ids,
        d_star_types,
        d_star_spectrs,
        d_out_planet_counts,
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
        nullptr,
        nullptr,
        nullptr);
    if (cudaGetLastError() != cudaSuccess)
    {
        cleanup();
        return false;
    }

    bool copy_ok = d2h_raw(out_galaxy_sigs, d_out_galaxy_sigs, static_cast<size_t>(seed_count) * sizeof(unsigned long long)) &&
                   d2h_raw(out_planet_sigs, d_out_planet_sigs, static_cast<size_t>(seed_count) * sizeof(unsigned long long)) &&
                   d2h_raw(out_vein_sigs, d_out_vein_sigs, static_cast<size_t>(seed_count) * sizeof(unsigned long long)) &&
                   d2h_raw(out_pipeline_sigs, d_out_pipeline_sigs, static_cast<size_t>(seed_count) * sizeof(unsigned long long));
    if (!copy_ok || cudaStreamSynchronize(stream) != cudaSuccess)
    {
        cleanup();
        return false;
    }

    cleanup();
    return true;
}

