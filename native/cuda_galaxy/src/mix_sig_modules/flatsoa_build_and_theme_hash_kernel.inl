__global__ void EvalThemeVeinHashKernel(
    int seed_count,
    int vein_len,
    int use_fp32_prob_compare,
    EnumMap em,
    const int* seed_star_offsets,
    const int* star_planet_offsets,
    const int* star_ids,
    const int* star_types,
    const int* star_spectrs,
    const int* star_planet_counts,
    const int* star_pos_qx,
    const int* star_pos_qy,
    const int* star_pos_qz,
    const int* star_indexes,
    const float* star_habitable_radiuses,
    const int* planet_ids,
    const int* planet_indexes,
    const int* planet_orbit_indexes,
    const int* planet_orbit_arounds,
    const int* planet_gen_seeds,
    const int* planet_gas_giants,
    const float* planet_core_habitable_bias,
    const float* planet_core_sun_distance,
    const float* planet_core_temperature_bias,
    const double* planet_core_num13,
    const double* planet_core_num14,
    const double* planet_core_rand1,
    int theme_count,
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
    int max_planet_type,
    const int* type_theme_offsets,
    const int* type_theme_values,
    unsigned long long* out_galaxy_sigs,
    unsigned long long* out_planet_sigs,
    unsigned long long* out_vein_sigs,
    unsigned long long* out_pipeline_sigs,
    int* out_birth_star_ids,
    int* out_birth_planet_ids,
    int* out_debug_theme_indexes)
{
    int seed_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed_idx < 0 || seed_idx >= seed_count)
        return;

    constexpr int kMaxAssigned = 8;
    constexpr int kMaxCandidates = 128;
    int assigned_theme_ids[kMaxAssigned];
    int vein_counts_local[32];
    int candidates[kMaxCandidates];

    const int star_begin = seed_star_offsets[seed_idx];
    const int star_end = seed_star_offsets[seed_idx + 1];
    const int galaxy_star_count = star_end - star_begin;

    int birth_star_id = 0;
    int birth_planet_id = 0;
    int habitable_count_seed = 0;

    for (int star_flat = star_begin; star_flat < star_end && birth_planet_id == 0; ++star_flat)
    {
        const int sid = star_ids[star_flat];
        (void)star_types;
        (void)star_spectrs;
        const int sindex = star_indexes[star_flat];
        const float shab = star_habitable_radiuses[star_flat];

        for (int i = 0; i < kMaxAssigned; ++i)
            assigned_theme_ids[i] = 0;

        const int pbegin = star_planet_offsets[star_flat];
        int pcount = star_planet_counts[star_flat];
        if (pcount < 0) pcount = 0;
        if (pcount > kStarPlanMaxPlanets) pcount = kStarPlanMaxPlanets;
        const int pend = pbegin + pcount;
        for (int pflat = pbegin; pflat < pend; ++pflat)
        {
            const int pid = planet_ids[pflat];
            const int pindex = planet_indexes[pflat];
            const int porbit_index = planet_orbit_indexes[pflat];
            const int porbit_around = planet_orbit_arounds[pflat];
            const bool gas_giant = planet_gas_giants[pflat] != 0;
            const float p_hab_bias = planet_core_habitable_bias[pflat];
            const float p_sun_dist = planet_core_sun_distance[pflat];
            const float p_temp_bias = planet_core_temperature_bias[pflat];
            const double p_num13 = planet_core_num13[pflat];
            const double p_num14 = planet_core_num14[pflat];
            const double p_rand1 = planet_core_rand1[pflat];

            int type_case = 0;
            int habitable_delta = 0;
            if (!gas_giant)
            {
                float num18 = ceilf(static_cast<float>(galaxy_star_count) * 0.29f);
                if (num18 < 11.0f) num18 = 11.0f;
                float num19 = num18 - static_cast<float>(habitable_count_seed);
                float num20 = static_cast<float>(galaxy_star_count - sindex);
                float num24 = (num19 / fmaxf(1.0f, num20)) * 0.5f + 0.175f;
                if (num24 < 0.08f) num24 = 0.08f;
                if (num24 > 0.8f) num24 = 0.8f;
                float num25 = powf(DClamp01(p_hab_bias / num24), num24 * 10.0f);
                float f2 = 1000.0f;
                if (shab > 0.0f && p_sun_dist > 0.0f)
                    f2 = p_sun_dist / shab;

                if ((p_num13 > static_cast<double>(num25) && sindex > 0) ||
                    (porbit_around > 0 && porbit_index == 1 && sindex == 0))
                {
                    type_case = 1;
                    habitable_delta = 1;
                }
                else if (f2 < 0.833333f)
                {
                    float num26 = fmaxf(0.15f, f2 * 2.5f - 0.85f);
                    type_case = (p_num14 >= static_cast<double>(num26)) ? 2 : 3;
                }
                else if (f2 < 1.2f)
                {
                    type_case = 3;
                }
                else
                {
                    float num27 = 0.9f / f2 - 0.1f;
                    type_case = (p_num14 >= static_cast<double>(num27)) ? 4 : 3;
                }
            }

            const int provisional_type = DMapTypeCaseToPlanetType(type_case, em);
            if (habitable_delta > 0)
                ++habitable_count_seed;

            int candidate_count = 0;
            if (provisional_type >= 0 && provisional_type <= max_planet_type)
            {
                int tb = type_theme_offsets[provisional_type];
                int te = type_theme_offsets[provisional_type + 1];
                for (int k = tb; k < te; ++k)
                {
                    int ti = type_theme_values[k];
                    bool ok = false;
                    if (sindex == 0 && provisional_type == em.planet_type_ocean)
                    {
                        ok = theme_distributes[ti] == em.theme_distribute_birth;
                    }
                    else
                    {
                        const double temp_gate = -0.10000000149011612;
                        const double temp_gate_eps = 2e-8;
                        double temp_prod = static_cast<double>(theme_temperatures[ti]) * static_cast<double>(p_temp_bias);
                        bool temp_ok = temp_prod >= (temp_gate + temp_gate_eps);
                        if (fabsf(theme_temperatures[ti]) < 0.5f && theme_planet_types[ti] == em.planet_type_desert)
                            temp_ok = fabsf(p_temp_bias) < fabsf(theme_temperatures[ti]) + 0.1f;

                        if (theme_planet_types[ti] == provisional_type && temp_ok)
                        {
                            if (sindex == 0)
                                ok = theme_distributes[ti] == em.theme_distribute_default;
                            else
                                ok = theme_distributes[ti] == em.theme_distribute_default || theme_distributes[ti] == em.theme_distribute_interstellar;
                        }
                    }
                    if (ok)
                    {
                        if (ti >= 0 && ti < theme_count)
                        {
                            const int theme_id = theme_ids[ti];
                            bool unique_ok = true;
                            int lim = pindex;
                            if (lim > kMaxAssigned)
                                lim = kMaxAssigned;
                            for (int j = 0; j < lim; ++j)
                            {
                                if (assigned_theme_ids[j] == theme_id)
                                {
                                    unique_ok = false;
                                    break;
                                }
                            }
                            if (unique_ok && candidate_count < kMaxCandidates)
                                candidates[candidate_count++] = ti;
                        }
                    }
                }
            }
            if (candidate_count == 0 && em.planet_type_desert >= 0 && em.planet_type_desert <= max_planet_type)
            {
                int tb = type_theme_offsets[em.planet_type_desert];
                int te = type_theme_offsets[em.planet_type_desert + 1];
                for (int k = tb; k < te; ++k)
                {
                    int ti = type_theme_values[k];
                    if (ti < 0 || ti >= theme_count)
                        continue;
                    const int theme_id = theme_ids[ti];
                    bool unique_ok = true;
                    int lim = pindex;
                    if (lim > kMaxAssigned)
                        lim = kMaxAssigned;
                    for (int j = 0; j < lim; ++j)
                    {
                        if (assigned_theme_ids[j] == theme_id)
                        {
                            unique_ok = false;
                            break;
                        }
                    }
                    if (unique_ok && candidate_count < kMaxCandidates)
                        candidates[candidate_count++] = ti;
                }
            }
            if (candidate_count == 0 && em.planet_type_desert >= 0 && em.planet_type_desert <= max_planet_type)
            {
                int tb = type_theme_offsets[em.planet_type_desert];
                int te = type_theme_offsets[em.planet_type_desert + 1];
                for (int k = tb; k < te; ++k)
                {
                    int ti = type_theme_values[k];
                    if (ti < 0 || ti >= theme_count)
                        continue;
                    if (candidate_count < kMaxCandidates)
                        candidates[candidate_count++] = ti;
                }
            }

            int theme_idx = 0;
            if (out_debug_theme_indexes != nullptr && seed_idx == 0 && star_flat == star_begin && pindex == 3 && candidate_count == 0)
            {
                printf("[kernel-theme-debug] seed=%d starFlat=%d pindex=%d provisional=%d candN=0 rand1=%.17g\n",
                    seed_idx, star_flat, pindex, provisional_type, p_rand1);
            }
            if (candidate_count > 0)
            {
                int pick = static_cast<int>(p_rand1 * static_cast<double>(candidate_count));
                if (pick < 0) pick = 0;
                pick %= candidate_count;
                theme_idx = candidates[pick];
                if (out_debug_theme_indexes != nullptr && seed_idx == 0 && star_flat == star_begin && pindex == 3)
                {
                    printf("[kernel-theme-debug] seed=%d starFlat=%d pindex=%d provisional=%d candN=%d pick=%d rand1=%.17g cand0=%d cand1=%d cand2=%d cand3=%d\n",
                        seed_idx,
                        star_flat,
                        pindex,
                        provisional_type,
                        candidate_count,
                        pick,
                        p_rand1,
                        candidate_count > 0 ? candidates[0] : -1,
                        candidate_count > 1 ? candidates[1] : -1,
                        candidate_count > 2 ? candidates[2] : -1,
                        candidate_count > 3 ? candidates[3] : -1);
                }
            }
            if (theme_idx < 0 || theme_idx >= theme_count)
                theme_idx = 0;
            if (out_debug_theme_indexes != nullptr)
            {
                int dbg_v = theme_idx;
                if (seed_idx == 0 && pflat == 3)
                    dbg_v = 1000000 + candidate_count * 100 + theme_idx;
                out_debug_theme_indexes[pflat] = dbg_v;
            }

            if (pindex >= 0 && pindex < kMaxAssigned)
                assigned_theme_ids[pindex] = theme_ids[theme_idx];

            if (birth_planet_id == 0 && sindex == 0 && theme_distributes[theme_idx] == em.theme_distribute_birth)
            {
                birth_planet_id = pid;
                birth_star_id = sid;
                break;
            }
        }
    }

    unsigned long long h_galaxy = kFnvOffset;
    unsigned long long h_planet = kFnvOffset;
    unsigned long long h_vein = kFnvOffset;
    unsigned long long h_pipeline = kFnvOffset;

    h_galaxy = MixHashDevice(h_galaxy, galaxy_star_count);
    h_planet = MixHashDevice(h_planet, galaxy_star_count);
    h_planet = MixHashDevice(h_planet, birth_star_id);
    h_planet = MixHashDevice(h_planet, birth_planet_id);
    h_vein = MixHashDevice(h_vein, galaxy_star_count);
    h_pipeline = MixHashDevice(h_pipeline, galaxy_star_count);
    h_pipeline = MixHashDevice(h_pipeline, birth_star_id);
    h_pipeline = MixHashDevice(h_pipeline, birth_planet_id);

    habitable_count_seed = 0;
    for (int star_flat = star_begin; star_flat < star_end; ++star_flat)
    {
        const int sid = star_ids[star_flat];
        const int stype = star_types[star_flat];
        const int sspectr = star_spectrs[star_flat];
        const int spcnt = star_planet_counts[star_flat];
        const int sx = star_pos_qx[star_flat];
        const int sy = star_pos_qy[star_flat];
        const int sz = star_pos_qz[star_flat];
        const int sindex = star_indexes[star_flat];
        const float shab = star_habitable_radiuses[star_flat];

        h_galaxy = MixHashDevice(h_galaxy, sid);
        h_galaxy = MixHashDevice(h_galaxy, stype);
        h_galaxy = MixHashDevice(h_galaxy, sspectr);
        h_galaxy = MixHashDevice(h_galaxy, spcnt);
        h_galaxy = MixHashDevice(h_galaxy, sx);
        h_galaxy = MixHashDevice(h_galaxy, sy);
        h_galaxy = MixHashDevice(h_galaxy, sz);

        h_planet = MixHashDevice(h_planet, sid);
        h_planet = MixHashDevice(h_planet, stype);
        h_planet = MixHashDevice(h_planet, sspectr);
        h_planet = MixHashDevice(h_planet, spcnt);
        h_planet = MixHashDevice(h_planet, sx);
        h_planet = MixHashDevice(h_planet, sy);
        h_planet = MixHashDevice(h_planet, sz);

        h_vein = MixHashDevice(h_vein, sid);

        h_pipeline = MixHashDevice(h_pipeline, sid);
        h_pipeline = MixHashDevice(h_pipeline, stype);
        h_pipeline = MixHashDevice(h_pipeline, sspectr);
        h_pipeline = MixHashDevice(h_pipeline, spcnt);
        h_pipeline = MixHashDevice(h_pipeline, sx);
        h_pipeline = MixHashDevice(h_pipeline, sy);
        h_pipeline = MixHashDevice(h_pipeline, sz);

        for (int i = 0; i < kMaxAssigned; ++i)
            assigned_theme_ids[i] = 0;

        const int pbegin = star_planet_offsets[star_flat];
        int pcount = star_planet_counts[star_flat];
        if (pcount < 0) pcount = 0;
        if (pcount > kStarPlanMaxPlanets) pcount = kStarPlanMaxPlanets;
        const int pend = pbegin + pcount;
        for (int pflat = pbegin; pflat < pend; ++pflat)
        {
            const int pid = planet_ids[pflat];
            const int pindex = planet_indexes[pflat];
            const int porbit_index = planet_orbit_indexes[pflat];
            const int porbit_around = planet_orbit_arounds[pflat];
            const int pgen_seed = planet_gen_seeds[pflat];
            const bool gas_giant = planet_gas_giants[pflat] != 0;
            const float p_hab_bias = planet_core_habitable_bias[pflat];
            const float p_sun_dist = planet_core_sun_distance[pflat];
            const float p_temp_bias = planet_core_temperature_bias[pflat];
            const double p_num13 = planet_core_num13[pflat];
            const double p_num14 = planet_core_num14[pflat];
            const double p_rand1 = planet_core_rand1[pflat];

            int type_case = 0;
            int habitable_delta = 0;
            if (!gas_giant)
            {
                float num18 = ceilf(static_cast<float>(galaxy_star_count) * 0.29f);
                if (num18 < 11.0f) num18 = 11.0f;
                float num19 = num18 - static_cast<float>(habitable_count_seed);
                float num20 = static_cast<float>(galaxy_star_count - sindex);
                float num24 = (num19 / fmaxf(1.0f, num20)) * 0.5f + 0.175f;
                if (num24 < 0.08f) num24 = 0.08f;
                if (num24 > 0.8f) num24 = 0.8f;
                float num25 = powf(DClamp01(p_hab_bias / num24), num24 * 10.0f);
                float f2 = 1000.0f;
                if (shab > 0.0f && p_sun_dist > 0.0f)
                    f2 = p_sun_dist / shab;

                if ((p_num13 > static_cast<double>(num25) && sindex > 0) ||
                    (porbit_around > 0 && porbit_index == 1 && sindex == 0))
                {
                    type_case = 1;
                    habitable_delta = 1;
                }
                else if (f2 < 0.833333f)
                {
                    float num26 = fmaxf(0.15f, f2 * 2.5f - 0.85f);
                    type_case = (p_num14 >= static_cast<double>(num26)) ? 2 : 3;
                }
                else if (f2 < 1.2f)
                {
                    type_case = 3;
                }
                else
                {
                    float num27 = 0.9f / f2 - 0.1f;
                    type_case = (p_num14 >= static_cast<double>(num27)) ? 4 : 3;
                }
            }

            const int provisional_type = DMapTypeCaseToPlanetType(type_case, em);
            if (habitable_delta > 0)
                ++habitable_count_seed;

            int candidate_count = 0;

            if (provisional_type >= 0 && provisional_type <= max_planet_type)
            {
                int tb = type_theme_offsets[provisional_type];
                int te = type_theme_offsets[provisional_type + 1];
                for (int k = tb; k < te; ++k)
                {
                    int ti = type_theme_values[k];
                    bool ok = false;
                    if (sindex == 0 && provisional_type == em.planet_type_ocean)
                    {
                        ok = theme_distributes[ti] == em.theme_distribute_birth;
                    }
                    else
                    {
                        const double temp_gate = -0.10000000149011612;
                        const double temp_gate_eps = 2e-8;
                        double temp_prod = static_cast<double>(theme_temperatures[ti]) * static_cast<double>(p_temp_bias);
                        bool temp_ok = temp_prod >= (temp_gate + temp_gate_eps);
                        if (fabsf(theme_temperatures[ti]) < 0.5f && theme_planet_types[ti] == em.planet_type_desert)
                            temp_ok = fabsf(p_temp_bias) < fabsf(theme_temperatures[ti]) + 0.1f;

                        if (theme_planet_types[ti] == provisional_type && temp_ok)
                        {
                            if (sindex == 0)
                                ok = theme_distributes[ti] == em.theme_distribute_default;
                            else
                                ok = theme_distributes[ti] == em.theme_distribute_default || theme_distributes[ti] == em.theme_distribute_interstellar;
                        }
                    }
                    if (ok)
                    {
                        if (ti >= 0 && ti < theme_count)
                        {
                            const int theme_id = theme_ids[ti];
                            bool unique_ok = true;
                            int lim = pindex;
                            if (lim > kMaxAssigned)
                                lim = kMaxAssigned;
                            for (int j = 0; j < lim; ++j)
                            {
                                if (assigned_theme_ids[j] == theme_id)
                                {
                                    unique_ok = false;
                                    break;
                                }
                            }
                            if (unique_ok && candidate_count < kMaxCandidates)
                                candidates[candidate_count++] = ti;
                        }
                    }
                }
            }
            if (candidate_count == 0 && em.planet_type_desert >= 0 && em.planet_type_desert <= max_planet_type)
            {
                int tb = type_theme_offsets[em.planet_type_desert];
                int te = type_theme_offsets[em.planet_type_desert + 1];
                for (int k = tb; k < te; ++k)
                {
                    int ti = type_theme_values[k];
                    if (ti < 0 || ti >= theme_count)
                        continue;
                    const int theme_id = theme_ids[ti];
                    bool unique_ok = true;
                    int lim = pindex;
                    if (lim > kMaxAssigned)
                        lim = kMaxAssigned;
                    for (int j = 0; j < lim; ++j)
                    {
                        if (assigned_theme_ids[j] == theme_id)
                        {
                            unique_ok = false;
                            break;
                        }
                    }
                    if (unique_ok && candidate_count < kMaxCandidates)
                        candidates[candidate_count++] = ti;
                }
            }
            if (candidate_count == 0 && em.planet_type_desert >= 0 && em.planet_type_desert <= max_planet_type)
            {
                int tb = type_theme_offsets[em.planet_type_desert];
                int te = type_theme_offsets[em.planet_type_desert + 1];
                for (int k = tb; k < te; ++k)
                {
                    int ti = type_theme_values[k];
                    if (ti < 0 || ti >= theme_count)
                        continue;
                    if (candidate_count < kMaxCandidates)
                        candidates[candidate_count++] = ti;
                }
            }

            int theme_idx = 0;
            if (candidate_count > 0)
            {
                int pick = static_cast<int>(p_rand1 * static_cast<double>(candidate_count));
                if (pick < 0) pick = 0;
                pick %= candidate_count;
                theme_idx = candidates[pick];
            }
            if (theme_idx < 0 || theme_idx >= theme_count)
                theme_idx = 0;
            if (out_debug_theme_indexes != nullptr)
                out_debug_theme_indexes[pflat] = theme_idx;

            const int ptype = theme_planet_types[theme_idx];
            const int ptheme = theme_ids[theme_idx];
            const int pwater = theme_water_item_ids[theme_idx];
            const bool is_gas_final = ptype == em.planet_type_gas;

            if (pindex >= 0 && pindex < kMaxAssigned)
                assigned_theme_ids[pindex] = ptheme;

            if (birth_planet_id == 0 && sindex == 0 && theme_distributes[theme_idx] == em.theme_distribute_birth)
            {
                birth_planet_id = pid;
                birth_star_id = sid;
            }

            h_planet = MixHashDevice(h_planet, ptype);
            h_planet = MixHashDevice(h_planet, ptheme);
            h_planet = MixHashDevice(h_planet, pwater);
            h_planet = MixHashDevice(h_planet, porbit_index);
            h_planet = MixHashDevice(h_planet, porbit_around);

            h_vein = MixHashDevice(h_vein, pid);

            h_pipeline = MixHashDevice(h_pipeline, ptype);
            h_pipeline = MixHashDevice(h_pipeline, ptheme);
            h_pipeline = MixHashDevice(h_pipeline, pwater);
            h_pipeline = MixHashDevice(h_pipeline, porbit_index);
            h_pipeline = MixHashDevice(h_pipeline, porbit_around);

            if (is_gas_final)
            {
                h_vein = MixHashDevice(h_vein, 0);
                continue;
            }

            int vmax = vein_len < 32 ? vein_len : 32;
            for (int vid = 0; vid < vmax; ++vid)
                vein_counts_local[vid] = 0;

            int vein_begin = theme_vein_spot_offsets[theme_idx];
            int vein_end = theme_vein_spot_offsets[theme_idx + 1];
            int vlen = vein_end - vein_begin;
            if (vlen < 0)
                vlen = 0;
            for (int i = 0; i < vlen; ++i)
            {
                int out_idx = i + 1;
                if (out_idx >= 0 && out_idx < vmax)
                    vein_counts_local[out_idx] = theme_vein_spot_values[vein_begin + i];
            }

            DotNet35RandomDeviceLite rng(pgen_seed);
            rng.InternalSample();
            rng.InternalSample();
            rng.InternalSample();
            rng.InternalSample();
            rng.InternalSample();
            DotNet35RandomDeviceLite rng2(rng.InternalSample());
            (void)rng2;
            int bonus_case = 0;
            float p = DCalcPAndBonusCase(stype, sspectr, em, bonus_case);
            const bool use_fp32 = use_fp32_prob_compare != 0;
            const bool is_birth_star = sindex == 0;

            if (bonus_case == 1)
            {
                if (9 < vmax)
                {
                    ++vein_counts_local[9];
                    ++vein_counts_local[9];
                    for (int i = 1; i < 12 && ProbHitDevice(rng, 0.449999988079071f, use_fp32); ++i)
                        ++vein_counts_local[9];
                }
                if (10 < vmax)
                {
                    ++vein_counts_local[10];
                    ++vein_counts_local[10];
                    for (int i = 1; i < 12 && ProbHitDevice(rng, 0.449999988079071f, use_fp32); ++i)
                        ++vein_counts_local[10];
                }
                if (12 < vmax)
                {
                    ++vein_counts_local[12];
                    for (int i = 1; i < 12 && ProbHitDevice(rng, 0.5f, use_fp32); ++i)
                        ++vein_counts_local[12];
                }
            }
            else if (bonus_case == 2)
            {
                if (14 < vmax)
                {
                    ++vein_counts_local[14];
                    for (int i = 1; i < 12 && ProbHitDevice(rng, 0.649999976158142f, use_fp32); ++i)
                        ++vein_counts_local[14];
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

                float appear_base = is_birth_star ? s0 : s1;
                float chain_prob = s2;
                float appear_prob = 1.0f - powf(1.0f - appear_base, p);
                if (ProbHitDevice(rng, appear_prob, use_fp32))
                {
                    if (vein_id > 0 && vein_id < vmax)
                        ++vein_counts_local[vein_id];
                    for (int i = 1; i < 12 && ProbHitDevice(rng, chain_prob, use_fp32); ++i)
                    {
                        if (vein_id > 0 && vein_id < vmax)
                            ++vein_counts_local[vein_id];
                    }
                }
            }

            for (int vid = 1; vid < vmax; ++vid)
            {
                int vv = vein_counts_local[vid];
                h_vein = MixHashDevice(h_vein, vv);
                h_pipeline = MixHashDevice(h_pipeline, vv);
            }
        }
    }

    out_planet_sigs[seed_idx] = h_planet;
    out_vein_sigs[seed_idx] = h_vein;
    out_galaxy_sigs[seed_idx] = h_galaxy;
    out_pipeline_sigs[seed_idx] = h_pipeline;
    if (out_birth_star_ids != nullptr)
        out_birth_star_ids[seed_idx] = birth_star_id;
    if (out_birth_planet_ids != nullptr)
        out_birth_planet_ids[seed_idx] = birth_planet_id;
}

