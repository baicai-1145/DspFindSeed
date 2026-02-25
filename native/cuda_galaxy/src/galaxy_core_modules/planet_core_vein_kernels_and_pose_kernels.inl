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
    int* out_counts,
    PoseGenSeedProfile* out_profiles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seed_count)
        return;

    dsp_vec3d_t* poses_seg = out_poses + static_cast<long long>(idx) * out_stride;
    dsp_vec3d_t* drunk_seg = out_drunk + static_cast<long long>(idx) * out_stride;
    int count = 0;
    if (out_profiles != nullptr)
    {
        if (collision_fp64 != 0)
        {
            count = GenerateRandomPosesParamsFp64Impl<true, true>(
                seeds[idx],
                max_count,
                min_dist,
                min_step_len,
                max_step_len,
                flatten,
                poses_seg,
                drunk_seg,
                out_profiles + idx);
        }
        else
        {
            count = GenerateRandomPosesParamsFp64Impl<true, false>(
                seeds[idx],
                max_count,
                min_dist,
                min_step_len,
                max_step_len,
                flatten,
                poses_seg,
                drunk_seg,
                out_profiles + idx);
        }
    }
    else
    {
        if (collision_fp64 != 0)
        {
            count = GenerateRandomPosesParamsFp64Impl<false, true>(
                seeds[idx],
                max_count,
                min_dist,
                min_step_len,
                max_step_len,
                flatten,
                poses_seg,
                drunk_seg,
                nullptr);
        }
        else
        {
            count = GenerateRandomPosesParamsFp64Impl<false, false>(
                seeds[idx],
                max_count,
                min_dist,
                min_step_len,
                max_step_len,
                flatten,
                poses_seg,
                drunk_seg,
                nullptr);
        }
    }
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
    out_result->num13 = num13;
    out_result->num14 = num14;
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

__global__ void RefreshPlanetVeinSpotsByThemeBatchKernel(
    const int* planet_seeds,
    const float* p_values,
    const int* bonus_cases,
    const int* is_birth_stars,
    const int* theme_indexes,
    int planet_count,
    int out_vein_len,
    int use_fp32_prob_compare,
    const int* theme_vein_spot_offsets,
    const int* theme_vein_spot_values,
    const int* theme_rare_vein_offsets,
    const int* theme_rare_vein_values,
    const int* theme_rare_settings_offsets,
    const float* theme_rare_settings_values,
    int theme_count,
    int* out_counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= planet_count)
        return;

    int* out_seg = out_counts + static_cast<long long>(idx) * out_vein_len;
    for (int i = 0; i < out_vein_len; ++i)
        out_seg[i] = 0;

    int theme_idx = theme_indexes[idx];
    if (theme_idx < 0 || theme_idx >= theme_count)
        return;

    int vein_begin = theme_vein_spot_offsets[theme_idx];
    int vein_end = theme_vein_spot_offsets[theme_idx + 1];
    int vlen = vein_end - vein_begin;
    if (vlen < 0)
        vlen = 0;
    for (int i = 0; i < vlen; ++i)
    {
        int out_idx = i + 1;
        if (out_idx >= 0 && out_idx < out_vein_len)
            out_seg[out_idx] = theme_vein_spot_values[vein_begin + i];
    }

    DotNet35RandomDevice rng(planet_seeds[idx]);
    rng.InternalSample();
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

    const bool is_birth = is_birth_stars[idx] != 0;
    const float p = p_values[idx];
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

        float appear_base = is_birth ? s0 : s1;
        float chain_prob = s2;
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

__global__ void GatherTempPosesHeadKernel(
    const dsp_vec3d_t* in_poses,
    int seed_count,
    int in_stride,
    int head_count,
    int sample_step,
    dsp_vec3d_t* out_poses,
    int out_stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seed_count * head_count;
    if (idx >= total)
        return;
    int seed_idx = idx / head_count;
    int local_idx = idx - seed_idx * head_count;
    int src_idx = local_idx * sample_step;
    out_poses[seed_idx * out_stride + local_idx] = in_poses[seed_idx * in_stride + src_idx];
}
} // namespace
