__global__ void BuildStarPlanetPlansKernel(
    const int* star_seeds,
    const int* star_types,
    const int* star_spectrs,
    const int* star_indexes,
    int star_count,
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
    int* out_planet_counts,
    int* out_orbit_arounds,
    int* out_orbit_indexes,
    int* out_numbers,
    int* out_gas_giants,
    int* out_info_seeds,
    int* out_gen_seeds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 0 || idx >= star_count)
        return;

    const int base = idx * kStarPlanMaxPlanets;
    for (int i = 0; i < kStarPlanMaxPlanets; ++i)
    {
        out_orbit_arounds[base + i] = 0;
        out_orbit_indexes[base + i] = 0;
        out_numbers[base + i] = 0;
        out_gas_giants[base + i] = 0;
        if (out_info_seeds != nullptr)
            out_info_seeds[base + i] = 0;
        out_gen_seeds[base + i] = 0;
    }
    out_planet_counts[idx] = 0;

    auto push_plan = [&](int orbit_around, int orbit_index, int number, int gas_giant, int info_seed, int gen_seed) {
        int pidx = out_planet_counts[idx];
        if (pidx < 0 || pidx >= kStarPlanMaxPlanets)
        {
            out_planet_counts[idx] = -1;
            return;
        }
        out_orbit_arounds[base + pidx] = orbit_around;
        out_orbit_indexes[base + pidx] = orbit_index;
        out_numbers[base + pidx] = number;
        out_gas_giants[base + pidx] = gas_giant;
        if (out_info_seeds != nullptr)
            out_info_seeds[base + pidx] = info_seed;
        out_gen_seeds[base + pidx] = gen_seed;
        out_planet_counts[idx] = pidx + 1;
    };

    const int star_seed = star_seeds[idx];
    const int star_type = star_types[idx];
    const int star_spectr = star_spectrs[idx];
    const int star_index = star_indexes[idx];

    DotNet35RandomDeviceLite rng1(star_seed);
    rng1.Next();
    rng1.Next();
    rng1.Next();
    DotNet35RandomDeviceLite rng2(rng1.Next());

    double num1 = rng2.NextDouble();
    double num2 = rng2.NextDouble();
    double num3 = rng2.NextDouble();
    double num4 = rng2.NextDouble();
    double num5 = rng2.NextDouble();
    (void)num4;
    (void)num5;
    rng2.NextDouble();
    rng2.NextDouble();

    if (star_type == star_type_black_hole || star_type == star_type_neutron_star)
    {
        push_plan(0, 3, 1, 0, rng2.Next(), rng2.Next());
        return;
    }

    if (star_type == star_type_white_dwarf)
    {
        if (num1 < 0.7)
        {
            push_plan(0, 3, 1, 0, rng2.Next(), rng2.Next());
        }
        else
        {
            if (num2 < 0.30000001192092896)
            {
                push_plan(0, 3, 1, 0, rng2.Next(), rng2.Next());
                push_plan(0, 4, 2, 0, rng2.Next(), rng2.Next());
            }
            else
            {
                push_plan(0, 4, 1, 1, rng2.Next(), rng2.Next());
                push_plan(1, 1, 1, 0, rng2.Next(), rng2.Next());
            }
        }
        return;
    }

    if (star_type == star_type_giant)
    {
        if (num1 < 0.30000001192092896)
        {
            push_plan(0, num3 > 0.5 ? 3 : 2, 1, 0, rng2.Next(), rng2.Next());
        }
        else if (num1 < 0.800000011920929)
        {
            if (num2 < 0.25)
            {
                push_plan(0, num3 > 0.5 ? 3 : 2, 1, 0, rng2.Next(), rng2.Next());
                push_plan(0, num3 > 0.5 ? 4 : 3, 2, 0, rng2.Next(), rng2.Next());
            }
            else
            {
                push_plan(0, 3, 1, 1, rng2.Next(), rng2.Next());
                push_plan(1, 1, 1, 0, rng2.Next(), rng2.Next());
            }
        }
        else
        {
            if (num2 < 0.15000000596046448)
            {
                push_plan(0, num3 > 0.5 ? 3 : 2, 1, 0, rng2.Next(), rng2.Next());
                push_plan(0, num3 > 0.5 ? 4 : 3, 2, 0, rng2.Next(), rng2.Next());
                push_plan(0, num3 > 0.5 ? 5 : 4, 3, 0, rng2.Next(), rng2.Next());
            }
            else if (num2 < 0.75)
            {
                push_plan(0, num3 > 0.5 ? 3 : 2, 1, 0, rng2.Next(), rng2.Next());
                push_plan(0, 4, 2, 1, rng2.Next(), rng2.Next());
                push_plan(2, 1, 1, 0, rng2.Next(), rng2.Next());
            }
            else
            {
                push_plan(0, num3 > 0.5 ? 4 : 3, 1, 1, rng2.Next(), rng2.Next());
                push_plan(1, 1, 1, 0, rng2.Next(), rng2.Next());
                push_plan(1, 2, 2, 0, rng2.Next(), rng2.Next());
            }
        }
        return;
    }

    double pGas[10] = {};
    int planet_count = 1;
    if (star_index == 0)
    {
        planet_count = 4;
        pGas[0] = 0.0;
        pGas[1] = 0.0;
        pGas[2] = 0.0;
    }
    else if (star_spectr == spectr_m)
    {
        planet_count = num1 >= 0.1 ? (num1 >= 0.3 ? (num1 >= 0.8 ? 4 : 3) : 2) : 1;
        if (planet_count <= 3)
        {
            pGas[0] = 0.2; pGas[1] = 0.2;
        }
        else
        {
            pGas[0] = 0.0; pGas[1] = 0.2; pGas[2] = 0.3;
        }
    }
    else if (star_spectr == spectr_k)
    {
        planet_count = num1 >= 0.1 ? (num1 >= 0.2 ? (num1 >= 0.7 ? (num1 >= 0.95 ? 5 : 4) : 3) : 2) : 1;
        if (planet_count <= 3)
        {
            pGas[0] = 0.18; pGas[1] = 0.18;
        }
        else
        {
            pGas[0] = 0.0; pGas[1] = 0.18; pGas[2] = 0.28; pGas[3] = 0.28;
        }
    }
    else if (star_spectr == spectr_g)
    {
        planet_count = num1 >= 0.4 ? (num1 >= 0.9 ? 5 : 4) : 3;
        if (planet_count <= 3)
        {
            pGas[0] = 0.18; pGas[1] = 0.18;
        }
        else
        {
            pGas[0] = 0.0; pGas[1] = 0.2; pGas[2] = 0.3; pGas[3] = 0.3;
        }
    }
    else if (star_spectr == spectr_f)
    {
        planet_count = num1 >= 0.35 ? (num1 >= 0.8 ? 5 : 4) : 3;
        if (planet_count <= 3)
        {
            pGas[0] = 0.2; pGas[1] = 0.2;
        }
        else
        {
            pGas[0] = 0.0; pGas[1] = 0.22; pGas[2] = 0.31; pGas[3] = 0.31;
        }
    }
    else if (star_spectr == spectr_a)
    {
        planet_count = num1 >= 0.3 ? (num1 >= 0.75 ? 5 : 4) : 3;
        if (planet_count <= 3)
        {
            pGas[0] = 0.2; pGas[1] = 0.2;
        }
        else
        {
            pGas[0] = 0.1; pGas[1] = 0.28; pGas[2] = 0.3; pGas[3] = 0.35;
        }
    }
    else if (star_spectr == spectr_b)
    {
        planet_count = num1 >= 0.3 ? (num1 >= 0.75 ? 6 : 5) : 4;
        if (planet_count <= 3)
        {
            pGas[0] = 0.2; pGas[1] = 0.2;
        }
        else
        {
            pGas[0] = 0.1; pGas[1] = 0.22; pGas[2] = 0.28; pGas[3] = 0.35; pGas[4] = 0.35;
        }
    }
    else if (star_spectr == spectr_o)
    {
        planet_count = num1 >= 0.5 ? 6 : 5;
        pGas[0] = 0.1; pGas[1] = 0.2; pGas[2] = 0.25; pGas[3] = 0.3; pGas[4] = 0.32; pGas[5] = 0.35;
    }

    int num8 = 0;
    int num9 = 0;
    int orbit_around = 0;
    int num10 = 1;
    for (int index = 0; index < planet_count; ++index)
    {
        int info_seed = rng2.Next();
        int gen_seed = rng2.Next();
        double num11 = rng2.NextDouble();
        double num12 = rng2.NextDouble();
        bool gas_giant = false;
        if (orbit_around == 0)
        {
            ++num8;
            if (index < planet_count - 1 && num11 < pGas[index])
            {
                gas_giant = true;
                if (num10 < 3)
                    num10 = 3;
            }
            while (star_index != 0 || num10 != 3)
            {
                int left = planet_count - index;
                int slots = 9 - num10;
                if (slots > left)
                {
                    float a = static_cast<float>(left) / static_cast<float>(slots);
                    double prob = num10 <= 3
                        ? static_cast<double>(DLerp(a, 1.0f, 0.15f) + 0.01f)
                        : static_cast<double>(DLerp(a, 1.0f, 0.45f) + 0.01f);
                    if (rng2.NextDouble() < prob)
                        break;
                }
                else
                {
                    break;
                }
                ++num10;
            }
            if (!gas_giant && (star_index == 0 && num10 == 3))
                gas_giant = true;
        }
        else
        {
            ++num9;
            gas_giant = false;
        }

        push_plan(
            orbit_around,
            orbit_around == 0 ? num10 : num9,
            orbit_around == 0 ? num8 : num9,
            gas_giant ? 1 : 0,
            info_seed,
            gen_seed);

        ++num10;
        if (gas_giant)
        {
            orbit_around = num8;
            num9 = 0;
        }
        if (num9 >= 1 && num12 < 0.8)
        {
            orbit_around = 0;
            num9 = 0;
        }
    }
}

constexpr int kLocalPlanetTypeGas = 0;
constexpr int kLocalPlanetTypeOcean = 1;
constexpr int kLocalPlanetTypeVocano = 2;
constexpr int kLocalPlanetTypeDesert = 3;
constexpr int kLocalPlanetTypeIce = 4;

constexpr int kLocalSingularityLaySide = 1;
constexpr int kLocalSingularityTidal1 = 2;
constexpr int kLocalSingularityTidal2 = 4;
constexpr int kLocalSingularityTidal4 = 8;
constexpr int kLocalSingularityClockwise = 16;

__device__ __forceinline__ void EvalPlanetCoreF32OneLocal(
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

    DotNet35RandomDeviceLite rng(info_seed);
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
        singularity_flags |= kLocalSingularityLaySide;
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
            singularity_flags |= kLocalSingularityTidal1;
        }
        else if (num15 > 0.930000007152557)
        {
            obliquity *= 0.1f;
            rotation_period = orbital_period * 0.5;
            singularity_flags |= kLocalSingularityTidal2;
        }
        else if (num15 > 0.899999976158142)
        {
            obliquity *= 0.2f;
            rotation_period = orbital_period * 0.25;
            singularity_flags |= kLocalSingularityTidal4;
        }
    }

    if (num15 > 0.85 && num15 <= 0.9)
    {
        rotation_period = -rotation_period;
        singularity_flags |= kLocalSingularityClockwise;
    }

    float habitable_bias = 0.0f;
    float temperature_bias = 0.0f;
    float radius = 0.0f;
    int type_case = kLocalPlanetTypeDesert;
    int habitable_count_delta = 0;

    if (gas_giant != 0)
    {
        type_case = kLocalPlanetTypeGas;
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
        float num24 = fminf(fmaxf((num19 / fmaxf(1.0f, num20)) * 0.5f + 0.175f, 0.08f), 0.8f);

        habitable_bias = num21 * num22;
        temperature_bias = static_cast<float>(1.20000004768372 / (f2 + 0.2f) - 1.0);
        float num25 = powf(fminf(fmaxf(habitable_bias / num24, 0.0f), 1.0f), num24 * 10.0f);

        if ((num13 > num25 && star_index > 0) || (orbit_around > 0 && orbit_index == 1 && star_index == 0))
        {
            type_case = kLocalPlanetTypeOcean;
            habitable_count_delta = 1;
        }
        else if (f2 < 0.833333f)
        {
            float num26 = fmaxf(0.15f, f2 * 2.5f - 0.85f);
            type_case = (num14 >= num26) ? kLocalPlanetTypeVocano : kLocalPlanetTypeDesert;
        }
        else if (f2 < 1.2f)
        {
            type_case = kLocalPlanetTypeDesert;
        }
        else
        {
            float num27 = 0.9f / f2 - 0.1f;
            type_case = (num14 >= num27) ? kLocalPlanetTypeIce : kLocalPlanetTypeDesert;
        }

        radius = 200.0f;
    }

    int precision = (type_case == kLocalPlanetTypeGas) ? 64 : 200;
    int segment = (type_case == kLocalPlanetTypeGas) ? 2 : 5;

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

__global__ void BuildPlanCorePackCompactKernel(
    const int* star_ids,
    const int* star_seeds,
    const int* star_types,
    const int* star_spectrs,
    const int* star_indexes,
    const int* star_counts_in_galaxy,
    const float* star_orbit_scalers,
    const double* star_masses,
    const float* star_habitable_radiuses,
    const float* star_light_balance_radiuses,
    int star_count,
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
    int* out_star_planet_counts,
    int* out_star_planet_offsets,
    int* out_total_planets,
    int* out_planet_ids,
    int* out_planet_indexes,
    int* out_planet_orbit_indexes,
    int* out_planet_orbit_arounds,
    int* out_planet_gen_seeds,
    int* out_planet_gas_giants,
    float* out_planet_core_habitable_bias,
    float* out_planet_core_sun_distance,
    float* out_planet_core_temperature_bias,
    double* out_planet_core_num13,
    double* out_planet_core_num14,
    double* out_planet_core_rand1)
{
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid < 0 || sid >= star_count)
        return;

    const int star_id = star_ids[sid];
    const int star_seed = star_seeds[sid];
    const int star_type = star_types[sid];
    const int star_spectr = star_spectrs[sid];
    const int star_index = star_indexes[sid];
    const int galaxy_star_count = star_counts_in_galaxy[sid];
    const float star_orbit_scaler = star_orbit_scalers[sid];
    const double star_mass = star_masses[sid];
    const float star_habitable_radius = star_habitable_radiuses[sid];
    const float star_light_balance_radius = star_light_balance_radiuses[sid];
    (void)star_type_main_seq;

    int plan_orbit_arounds[kStarPlanMaxPlanets];
    int plan_orbit_indexes[kStarPlanMaxPlanets];
    int plan_numbers[kStarPlanMaxPlanets];
    int plan_gas_giants[kStarPlanMaxPlanets];
    int plan_info_seeds[kStarPlanMaxPlanets];
    int plan_gen_seeds[kStarPlanMaxPlanets];
    for (int i = 0; i < kStarPlanMaxPlanets; ++i)
    {
        plan_orbit_arounds[i] = 0;
        plan_orbit_indexes[i] = 0;
        plan_numbers[i] = 0;
        plan_gas_giants[i] = 0;
        plan_info_seeds[i] = 0;
        plan_gen_seeds[i] = 0;
    }
    int plan_count = 0;
    auto push_plan = [&](int orbit_around, int orbit_index, int number, int gas_giant, int info_seed, int gen_seed) {
        int pidx = plan_count;
        if (pidx < 0 || pidx >= kStarPlanMaxPlanets)
        {
            plan_count = 0;
            return;
        }
        plan_orbit_arounds[pidx] = orbit_around;
        plan_orbit_indexes[pidx] = orbit_index;
        plan_numbers[pidx] = number;
        plan_gas_giants[pidx] = gas_giant;
        plan_info_seeds[pidx] = info_seed;
        plan_gen_seeds[pidx] = gen_seed;
        plan_count = pidx + 1;
    };

    DotNet35RandomDeviceLite rng1(star_seed);
    rng1.Next();
    rng1.Next();
    rng1.Next();
    DotNet35RandomDeviceLite rng2(rng1.Next());

    double num1 = rng2.NextDouble();
    double num2 = rng2.NextDouble();
    double num3 = rng2.NextDouble();
    double num4 = rng2.NextDouble();
    double num5 = rng2.NextDouble();
    (void)num4;
    (void)num5;
    rng2.NextDouble();
    rng2.NextDouble();

    if (star_type == star_type_black_hole || star_type == star_type_neutron_star)
    {
        push_plan(0, 3, 1, 0, rng2.Next(), rng2.Next());
    }
    else if (star_type == star_type_white_dwarf)
    {
        if (num1 < 0.7)
        {
            push_plan(0, 3, 1, 0, rng2.Next(), rng2.Next());
        }
        else
        {
            if (num2 < 0.30000001192092896)
            {
                push_plan(0, 3, 1, 0, rng2.Next(), rng2.Next());
                push_plan(0, 4, 2, 0, rng2.Next(), rng2.Next());
            }
            else
            {
                push_plan(0, 4, 1, 1, rng2.Next(), rng2.Next());
                push_plan(1, 1, 1, 0, rng2.Next(), rng2.Next());
            }
        }
    }
    else if (star_type == star_type_giant)
    {
        if (num1 < 0.30000001192092896)
        {
            push_plan(0, num3 > 0.5 ? 3 : 2, 1, 0, rng2.Next(), rng2.Next());
        }
        else if (num1 < 0.800000011920929)
        {
            if (num2 < 0.25)
            {
                push_plan(0, num3 > 0.5 ? 3 : 2, 1, 0, rng2.Next(), rng2.Next());
                push_plan(0, num3 > 0.5 ? 4 : 3, 2, 0, rng2.Next(), rng2.Next());
            }
            else
            {
                push_plan(0, 3, 1, 1, rng2.Next(), rng2.Next());
                push_plan(1, 1, 1, 0, rng2.Next(), rng2.Next());
            }
        }
        else
        {
            if (num2 < 0.15000000596046448)
            {
                push_plan(0, num3 > 0.5 ? 3 : 2, 1, 0, rng2.Next(), rng2.Next());
                push_plan(0, num3 > 0.5 ? 4 : 3, 2, 0, rng2.Next(), rng2.Next());
                push_plan(0, num3 > 0.5 ? 5 : 4, 3, 0, rng2.Next(), rng2.Next());
            }
            else if (num2 < 0.75)
            {
                push_plan(0, num3 > 0.5 ? 3 : 2, 1, 0, rng2.Next(), rng2.Next());
                push_plan(0, 4, 2, 1, rng2.Next(), rng2.Next());
                push_plan(2, 1, 1, 0, rng2.Next(), rng2.Next());
            }
            else
            {
                push_plan(0, num3 > 0.5 ? 4 : 3, 1, 1, rng2.Next(), rng2.Next());
                push_plan(1, 1, 1, 0, rng2.Next(), rng2.Next());
                push_plan(1, 2, 2, 0, rng2.Next(), rng2.Next());
            }
        }
    }
    else
    {
        double pGas[10] = {};
        int planet_count = 1;
        if (star_index == 0)
        {
            planet_count = 4;
            pGas[0] = 0.0;
            pGas[1] = 0.0;
            pGas[2] = 0.0;
        }
        else if (star_spectr == spectr_m)
        {
            planet_count = num1 >= 0.1 ? (num1 >= 0.3 ? (num1 >= 0.8 ? 4 : 3) : 2) : 1;
            if (planet_count <= 3)
            {
                pGas[0] = 0.2; pGas[1] = 0.2;
            }
            else
            {
                pGas[0] = 0.0; pGas[1] = 0.2; pGas[2] = 0.3;
            }
        }
        else if (star_spectr == spectr_k)
        {
            planet_count = num1 >= 0.1 ? (num1 >= 0.2 ? (num1 >= 0.7 ? (num1 >= 0.95 ? 5 : 4) : 3) : 2) : 1;
            if (planet_count <= 3)
            {
                pGas[0] = 0.18; pGas[1] = 0.18;
            }
            else
            {
                pGas[0] = 0.0; pGas[1] = 0.18; pGas[2] = 0.28; pGas[3] = 0.28;
            }
        }
        else if (star_spectr == spectr_g)
        {
            planet_count = num1 >= 0.4 ? (num1 >= 0.9 ? 5 : 4) : 3;
            if (planet_count <= 3)
            {
                pGas[0] = 0.18; pGas[1] = 0.18;
            }
            else
            {
                pGas[0] = 0.0; pGas[1] = 0.2; pGas[2] = 0.3; pGas[3] = 0.3;
            }
        }
        else if (star_spectr == spectr_f)
        {
            planet_count = num1 >= 0.35 ? (num1 >= 0.8 ? 5 : 4) : 3;
            if (planet_count <= 3)
            {
                pGas[0] = 0.2; pGas[1] = 0.2;
            }
            else
            {
                pGas[0] = 0.0; pGas[1] = 0.22; pGas[2] = 0.31; pGas[3] = 0.31;
            }
        }
        else if (star_spectr == spectr_a)
        {
            planet_count = num1 >= 0.3 ? (num1 >= 0.75 ? 5 : 4) : 3;
            if (planet_count <= 3)
            {
                pGas[0] = 0.2; pGas[1] = 0.2;
            }
            else
            {
                pGas[0] = 0.1; pGas[1] = 0.28; pGas[2] = 0.3; pGas[3] = 0.35;
            }
        }
        else if (star_spectr == spectr_b)
        {
            planet_count = num1 >= 0.3 ? (num1 >= 0.75 ? 6 : 5) : 4;
            if (planet_count <= 3)
            {
                pGas[0] = 0.2; pGas[1] = 0.2;
            }
            else
            {
                pGas[0] = 0.1; pGas[1] = 0.22; pGas[2] = 0.28; pGas[3] = 0.35; pGas[4] = 0.35;
            }
        }
        else if (star_spectr == spectr_o)
        {
            planet_count = num1 >= 0.5 ? 6 : 5;
            pGas[0] = 0.1; pGas[1] = 0.2; pGas[2] = 0.25; pGas[3] = 0.3; pGas[4] = 0.32; pGas[5] = 0.35;
        }

        int num8 = 0;
        int num9 = 0;
        int orbit_around = 0;
        int num10 = 1;
        for (int index = 0; index < planet_count; ++index)
        {
            int info_seed = rng2.Next();
            int gen_seed = rng2.Next();
            double num11 = rng2.NextDouble();
            double num12 = rng2.NextDouble();
            bool gas_giant = false;
            if (orbit_around == 0)
            {
                ++num8;
                if (index < planet_count - 1 && num11 < pGas[index])
                {
                    gas_giant = true;
                    if (num10 < 3)
                        num10 = 3;
                }
                while (star_index != 0 || num10 != 3)
                {
                    int left = planet_count - index;
                    int slots = 9 - num10;
                    if (slots > left)
                    {
                        float a = static_cast<float>(left) / static_cast<float>(slots);
                        double prob = num10 <= 3
                            ? static_cast<double>(DLerp(a, 1.0f, 0.15f) + 0.01f)
                            : static_cast<double>(DLerp(a, 1.0f, 0.45f) + 0.01f);
                        if (rng2.NextDouble() < prob)
                            break;
                    }
                    else
                    {
                        break;
                    }
                    ++num10;
                }
                if (!gas_giant && (star_index == 0 && num10 == 3))
                    gas_giant = true;
            }
            else
            {
                ++num9;
                gas_giant = false;
            }

            push_plan(
                orbit_around,
                orbit_around == 0 ? num10 : num9,
                orbit_around == 0 ? num8 : num9,
                gas_giant ? 1 : 0,
                info_seed,
                gen_seed);

            ++num10;
            if (gas_giant)
            {
                orbit_around = num8;
                num9 = 0;
            }
            if (num9 >= 1 && num12 < 0.8)
            {
                orbit_around = 0;
                num9 = 0;
            }
        }
    }

    int pcount = plan_count;
    if (pcount < 0)
        pcount = 0;
    if (pcount > kStarPlanMaxPlanets)
        pcount = kStarPlanMaxPlanets;
    out_star_planet_counts[sid] = pcount;
    int out_base = atomicAdd(out_total_planets, pcount);
    out_star_planet_offsets[sid] = out_base;
    if (pcount <= 0)
        return;

    int primary_number_to_local[kStarPlanMaxPlanets + 1];
    float parent_real_radius[kStarPlanMaxPlanets];
    float parent_orbit_radius[kStarPlanMaxPlanets];
    double parent_orbital_period[kStarPlanMaxPlanets];
    for (int i = 0; i <= kStarPlanMaxPlanets; ++i)
        primary_number_to_local[i] = -1;
    for (int i = 0; i < kStarPlanMaxPlanets; ++i)
    {
        parent_real_radius[i] = 0.0f;
        parent_orbit_radius[i] = 0.0f;
        parent_orbital_period[i] = 0.0;
    }

    const int boost_inclination_ns = (star_type == star_type_neutron_star || star_type == star_type_black_hole) ? 1 : 0;
    int compact_type_case = 0;
    if (star_type == star_type_white_dwarf)
        compact_type_case = 1;
    else if (star_type == star_type_neutron_star || star_type == star_type_black_hole)
        compact_type_case = 2;

    for (int li = 0; li < pcount; ++li)
    {
        const int orbit_around = plan_orbit_arounds[li];
        const int orbit_index = plan_orbit_indexes[li];
        const int gas_giant = plan_gas_giants[li];
        const int info_seed = plan_info_seeds[li];
        const int number = plan_numbers[li];
        const int gen_seed = plan_gen_seeds[li];

        float around_real_radius = 0.0f;
        float around_orbit_radius = 0.0f;
        double around_orbital_period = 0.0;
        if (orbit_around > 0 && orbit_around <= kStarPlanMaxPlanets)
        {
            int parent_li = primary_number_to_local[orbit_around];
            if (parent_li >= 0 && parent_li < kStarPlanMaxPlanets)
            {
                around_real_radius = parent_real_radius[parent_li];
                around_orbit_radius = parent_orbit_radius[parent_li];
                around_orbital_period = parent_orbital_period[parent_li];
            }
        }

        dsp_planet_core_f32_out_t core_full{};
        EvalPlanetCoreF32OneLocal(
            info_seed,
            orbit_around,
            orbit_index,
            gas_giant,
            star_index,
            galaxy_star_count,
            0,
            boost_inclination_ns,
            compact_type_case,
            star_orbit_scaler,
            star_mass,
            star_habitable_radius,
            star_light_balance_radius,
            around_real_radius,
            around_orbit_radius,
            around_orbital_period,
            &core_full);

        if (orbit_around == 0 && number >= 0 && number <= kStarPlanMaxPlanets)
        {
            primary_number_to_local[number] = li;
            parent_real_radius[li] = core_full.radius * core_full.scale;
            parent_orbit_radius[li] = core_full.orbit_radius;
            parent_orbital_period[li] = core_full.orbital_period;
        }

        const int out_p = out_base + li;
        out_planet_ids[out_p] = star_id * 100 + li + 1;
        out_planet_indexes[out_p] = li;
        out_planet_orbit_indexes[out_p] = orbit_index;
        out_planet_orbit_arounds[out_p] = orbit_around;
        out_planet_gen_seeds[out_p] = gen_seed;
        out_planet_gas_giants[out_p] = gas_giant;
        out_planet_core_habitable_bias[out_p] = core_full.habitable_bias;
        out_planet_core_sun_distance[out_p] = core_full.sun_distance;
        out_planet_core_temperature_bias[out_p] = core_full.temperature_bias;
        out_planet_core_num13[out_p] = core_full.num13;
        out_planet_core_num14[out_p] = core_full.num14;
        out_planet_core_rand1[out_p] = core_full.rand1;
    }
}

