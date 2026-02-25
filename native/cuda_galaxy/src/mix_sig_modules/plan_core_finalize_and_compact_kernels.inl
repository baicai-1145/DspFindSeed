__global__ void FinalizeStarPlanetOffsetsKernel(
    int star_count,
    const int* total_planets,
    int* star_planet_offsets)
{
    if (blockIdx.x == 0 && threadIdx.x == 0 && star_planet_offsets != nullptr && total_planets != nullptr && star_count >= 0)
        star_planet_offsets[star_count] = *total_planets;
}

__global__ void EvalPlanetCoreFromPlansKernel(
    const int* star_types,
    const int* star_indexes,
    const int* star_counts_in_galaxy,
    const float* star_orbit_scalers,
    const double* star_masses,
    const float* star_habitable_radiuses,
    const float* star_light_balance_radiuses,
    const int* star_planet_counts,
    const int* plan_orbit_arounds,
    const int* plan_orbit_indexes,
    const int* plan_numbers,
    const int* plan_gas_giants,
    const int* plan_info_seeds,
    int star_count,
    int star_type_white_dwarf,
    int star_type_neutron_star,
    int star_type_black_hole,
    CoreLite* out_core)
{
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid < 0 || sid >= star_count)
        return;

    int pcount = star_planet_counts[sid];
    if (pcount < 0)
        pcount = 0;
    if (pcount > kStarPlanMaxPlanets)
        pcount = kStarPlanMaxPlanets;
    int base = sid * kStarPlanMaxPlanets;
    int stype = star_types[sid];
    int sidx = star_indexes[sid];
    int sgal = star_counts_in_galaxy[sid];
    float s_orbit = star_orbit_scalers[sid];
    double s_mass = star_masses[sid];
    float s_hab = star_habitable_radiuses[sid];
    float s_light = star_light_balance_radiuses[sid];

    int boost_inclination_ns = (stype == star_type_neutron_star || stype == star_type_black_hole) ? 1 : 0;
    int compact_type_case = 0;
    if (stype == star_type_white_dwarf)
        compact_type_case = 1;
    else if (stype == star_type_neutron_star || stype == star_type_black_hole)
        compact_type_case = 2;

    for (int li = 0; li < pcount; ++li)
    {
        int p = base + li;
        int orbit_around = plan_orbit_arounds[p];
        int orbit_index = plan_orbit_indexes[p];
        int gas_giant = plan_gas_giants[p];
        int info_seed = plan_info_seeds[p];

        float around_real_radius = 0.0f;
        float around_orbit_radius = 0.0f;
        double around_orbital_period = 0.0;

        if (orbit_around > 0)
        {
            int parent = -1;
            for (int q = 0; q < li; ++q)
            {
                int qp = base + q;
                if (plan_orbit_arounds[qp] == 0 && plan_numbers[qp] == orbit_around)
                {
                    parent = qp;
                    break;
                }
            }
            if (parent >= base)
            {
                around_real_radius = out_core[parent].radius * out_core[parent].scale;
                around_orbit_radius = out_core[parent].orbit_radius;
                around_orbital_period = out_core[parent].orbital_period;
            }
        }

        dsp_planet_core_f32_out_t core_full{};
        EvalPlanetCoreF32OneLocal(
            info_seed,
            orbit_around,
            orbit_index,
            gas_giant,
            sidx,
            sgal,
            0,
            boost_inclination_ns,
            compact_type_case,
            s_orbit,
            s_mass,
            s_hab,
            s_light,
            around_real_radius,
            around_orbit_radius,
            around_orbital_period,
            &core_full);
        CoreLite core_lite{};
        core_lite.orbit_radius = core_full.orbit_radius;
        core_lite.orbital_period = core_full.orbital_period;
        core_lite.scale = core_full.scale;
        core_lite.radius = core_full.radius;
        core_lite.habitable_bias = core_full.habitable_bias;
        core_lite.sun_distance = core_full.sun_distance;
        core_lite.temperature_bias = core_full.temperature_bias;
        core_lite.num13 = core_full.num13;
        core_lite.num14 = core_full.num14;
        core_lite.rand1 = core_full.rand1;
        core_lite.theme_seed = core_full.theme_seed;
        out_core[p] = core_lite;
    }
}

__global__ void BuildStarPlanetOffsetsAtomicKernel(
    const int* star_planet_counts,
    int star_count,
    int* star_planet_offsets,
    int* out_total_planets)
{
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid < 0 || sid >= star_count)
        return;
    int pc = star_planet_counts[sid];
    if (pc < 0) pc = 0;
    if (pc > kStarPlanMaxPlanets) pc = kStarPlanMaxPlanets;
    int off = atomicAdd(out_total_planets, pc);
    star_planet_offsets[sid] = off;
}

__global__ void EvalPlanetCoreAndPackCompactFromPlansKernel(
    const int* star_ids,
    const int* star_types,
    const int* star_indexes,
    const int* star_counts_in_galaxy,
    const float* star_orbit_scalers,
    const double* star_masses,
    const float* star_habitable_radiuses,
    const float* star_light_balance_radiuses,
    const int* star_planet_counts,
    const int* star_planet_offsets,
    const int* plan_orbit_arounds,
    const int* plan_orbit_indexes,
    const int* plan_numbers,
    const int* plan_gas_giants,
    const int* plan_info_seeds,
    const int* plan_gen_seeds,
    int star_count,
    int star_type_white_dwarf,
    int star_type_neutron_star,
    int star_type_black_hole,
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

    int pcount = star_planet_counts[sid];
    if (pcount < 0) pcount = 0;
    if (pcount > kStarPlanMaxPlanets) pcount = kStarPlanMaxPlanets;
    if (pcount <= 0)
        return;

    const int base = sid * kStarPlanMaxPlanets;
    const int out_base = star_planet_offsets[sid];
    const int star_id = star_ids[sid];
    const int stype = star_types[sid];
    const int sidx = star_indexes[sid];
    const int sgal = star_counts_in_galaxy[sid];
    const float s_orbit = star_orbit_scalers[sid];
    const double s_mass = star_masses[sid];
    const float s_hab = star_habitable_radiuses[sid];
    const float s_light = star_light_balance_radiuses[sid];

    const int boost_inclination_ns = (stype == star_type_neutron_star || stype == star_type_black_hole) ? 1 : 0;
    int compact_type_case = 0;
    if (stype == star_type_white_dwarf)
        compact_type_case = 1;
    else if (stype == star_type_neutron_star || stype == star_type_black_hole)
        compact_type_case = 2;

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

    for (int li = 0; li < pcount; ++li)
    {
        const int p = base + li;
        const int orbit_around = plan_orbit_arounds[p];
        const int orbit_index = plan_orbit_indexes[p];
        const int gas_giant = plan_gas_giants[p];
        const int info_seed = plan_info_seeds[p];
        const int number = plan_numbers[p];
        const int gen_seed = plan_gen_seeds[p];

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
            sidx,
            sgal,
            0,
            boost_inclination_ns,
            compact_type_case,
            s_orbit,
            s_mass,
            s_hab,
            s_light,
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

