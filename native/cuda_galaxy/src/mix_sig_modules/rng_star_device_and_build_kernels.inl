struct DotNet35RandomDeviceLite
{
    int inext;
    int inextp;
    int seed_array[56];

    __device__ explicit DotNet35RandomDeviceLite(int seed)
    {
        int subtraction = seed == static_cast<int>(0x80000000u) ? kMBig : (seed < 0 ? -seed : seed);
        int mj = DSubWrapI32(kMSeed, subtraction);
        seed_array[55] = mj;
        int mk = 1;
        for (int i = 1; i < 55; ++i)
        {
            int ii = (21 * i) % 55;
            seed_array[ii] = mk;
            mk = DSubWrapI32(mj, mk);
            if (mk < 0)
                mk = DAddWrapI32(mk, kMBig);
            mj = seed_array[ii];
        }
        for (int k = 1; k < 5; ++k)
        {
            for (int i = 1; i < 56; ++i)
            {
                seed_array[i] = DSubWrapI32(seed_array[i], seed_array[1 + (i + 30) % 55]);
                if (seed_array[i] < 0)
                    seed_array[i] = DAddWrapI32(seed_array[i], kMBig);
            }
        }
        inext = 0;
        inextp = 31;
    }

    __device__ int InternalSample()
    {
        int loc_inext = inext + 1;
        if (loc_inext >= 56)
            loc_inext = 1;
        int loc_inextp = inextp + 1;
        if (loc_inextp >= 56)
            loc_inextp = 1;

        int ret = DSubWrapI32(seed_array[loc_inext], seed_array[loc_inextp]);
        if (ret < 0)
            ret = DAddWrapI32(ret, kMBig);
        seed_array[loc_inext] = ret;
        inext = loc_inext;
        inextp = loc_inextp;
        return ret;
    }

    __device__ double Sample()
    {
        int ret = InternalSample();
        return static_cast<double>(ret) * 4.6566128752457969e-10;
    }

    __device__ int Next()
    {
        return static_cast<int>(Sample() * 2147483647.0);
    }

    __device__ double NextDouble()
    {
        return Sample();
    }
};

__device__ __forceinline__ float DLerp(float a, float b, float t)
{
    return a + (b - a) * t;
}

__device__ __forceinline__ float DClamp(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ float DRandNormal(float average_value, float standard_deviation, double r1, double r2)
{
    const double pi = 3.14159265358979;
    return average_value + standard_deviation * static_cast<float>(sqrt(-2.0 * log(1.0 - r1)) * sin(2.0 * pi * r2));
}

__device__ __forceinline__ int DRoundToIntBankers(double v)
{
    return static_cast<int>(nearbyint(v));
}

__device__ __forceinline__ int DSpectrFromIndex(int idx, const EnumMap& m)
{
    switch (idx)
    {
    case 0: return m.spectr_m;
    case 1: return m.spectr_k;
    case 2: return m.spectr_g;
    case 3: return m.spectr_f;
    case 4: return m.spectr_a;
    case 5: return m.spectr_b;
    case 6: return m.spectr_o;
    default: return m.spectr_x;
    }
}

__device__ __forceinline__ void DSetStarAgeLite(
    int& star_type,
    int& star_spectr,
    float& star_mass,
    float& star_orbit_scaler,
    float& star_habitable_radius,
    float& star_light_balance_radius,
    float age,
    double rn,
    double rt,
    const EnumMap& m,
    float& luminosity)
{
    float num1 = static_cast<float>(rn * 0.1 + 0.95);
    float num2 = static_cast<float>(rt * 0.4 + 0.8);
    float num3 = static_cast<float>(rt * 9.0 + 1.0);
    (void)num3;

    if (age >= 1.0f)
    {
        if (star_mass >= 18.0f)
        {
            star_type = m.star_type_black_hole;
            star_spectr = m.spectr_x;
            star_mass *= 2.5f * num2;
            luminosity *= (1.0f / 1000.0f) * num1;
            star_habitable_radius = 0.0f;
            star_light_balance_radius *= 0.4f * num1;
        }
        else if (star_mass >= 7.0f)
        {
            star_type = m.star_type_neutron_star;
            star_spectr = m.spectr_x;
            star_mass *= 0.2f * num1;
            luminosity *= 0.1f * num1;
            star_habitable_radius = 0.0f;
            star_light_balance_radius *= 3.0f * num1;
            star_orbit_scaler *= 1.5f * num1;
        }
        else
        {
            star_type = m.star_type_white_dwarf;
            star_spectr = m.spectr_x;
            star_mass *= 0.2f * num1;
            luminosity *= 0.04f * num2;
            star_habitable_radius *= 0.15f * num2;
            star_light_balance_radius *= 0.2f * num1;
        }
    }
    else if (age >= 0.959999978542328f)
    {
        float num4 = static_cast<float>(pow(5.0, fabs(log10(static_cast<double>(star_mass)) - 0.7)) * 5.0);
        if (num4 > 10.0f)
            num4 = static_cast<float>((log(num4 * 0.1f) + 1.0) * 10.0);
        float num5 = static_cast<float>(1.0 - pow(age, 30.0f) * 0.5);
        star_type = m.star_type_giant;
        star_mass = num5 * star_mass;
        luminosity = 1.6f * luminosity;
        star_habitable_radius = 9.0f * star_habitable_radius;
        star_light_balance_radius = 3.0f * star_habitable_radius;
        star_orbit_scaler = 3.3f * star_orbit_scaler;
    }
}

__device__ __forceinline__ void DCreateBirthStarFields(
    int galaxy_star_count,
    int seed,
    const EnumMap& m,
    int* out_star_type,
    int* out_star_spectr,
    float* out_star_orbit_scaler,
    double* out_star_mass,
    float* out_star_habitable_radius,
    float* out_star_light_balance_radius,
    int* out_pos_qx,
    int* out_pos_qy,
    int* out_pos_qz)
{
    (void)galaxy_star_count;

    int star_type = m.star_type_main_seq;
    int star_spectr = m.spectr_x;
    float star_orbit_scaler = 1.0f;
    int pos_qx = 0;
    int pos_qy = 0;
    int pos_qz = 0;

    DotNet35RandomDeviceLite rng1(seed);
    rng1.Next();
    int seed2 = rng1.Next();
    DotNet35RandomDeviceLite rng2(seed2);
    double r1 = rng2.NextDouble();
    double r2 = rng2.NextDouble();
    double num1 = rng2.NextDouble();
    double rn = rng2.NextDouble();
    double rt = rng2.NextDouble();
    double num2 = rng2.NextDouble() * 0.2 + 0.9;
    double num3 = pow(2.0, rng2.NextDouble() * 0.4 - 0.2);

    float p1 = DClamp(DRandNormal(0.0f, 0.08f, r1, r2), -0.2f, 0.2f);
    float star_mass = powf(2.0f, p1);

    double d = 2.0 + 0.4 * (1.0 - static_cast<double>(star_mass));
    float lifetime = static_cast<float>(10000.0 * pow(0.1, log10(static_cast<double>(star_mass) * 0.5) / log10(d) + 1.0) * num2);
    float age = static_cast<float>(num1 * 0.4 + 0.3);
    float num4 = static_cast<float>(1.0 - pow(DClamp(age, 0.0f, 1.0f), 20.0f) * 0.5) * star_mass;
    float temperature = static_cast<float>(pow(static_cast<double>(num4), 0.56 + 0.14 / (log10(static_cast<double>(num4) + 4.0) / log10(5.0))) * 4450.0 + 1300.0);
    double num5 = log10((static_cast<double>(temperature) - 1300.0) / 4500.0) / log10(2.6) - 0.5;
    if (num5 < 0.0) num5 *= 4.0;
    if (num5 > 2.0) num5 = 2.0;
    else if (num5 < -4.0) num5 = -4.0;
    star_spectr = DSpectrFromIndex(static_cast<int>(round(num5 + 4.0)), m);

    float luminosity = powf(num4, 0.7f);
    (void)num3;
    float p2 = static_cast<float>(num5) + 2.0f;
    float star_habitable_radius = powf(1.7f, p2) + 0.2f;
    float star_light_balance_radius = powf(1.7f, p2);
    star_orbit_scaler = powf(1.35f, p2);
    if (star_orbit_scaler < 1.0f)
        star_orbit_scaler = DLerp(star_orbit_scaler, 1.0f, 0.6f);
    DSetStarAgeLite(star_type, star_spectr, star_mass, star_orbit_scaler, star_habitable_radius, star_light_balance_radius, age, rn, rt, m, luminosity);
    (void)lifetime;

    *out_star_type = star_type;
    *out_star_spectr = star_spectr;
    *out_star_orbit_scaler = star_orbit_scaler;
    *out_star_mass = static_cast<double>(star_mass);
    *out_star_habitable_radius = star_habitable_radius;
    *out_star_light_balance_radius = star_light_balance_radius;
    *out_pos_qx = pos_qx;
    *out_pos_qy = pos_qy;
    *out_pos_qz = pos_qz;
}

__device__ __forceinline__ void DCreateStarFields(
    int galaxy_star_count,
    const dsp_vec3d_t& pos,
    int id,
    int seed,
    int need_type,
    int need_spectr,
    const EnumMap& m,
    int* out_star_type,
    int* out_star_spectr,
    float* out_star_orbit_scaler,
    double* out_star_mass,
    float* out_star_habitable_radius,
    float* out_star_light_balance_radius,
    int* out_pos_qx,
    int* out_pos_qy,
    int* out_pos_qz)
{
    int star_index = id - 1;
    int star_type = m.star_type_main_seq;
    int star_spectr = m.spectr_x;
    float star_orbit_scaler = 1.0f;
    int pos_qx = DRoundToIntBankers(pos.x * 2400.0);
    int pos_qy = DRoundToIntBankers(pos.y * 2400.0);
    int pos_qz = DRoundToIntBankers(pos.z * 2400.0);

    float level = galaxy_star_count <= 1 ? 0.0f : static_cast<float>(star_index) / static_cast<float>(galaxy_star_count - 1);
    DotNet35RandomDeviceLite rng1(seed);
    rng1.Next();
    int seed2 = rng1.Next();
    DotNet35RandomDeviceLite rng2(seed2);
    double r1 = rng2.NextDouble();
    double r2 = rng2.NextDouble();
    double num2 = rng2.NextDouble();
    double rn = rng2.NextDouble();
    double rt = rng2.NextDouble();
    double num3 = (rng2.NextDouble() - 0.5) * 0.2;
    double num4 = rng2.NextDouble() * 0.2 + 0.9;
    double y = rng2.NextDouble() * 0.4 - 0.2;
    double num5 = pow(2.0, y);
    float num6 = DLerp(-0.98f, 0.88f, level);
    float average = num6 >= 0.0f ? num6 + 0.65f : num6 - 0.65f;
    float stddev = 0.33f;
    if (need_type == m.star_type_giant)
    {
        average = y > -0.08 ? -1.5f : 1.6f;
        stddev = 0.3f;
    }
    float num7 = DRandNormal(average, stddev, r1, r2);
    if (need_spectr == m.spectr_m) num7 = -3.0f;
    else if (need_spectr == m.spectr_o) num7 = 3.0f;

    float p1 = DClamp(num7 <= 0.0f ? num7 : num7 * 2.0f, -2.4f, 4.65f) + static_cast<float>(num3) + 1.0f;
    float star_mass = 0.0f;
    if (need_type == m.star_type_white_dwarf) star_mass = static_cast<float>(1.0 + r2 * 5.0);
    else if (need_type == m.star_type_neutron_star) star_mass = static_cast<float>(7.0 + r1 * 11.0);
    else if (need_type == m.star_type_black_hole) star_mass = static_cast<float>(18.0 + r1 * r2 * 30.0);
    else star_mass = powf(2.0f, p1);

    double d = 5.0;
    if (star_mass < 2.0f)
        d = 2.0 + 0.4 * (1.0 - static_cast<double>(star_mass));
    float lifetime = static_cast<float>(10000.0 * pow(0.1, log10(static_cast<double>(star_mass) * 0.5) / log10(d) + 1.0) * num4);
    float age = 0.0f;
    if (need_type == m.star_type_giant)
    {
        lifetime = static_cast<float>(10000.0 * pow(0.1, log10(static_cast<double>(star_mass) * 0.58) / log10(d) + 1.0) * num4);
        age = static_cast<float>(num2 * 0.0399999991059303 + 0.959999978542328);
    }
    else if (need_type == m.star_type_white_dwarf || need_type == m.star_type_neutron_star || need_type == m.star_type_black_hole)
    {
        age = static_cast<float>(num2 * 0.400000005960464 + 1.0);
        if (need_type == m.star_type_white_dwarf) lifetime += 10000.0f;
        else if (need_type == m.star_type_neutron_star) lifetime += 1000.0f;
    }
    else
    {
        if (star_mass >= 0.8f) age = static_cast<float>(num2 * 0.699999988079071 + 0.200000002980232);
        else if (star_mass >= 0.5f) age = static_cast<float>(num2 * 0.400000005960464 + 0.100000001490116);
        else age = static_cast<float>(num2 * 0.119999997317791 + 0.0199999995529652);
    }

    float num8 = lifetime * age;
    if (num8 > 5000.0f)
        num8 = static_cast<float>((log(num8 / 5000.0f) + 1.0) * 5000.0);
    if (num8 > 8000.0f)
        num8 = static_cast<float>((log(log(log(num8 / 8000.0f) + 1.0f) + 1.0f) + 1.0) * 8000.0);
    lifetime = num8 / age;

    float num9 = static_cast<float>(1.0 - pow(DClamp(age, 0.0f, 1.0f), 20.0f) * 0.5) * star_mass;
    float temperature = static_cast<float>(pow(static_cast<double>(num9), 0.56 + 0.14 / (log10(static_cast<double>(num9) + 4.0) / log10(5.0))) * 4450.0 + 1300.0);
    double num10 = log10((static_cast<double>(temperature) - 1300.0) / 4500.0) / log10(2.6) - 0.5;
    if (num10 < 0.0) num10 *= 4.0;
    if (num10 > 2.0) num10 = 2.0;
    else if (num10 < -4.0) num10 = -4.0;
    star_spectr = DSpectrFromIndex(static_cast<int>(round(num10 + 4.0)), m);

    float luminosity = powf(num9, 0.7f);
    (void)num5;
    float p2 = static_cast<float>(num10) + 2.0f;
    float star_habitable_radius = powf(1.7f, p2) + 0.25f;
    float star_light_balance_radius = powf(1.7f, p2);
    star_orbit_scaler = powf(1.35f, p2);
    if (star_orbit_scaler < 1.0f)
        star_orbit_scaler = DLerp(star_orbit_scaler, 1.0f, 0.6f);
    DSetStarAgeLite(star_type, star_spectr, star_mass, star_orbit_scaler, star_habitable_radius, star_light_balance_radius, age, rn, rt, m, luminosity);
    (void)lifetime;

    *out_star_type = star_type;
    *out_star_spectr = star_spectr;
    *out_star_orbit_scaler = star_orbit_scaler;
    *out_star_mass = static_cast<double>(star_mass);
    *out_star_habitable_radius = star_habitable_radius;
    *out_star_light_balance_radius = star_light_balance_radius;
    *out_pos_qx = pos_qx;
    *out_pos_qy = pos_qy;
    *out_pos_qz = pos_qz;
}

__global__ void BuildStarsFromPoseKernel(
    const int* galaxy_seeds,
    const int* seed_star_counts,
    const int* seed_star_offsets,
    const dsp_vec3d_t* pose_raw,
    int pose_stride,
    int seed_count,
    EnumMap em,
    int* out_star_ids,
    int* out_star_seeds,
    int* out_star_types,
    int* out_star_spectrs,
    int* out_star_indexes,
    int* out_star_counts_in_galaxy,
    int* out_star_pos_qx,
    int* out_star_pos_qy,
    int* out_star_pos_qz,
    float* out_star_orbit_scalers,
    double* out_star_masses,
    float* out_star_habitable_radiuses,
    float* out_star_light_balance_radiuses)
{
    int seed_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed_idx < 0 || seed_idx >= seed_count)
        return;

    const int count = seed_star_counts[seed_idx];
    if (count <= 0)
        return;
    const int base = seed_star_offsets[seed_idx];
    const int galaxy_seed = galaxy_seeds[seed_idx];

    DotNet35RandomDeviceLite galaxy_rng(galaxy_seed);
    galaxy_rng.Next(); // consumed by pose seed

    float num1 = static_cast<float>(galaxy_rng.NextDouble());
    float num2 = static_cast<float>(galaxy_rng.NextDouble());
    float num3 = static_cast<float>(galaxy_rng.NextDouble());
    float num4 = static_cast<float>(galaxy_rng.NextDouble());

    int num5 = static_cast<int>(ceil(0.01 * count + num1 * 0.300000011920929));
    int num6 = static_cast<int>(ceil(0.01 * count + num2 * 0.300000011920929));
    int num7 = static_cast<int>(ceil(0.0160000007599592 * count + num3 * 0.400000005960464));
    int num8 = static_cast<int>(ceil(0.0130000002682209 * count + num4 * 1.39999997615814));
    int num9 = count - num5;
    int num10 = num9 - num6;
    int num11 = num10 - num7;
    int num12 = (num11 - 1) / num8;
    int num13 = num12 / 2;

    for (int si = 0; si < count; ++si)
    {
        const int star_flat = base + si;
        const int star_seed = galaxy_rng.Next();
        const int star_id = si + 1;
        int star_type = em.star_type_main_seq;
        int star_spectr = em.spectr_x;
        int pos_qx = 0;
        int pos_qy = 0;
        int pos_qz = 0;
        float star_orbit_scaler = 1.0f;
        double star_mass = 0.0;
        float star_habitable_radius = 0.0f;
        float star_light_balance_radius = 0.0f;

        if (si == 0)
        {
            DCreateBirthStarFields(
                count, star_seed, em,
                &star_type, &star_spectr,
                &star_orbit_scaler,
                &star_mass,
                &star_habitable_radius,
                &star_light_balance_radius,
                &pos_qx, &pos_qy, &pos_qz);
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

            const dsp_vec3d_t pos = pose_raw[static_cast<size_t>(seed_idx) * static_cast<size_t>(pose_stride) + static_cast<size_t>(si)];
            DCreateStarFields(
                count, pos, star_id, star_seed, need_type, need_spectr, em,
                &star_type, &star_spectr,
                &star_orbit_scaler,
                &star_mass,
                &star_habitable_radius,
                &star_light_balance_radius,
                &pos_qx, &pos_qy, &pos_qz);
        }

        out_star_ids[star_flat] = star_id;
        out_star_seeds[star_flat] = star_seed;
        out_star_types[star_flat] = star_type;
        out_star_spectrs[star_flat] = star_spectr;
        out_star_indexes[star_flat] = si;
        out_star_counts_in_galaxy[star_flat] = count;
        out_star_pos_qx[star_flat] = pos_qx;
        out_star_pos_qy[star_flat] = pos_qy;
        out_star_pos_qz[star_flat] = pos_qz;
        out_star_orbit_scalers[star_flat] = star_orbit_scaler;
        out_star_masses[star_flat] = star_mass;
        out_star_habitable_radiuses[star_flat] = star_habitable_radius;
        out_star_light_balance_radiuses[star_flat] = star_light_balance_radius;
    }
}

__global__ void ClampPlanetCountsKernel(
    int* star_planet_counts,
    int star_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 0 || idx >= star_count)
        return;
    int pc = star_planet_counts[idx];
    if (pc < 0)
        pc = 0;
    if (pc > kStarPlanMaxPlanets)
        pc = kStarPlanMaxPlanets;
    star_planet_counts[idx] = pc;
}

__global__ void PackPlanCoreCompactKernel(
    int star_count,
    const int* star_ids,
    const int* star_planet_counts,
    const int* star_planet_offsets,
    const int* plan_orbit_arounds,
    const int* plan_orbit_indexes,
    const int* plan_gen_seeds,
    const int* plan_gas_giants,
    const CoreLite* core_flat,
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

    int pc = star_planet_counts[sid];
    if (pc <= 0)
        return;
    if (pc > kStarPlanMaxPlanets)
        pc = kStarPlanMaxPlanets;
    const int star_id = star_ids[sid];
    const int in_base = sid * kStarPlanMaxPlanets;
    const int out_base = star_planet_offsets[sid];
    for (int li = 0; li < pc; ++li)
    {
        const int in_p = in_base + li;
        const int out_p = out_base + li;
        const CoreLite& core = core_flat[in_p];
        out_planet_ids[out_p] = star_id * 100 + li + 1;
        out_planet_indexes[out_p] = li;
        out_planet_orbit_indexes[out_p] = plan_orbit_indexes[in_p];
        out_planet_orbit_arounds[out_p] = plan_orbit_arounds[in_p];
        out_planet_gen_seeds[out_p] = plan_gen_seeds[in_p];
        out_planet_gas_giants[out_p] = plan_gas_giants[in_p];
        out_planet_core_habitable_bias[out_p] = core.habitable_bias;
        out_planet_core_sun_distance[out_p] = core.sun_distance;
        out_planet_core_temperature_bias[out_p] = core.temperature_bias;
        out_planet_core_num13[out_p] = core.num13;
        out_planet_core_num14[out_p] = core.num14;
        out_planet_core_rand1[out_p] = core.rand1;
    }
}

