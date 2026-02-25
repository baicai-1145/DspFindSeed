struct DotNet35RandomHost
{
    int inext;
    int inextp;
    int seed_array[56];

    explicit DotNet35RandomHost(int seed)
    {
        int subtraction = seed == static_cast<int>(0x80000000u) ? kMBig : (seed < 0 ? -seed : seed);
        int mj = SubWrapI32(kMSeed, subtraction);

        seed_array[55] = mj;
        int mk = 1;
        for (int i = 1; i < 55; ++i)
        {
            int ii = (21 * i) % 55;
            seed_array[ii] = mk;
            mk = SubWrapI32(mj, mk);
            if (mk < 0)
                mk = AddWrapI32(mk, kMBig);
            mj = seed_array[ii];
        }
        for (int k = 1; k < 5; ++k)
        {
            for (int i = 1; i < 56; ++i)
            {
                seed_array[i] = SubWrapI32(seed_array[i], seed_array[1 + (i + 30) % 55]);
                if (seed_array[i] < 0)
                    seed_array[i] = AddWrapI32(seed_array[i], kMBig);
            }
        }
        inext = 0;
        inextp = 31;
    }

    double Sample()
    {
        int loc_inext = inext + 1;
        if (loc_inext >= 56)
            loc_inext = 1;

        int loc_inextp = inextp + 1;
        if (loc_inextp >= 56)
            loc_inextp = 1;

        int ret = SubWrapI32(seed_array[loc_inext], seed_array[loc_inextp]);
        if (ret < 0)
            ret = AddWrapI32(ret, kMBig);

        seed_array[loc_inext] = ret;
        inext = loc_inext;
        inextp = loc_inextp;
        return static_cast<double>(ret) * 4.6566128752457969e-10;
    }

    int Next()
    {
        return static_cast<int>(Sample() * 2147483647.0);
    }

    double NextDouble()
    {
        return Sample();
    }
};

template <typename T>
inline T Clamp(T v, T lo, T hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

inline float Clamp01(float v)
{
    return Clamp(v, 0.0f, 1.0f);
}

inline float Lerp(float a, float b, float t)
{
    return a + (b - a) * t;
}

inline float RandNormal(float average_value, float standard_deviation, double r1, double r2)
{
    const double pi = 3.14159265358979;
    return average_value + standard_deviation * static_cast<float>(std::sqrt(-2.0 * std::log(1.0 - r1)) * std::sin(2.0 * pi * r2));
}

inline int RoundToIntBankers(double v)
{
    return static_cast<int>(std::nearbyint(v));
}

struct EnumMap
{
    int star_type_main_seq;
    int star_type_giant;
    int star_type_white_dwarf;
    int star_type_neutron_star;
    int star_type_black_hole;

    int spectr_m;
    int spectr_k;
    int spectr_g;
    int spectr_f;
    int spectr_a;
    int spectr_b;
    int spectr_o;
    int spectr_x;

    int planet_type_gas;
    int planet_type_ocean;
    int planet_type_vocano;
    int planet_type_desert;
    int planet_type_ice;

    int theme_distribute_default;
    int theme_distribute_birth;
    int theme_distribute_interstellar;
};

inline int SpectrFromIndex(int idx, const EnumMap& m)
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

struct ThemeLite
{
    int id;
    int planet_type;
    float temperature;
    int distribute;
    int water_item_id;
    int vein_begin;
    int vein_end;
    int rare_begin;
    int rare_end;
    int rare_settings_begin;
    int rare_settings_end;
};

struct StarLite
{
    int id;
    int index;
    int seed;
    int type;
    int spectr;
    int planet_count;
    float mass;
    float orbit_scaler;
    float habitable_radius;
    float light_balance_radius;
    double pos_x;
    double pos_y;
    double pos_z;
    int pos_qx;
    int pos_qy;
    int pos_qz;
};

struct CoreLite
{
    float orbit_radius;
    double orbital_period;
    float scale;
    float radius;
    float habitable_bias;
    float sun_distance;
    float temperature_bias;
    double num13;
    double num14;
    double rand1;
    int theme_seed;
};

struct PlanetPlanLite
{
    int id;
    int index;
    int orbit_around;
    int orbit_index;
    int number;
    int info_seed;
    int gen_seed;
    int parent_planet_index;
    bool gas_giant;
    CoreLite core;
    int type;
    int theme;
    int theme_index;
    int water_item_id;
    bool is_gas_final;
};

struct StarCtx
{
    StarLite star;
    std::vector<PlanetPlanLite> planets;
};

struct SeedCtx
{
    int galaxy_seed;
    int star_count;
    int birth_star_id;
    int birth_planet_id;
    std::vector<StarCtx> stars;
};

struct PlanetRef
{
    int seed_idx;
    int star_idx;
    int planet_idx;
};

inline void SetStarAgeLite(
    StarLite& star,
    float age,
    double rn,
    double rt,
    const EnumMap& m,
    float& luminosity)
{
    float num1 = static_cast<float>(rn * 0.1 + 0.95);
    float num2 = static_cast<float>(rt * 0.4 + 0.8);
    float num3 = static_cast<float>(rt * 9.0 + 1.0);

    if (age >= 1.0f)
    {
        if (star.mass >= 18.0f)
        {
            star.type = m.star_type_black_hole;
            star.spectr = m.spectr_x;
            star.mass *= 2.5f * num2;
            luminosity *= (1.0f / 1000.0f) * num1;
            star.habitable_radius = 0.0f;
            star.light_balance_radius *= 0.4f * num1;
        }
        else if (star.mass >= 7.0f)
        {
            star.type = m.star_type_neutron_star;
            star.spectr = m.spectr_x;
            star.mass *= 0.2f * num1;
            luminosity *= 0.1f * num1;
            star.habitable_radius = 0.0f;
            star.light_balance_radius *= 3.0f * num1;
            star.orbit_scaler *= 1.5f * num1;
        }
        else
        {
            star.type = m.star_type_white_dwarf;
            star.spectr = m.spectr_x;
            star.mass *= 0.2f * num1;
            luminosity *= 0.04f * num2;
            star.habitable_radius *= 0.15f * num2;
            star.light_balance_radius *= 0.2f * num1;
        }
    }
    else if (age >= 0.959999978542328f)
    {
        float num4 = static_cast<float>(std::pow(5.0, std::abs(std::log10(static_cast<double>(star.mass)) - 0.7)) * 5.0);
        if (num4 > 10.0f)
            num4 = static_cast<float>((std::log(num4 * 0.1f) + 1.0) * 10.0);
        float num5 = static_cast<float>(1.0 - std::pow(age, 30.0f) * 0.5);
        star.type = m.star_type_giant;
        star.mass = num5 * star.mass;
        luminosity = 1.6f * luminosity;
        star.habitable_radius = 9.0f * star.habitable_radius;
        star.light_balance_radius = 3.0f * star.habitable_radius;
        star.orbit_scaler = 3.3f * star.orbit_scaler;
    }
}

inline StarLite CreateBirthStarLite(
    int galaxy_star_count,
    int seed,
    const EnumMap& m)
{
    (void)galaxy_star_count;
    StarLite s{};
    s.id = 1;
    s.index = 0;
    s.seed = seed;
    s.type = m.star_type_main_seq;
    s.spectr = m.spectr_x;
    s.orbit_scaler = 1.0f;
    s.pos_x = 0.0;
    s.pos_y = 0.0;
    s.pos_z = 0.0;
    s.pos_qx = 0;
    s.pos_qy = 0;
    s.pos_qz = 0;

    DotNet35RandomHost rng1(seed);
    rng1.Next();
    int seed2 = rng1.Next();
    DotNet35RandomHost rng2(seed2);
    double r1 = rng2.NextDouble();
    double r2 = rng2.NextDouble();
    double num1 = rng2.NextDouble();
    double rn = rng2.NextDouble();
    double rt = rng2.NextDouble();
    double num2 = rng2.NextDouble() * 0.2 + 0.9;
    double num3 = std::pow(2.0, rng2.NextDouble() * 0.4 - 0.2);

    float p1 = Clamp(RandNormal(0.0f, 0.08f, r1, r2), -0.2f, 0.2f);
    s.mass = std::pow(2.0f, p1);

    double d = 2.0 + 0.4 * (1.0 - static_cast<double>(s.mass));
    float lifetime = static_cast<float>(10000.0 * std::pow(0.1, std::log10(static_cast<double>(s.mass) * 0.5) / std::log10(d) + 1.0) * num2);
    float age = static_cast<float>(num1 * 0.4 + 0.3);
    float num4 = static_cast<float>(1.0 - std::pow(Clamp01(age), 20.0f) * 0.5) * s.mass;
    float temperature = static_cast<float>(std::pow(static_cast<double>(num4), 0.56 + 0.14 / (std::log10(static_cast<double>(num4) + 4.0) / std::log10(5.0))) * 4450.0 + 1300.0);
    double num5 = std::log10((static_cast<double>(temperature) - 1300.0) / 4500.0) / std::log10(2.6) - 0.5;
    if (num5 < 0.0) num5 *= 4.0;
    if (num5 > 2.0) num5 = 2.0;
    else if (num5 < -4.0) num5 = -4.0;
    s.spectr = SpectrFromIndex(static_cast<int>(std::round(num5 + 4.0)), m);

    float luminosity = std::pow(num4, 0.7f);
    (void)num3;
    float p2 = static_cast<float>(num5) + 2.0f;
    s.habitable_radius = std::pow(1.7f, p2) + 0.2f;
    s.light_balance_radius = std::pow(1.7f, p2);
    s.orbit_scaler = std::pow(1.35f, p2);
    if (s.orbit_scaler < 1.0f)
        s.orbit_scaler = Lerp(s.orbit_scaler, 1.0f, 0.6f);
    SetStarAgeLite(s, age, rn, rt, m, luminosity);
    (void)lifetime;
    return s;
}

inline StarLite CreateStarLite(
    int galaxy_star_count,
    const dsp_vec3d_t& pos,
    int id,
    int seed,
    int need_type,
    int need_spectr,
    const EnumMap& m)
{
    StarLite s{};
    s.id = id;
    s.index = id - 1;
    s.seed = seed;
    s.type = m.star_type_main_seq;
    s.spectr = m.spectr_x;
    s.orbit_scaler = 1.0f;
    s.pos_x = pos.x;
    s.pos_y = pos.y;
    s.pos_z = pos.z;
    s.pos_qx = RoundToIntBankers(pos.x * 2400.0);
    s.pos_qy = RoundToIntBankers(pos.y * 2400.0);
    s.pos_qz = RoundToIntBankers(pos.z * 2400.0);

    float level = galaxy_star_count <= 1 ? 0.0f : static_cast<float>(s.index) / static_cast<float>(galaxy_star_count - 1);
    DotNet35RandomHost rng1(seed);
    rng1.Next();
    int seed2 = rng1.Next();
    DotNet35RandomHost rng2(seed2);
    double r1 = rng2.NextDouble();
    double r2 = rng2.NextDouble();
    double num2 = rng2.NextDouble();
    double rn = rng2.NextDouble();
    double rt = rng2.NextDouble();
    double num3 = (rng2.NextDouble() - 0.5) * 0.2;
    double num4 = rng2.NextDouble() * 0.2 + 0.9;
    double y = rng2.NextDouble() * 0.4 - 0.2;
    double num5 = std::pow(2.0, y);
    float num6 = Lerp(-0.98f, 0.88f, level);
    float average = num6 >= 0.0f ? num6 + 0.65f : num6 - 0.65f;
    float stddev = 0.33f;
    if (need_type == m.star_type_giant)
    {
        average = y > -0.08 ? -1.5f : 1.6f;
        stddev = 0.3f;
    }
    float num7 = RandNormal(average, stddev, r1, r2);
    if (need_spectr == m.spectr_m) num7 = -3.0f;
    else if (need_spectr == m.spectr_o) num7 = 3.0f;

    float p1 = Clamp(num7 <= 0.0f ? num7 : num7 * 2.0f, -2.4f, 4.65f) + static_cast<float>(num3) + 1.0f;
    if (need_type == m.star_type_white_dwarf) s.mass = static_cast<float>(1.0 + r2 * 5.0);
    else if (need_type == m.star_type_neutron_star) s.mass = static_cast<float>(7.0 + r1 * 11.0);
    else if (need_type == m.star_type_black_hole) s.mass = static_cast<float>(18.0 + r1 * r2 * 30.0);
    else s.mass = std::pow(2.0f, p1);

    double d = 5.0;
    if (s.mass < 2.0f)
        d = 2.0 + 0.4 * (1.0 - static_cast<double>(s.mass));
    float lifetime = static_cast<float>(10000.0 * std::pow(0.1, std::log10(static_cast<double>(s.mass) * 0.5) / std::log10(d) + 1.0) * num4);
    float age = 0.0f;
    if (need_type == m.star_type_giant)
    {
        lifetime = static_cast<float>(10000.0 * std::pow(0.1, std::log10(static_cast<double>(s.mass) * 0.58) / std::log10(d) + 1.0) * num4);
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
        if (s.mass >= 0.8f) age = static_cast<float>(num2 * 0.699999988079071 + 0.200000002980232);
        else if (s.mass >= 0.5f) age = static_cast<float>(num2 * 0.400000005960464 + 0.100000001490116);
        else age = static_cast<float>(num2 * 0.119999997317791 + 0.0199999995529652);
    }

    float num8 = lifetime * age;
    if (num8 > 5000.0f)
        num8 = static_cast<float>((std::log(num8 / 5000.0f) + 1.0) * 5000.0);
    if (num8 > 8000.0f)
        num8 = static_cast<float>((std::log(std::log(std::log(num8 / 8000.0f) + 1.0f) + 1.0f) + 1.0) * 8000.0);
    lifetime = num8 / age;

    float num9 = static_cast<float>(1.0 - std::pow(Clamp01(age), 20.0f) * 0.5) * s.mass;
    float temperature = static_cast<float>(std::pow(static_cast<double>(num9), 0.56 + 0.14 / (std::log10(static_cast<double>(num9) + 4.0) / std::log10(5.0))) * 4450.0 + 1300.0);
    double num10 = std::log10((static_cast<double>(temperature) - 1300.0) / 4500.0) / std::log10(2.6) - 0.5;
    if (num10 < 0.0) num10 *= 4.0;
    if (num10 > 2.0) num10 = 2.0;
    else if (num10 < -4.0) num10 = -4.0;
    s.spectr = SpectrFromIndex(static_cast<int>(std::round(num10 + 4.0)), m);

    float luminosity = std::pow(num9, 0.7f);
    (void)num5;
    float p2 = static_cast<float>(num10) + 2.0f;
    s.habitable_radius = std::pow(1.7f, p2) + 0.25f;
    s.light_balance_radius = std::pow(1.7f, p2);
    s.orbit_scaler = std::pow(1.35f, p2);
    if (s.orbit_scaler < 1.0f)
        s.orbit_scaler = Lerp(s.orbit_scaler, 1.0f, 0.6f);
    SetStarAgeLite(s, age, rn, rt, m, luminosity);
    (void)lifetime;
    return s;
}

inline int CompactTypeCaseByStarType(int star_type, const EnumMap& m)
{
    if (star_type == m.star_type_white_dwarf) return 1;
    if (star_type == m.star_type_neutron_star) return 2;
    if (star_type == m.star_type_black_hole) return 3;
    return 0;
}

inline bool BoostInclinationNsByStarType(int star_type, const EnumMap& m)
{
    return star_type == m.star_type_neutron_star || star_type == m.star_type_black_hole;
}

inline int MapTypeCaseToPlanetType(int type_case, const EnumMap& m)
{
    switch (type_case)
    {
    case 0: return m.planet_type_gas;
    case 1: return m.planet_type_ocean;
    case 2: return m.planet_type_vocano;
    case 4: return m.planet_type_ice;
    default: return m.planet_type_desert;
    }
}

inline void BuildStarPlanetPlans(
    StarCtx& star_ctx,
    const EnumMap& m)
{
    const int star_type = star_ctx.star.type;
    const int star_spectr = star_ctx.star.spectr;
    DotNet35RandomHost rng1(star_ctx.star.seed);
    rng1.Next();
    rng1.Next();
    rng1.Next();
    DotNet35RandomHost rng2(rng1.Next());

    double num1 = rng2.NextDouble();
    double num2 = rng2.NextDouble();
    double num3 = rng2.NextDouble();
    double num4 = rng2.NextDouble();
    double num5 = rng2.NextDouble();
    (void)num4;
    (void)num5;
    rng2.NextDouble();
    rng2.NextDouble();

    auto push_plan = [&](int index, int orbit_around, int orbit_index, int number, bool gas_giant, int info_seed, int gen_seed) {
        PlanetPlanLite p{};
        p.index = index;
        p.orbit_around = orbit_around;
        p.orbit_index = orbit_index;
        p.number = number;
        p.gas_giant = gas_giant;
        p.info_seed = info_seed;
        p.gen_seed = gen_seed;
        p.id = star_ctx.star.id * 100 + index + 1;
        p.parent_planet_index = -1;
        if (orbit_around > 0)
        {
            for (size_t i = 0; i < star_ctx.planets.size(); ++i)
            {
                if (star_ctx.planets[i].number == orbit_around && star_ctx.planets[i].orbit_around == 0)
                {
                    p.parent_planet_index = static_cast<int>(i);
                    break;
                }
            }
        }
        star_ctx.planets.push_back(p);
    };

    if (star_type == m.star_type_black_hole || star_type == m.star_type_neutron_star)
    {
        int info_seed = rng2.Next();
        int gen_seed = rng2.Next();
        push_plan(0, 0, 3, 1, false, info_seed, gen_seed);
    }
    else if (star_type == m.star_type_white_dwarf)
    {
        if (num1 < 0.7)
        {
            int info_seed = rng2.Next();
            int gen_seed = rng2.Next();
            push_plan(0, 0, 3, 1, false, info_seed, gen_seed);
        }
        else
        {
            if (num2 < 0.30000001192092896)
            {
                int info_seed1 = rng2.Next();
                int gen_seed1 = rng2.Next();
                push_plan(0, 0, 3, 1, false, info_seed1, gen_seed1);
                int info_seed2 = rng2.Next();
                int gen_seed2 = rng2.Next();
                push_plan(1, 0, 4, 2, false, info_seed2, gen_seed2);
            }
            else
            {
                int info_seed1 = rng2.Next();
                int gen_seed1 = rng2.Next();
                push_plan(0, 0, 4, 1, true, info_seed1, gen_seed1);
                int info_seed2 = rng2.Next();
                int gen_seed2 = rng2.Next();
                push_plan(1, 1, 1, 1, false, info_seed2, gen_seed2);
            }
        }
    }
    else if (star_type == m.star_type_giant)
    {
        if (num1 < 0.30000001192092896)
        {
            int info_seed = rng2.Next();
            int gen_seed = rng2.Next();
            push_plan(0, 0, num3 > 0.5 ? 3 : 2, 1, false, info_seed, gen_seed);
        }
        else if (num1 < 0.800000011920929)
        {
            if (num2 < 0.25)
            {
                int info_seed1 = rng2.Next();
                int gen_seed1 = rng2.Next();
                push_plan(0, 0, num3 > 0.5 ? 3 : 2, 1, false, info_seed1, gen_seed1);
                int info_seed2 = rng2.Next();
                int gen_seed2 = rng2.Next();
                push_plan(1, 0, num3 > 0.5 ? 4 : 3, 2, false, info_seed2, gen_seed2);
            }
            else
            {
                int info_seed1 = rng2.Next();
                int gen_seed1 = rng2.Next();
                push_plan(0, 0, 3, 1, true, info_seed1, gen_seed1);
                int info_seed2 = rng2.Next();
                int gen_seed2 = rng2.Next();
                push_plan(1, 1, 1, 1, false, info_seed2, gen_seed2);
            }
        }
        else
        {
            if (num2 < 0.15000000596046448)
            {
                int info_seed1 = rng2.Next();
                int gen_seed1 = rng2.Next();
                push_plan(0, 0, num3 > 0.5 ? 3 : 2, 1, false, info_seed1, gen_seed1);
                int info_seed2 = rng2.Next();
                int gen_seed2 = rng2.Next();
                push_plan(1, 0, num3 > 0.5 ? 4 : 3, 2, false, info_seed2, gen_seed2);
                int info_seed3 = rng2.Next();
                int gen_seed3 = rng2.Next();
                push_plan(2, 0, num3 > 0.5 ? 5 : 4, 3, false, info_seed3, gen_seed3);
            }
            else if (num2 < 0.75)
            {
                int info_seed1 = rng2.Next();
                int gen_seed1 = rng2.Next();
                push_plan(0, 0, num3 > 0.5 ? 3 : 2, 1, false, info_seed1, gen_seed1);
                int info_seed2 = rng2.Next();
                int gen_seed2 = rng2.Next();
                push_plan(1, 0, 4, 2, true, info_seed2, gen_seed2);
                int info_seed3 = rng2.Next();
                int gen_seed3 = rng2.Next();
                push_plan(2, 2, 1, 1, false, info_seed3, gen_seed3);
            }
            else
            {
                int info_seed1 = rng2.Next();
                int gen_seed1 = rng2.Next();
                push_plan(0, 0, num3 > 0.5 ? 4 : 3, 1, true, info_seed1, gen_seed1);
                int info_seed2 = rng2.Next();
                int gen_seed2 = rng2.Next();
                push_plan(1, 1, 1, 1, false, info_seed2, gen_seed2);
                int info_seed3 = rng2.Next();
                int gen_seed3 = rng2.Next();
                push_plan(2, 1, 2, 2, false, info_seed3, gen_seed3);
            }
        }
    }
    else
    {
        double pGas[10] = {};
        int planet_count = 1;
        if (star_ctx.star.index == 0)
        {
            planet_count = 4;
            pGas[0] = 0.0;
            pGas[1] = 0.0;
            pGas[2] = 0.0;
        }
        else if (star_spectr == m.spectr_m)
        {
            planet_count = num1 >= 0.1 ? (num1 >= 0.3 ? (num1 >= 0.8 ? 4 : 3) : 2) : 1;
            if (planet_count <= 3)
            {
                pGas[0] = 0.2;
                pGas[1] = 0.2;
            }
            else
            {
                pGas[0] = 0.0;
                pGas[1] = 0.2;
                pGas[2] = 0.3;
            }
        }
        else if (star_spectr == m.spectr_k)
        {
            planet_count = num1 >= 0.1 ? (num1 >= 0.2 ? (num1 >= 0.7 ? (num1 >= 0.95 ? 5 : 4) : 3) : 2) : 1;
            if (planet_count <= 3)
            {
                pGas[0] = 0.18;
                pGas[1] = 0.18;
            }
            else
            {
                pGas[0] = 0.0;
                pGas[1] = 0.18;
                pGas[2] = 0.28;
                pGas[3] = 0.28;
            }
        }
        else if (star_spectr == m.spectr_g)
        {
            planet_count = num1 >= 0.4 ? (num1 >= 0.9 ? 5 : 4) : 3;
            if (planet_count <= 3)
            {
                pGas[0] = 0.18;
                pGas[1] = 0.18;
            }
            else
            {
                pGas[0] = 0.0;
                pGas[1] = 0.2;
                pGas[2] = 0.3;
                pGas[3] = 0.3;
            }
        }
        else if (star_spectr == m.spectr_f)
        {
            planet_count = num1 >= 0.35 ? (num1 >= 0.8 ? 5 : 4) : 3;
            if (planet_count <= 3)
            {
                pGas[0] = 0.2;
                pGas[1] = 0.2;
            }
            else
            {
                pGas[0] = 0.0;
                pGas[1] = 0.22;
                pGas[2] = 0.31;
                pGas[3] = 0.31;
            }
        }
        else if (star_spectr == m.spectr_a)
        {
            planet_count = num1 >= 0.3 ? (num1 >= 0.75 ? 5 : 4) : 3;
            if (planet_count <= 3)
            {
                pGas[0] = 0.2;
                pGas[1] = 0.2;
            }
            else
            {
                pGas[0] = 0.1;
                pGas[1] = 0.28;
                pGas[2] = 0.3;
                pGas[3] = 0.35;
            }
        }
        else if (star_spectr == m.spectr_b)
        {
            planet_count = num1 >= 0.3 ? (num1 >= 0.75 ? 6 : 5) : 4;
            if (planet_count <= 3)
            {
                pGas[0] = 0.2;
                pGas[1] = 0.2;
            }
            else
            {
                pGas[0] = 0.1;
                pGas[1] = 0.22;
                pGas[2] = 0.28;
                pGas[3] = 0.35;
                pGas[4] = 0.35;
            }
        }
        else if (star_spectr == m.spectr_o)
        {
            planet_count = num1 >= 0.5 ? 6 : 5;
            pGas[0] = 0.1;
            pGas[1] = 0.2;
            pGas[2] = 0.25;
            pGas[3] = 0.3;
            pGas[4] = 0.32;
            pGas[5] = 0.35;
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
                while (star_ctx.star.index != 0 || num10 != 3)
                {
                    int left = planet_count - index;
                    int slots = 9 - num10;
                    if (slots > left)
                    {
                        float a = static_cast<float>(left) / static_cast<float>(slots);
                        double prob = num10 <= 3
                            ? static_cast<double>(Lerp(a, 1.0f, 0.15f) + 0.01f)
                            : static_cast<double>(Lerp(a, 1.0f, 0.45f) + 0.01f);
                        if (rng2.NextDouble() < prob)
                            break;
                    }
                    else
                    {
                        break;
                    }
                    ++num10;
                }
                if (!gas_giant && (star_ctx.star.index == 0 && num10 == 3))
                    gas_giant = true;
            }
            else
            {
                ++num9;
                gas_giant = false;
            }

            push_plan(
                index,
                orbit_around,
                orbit_around == 0 ? num10 : num9,
                orbit_around == 0 ? num8 : num9,
                gas_giant,
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

    star_ctx.star.planet_count = static_cast<int>(star_ctx.planets.size());
}

inline float CalcPAndBonusCase(int star_type, int spectr, const EnumMap& m, int& bonus_case)
{
    bonus_case = 0;
    float p = 1.0f;
    if (star_type == m.star_type_main_seq)
    {
        if (spectr == m.spectr_m) p = 2.5f;
        else if (spectr == m.spectr_k) p = 1.0f;
        else if (spectr == m.spectr_g) p = 0.7f;
        else if (spectr == m.spectr_f) p = 0.6f;
        else if (spectr == m.spectr_a) p = 1.0f;
        else if (spectr == m.spectr_b) p = 0.4f;
        else if (spectr == m.spectr_o) p = 1.6f;
    }
    else if (star_type == m.star_type_giant)
    {
        p = 2.5f;
    }
    else if (star_type == m.star_type_white_dwarf)
    {
        p = 3.5f;
        bonus_case = 1;
    }
    else if (star_type == m.star_type_neutron_star)
    {
        p = 4.5f;
        bonus_case = 2;
    }
    else if (star_type == m.star_type_black_hole)
    {
        p = 5.0f;
        bonus_case = 2;
    }
    return p;
}

__device__ __forceinline__ int DAddWrapI32(int a, int b)
{
    unsigned int ua = static_cast<unsigned int>(a);
    unsigned int ub = static_cast<unsigned int>(b);
    return static_cast<int>(ua + ub);
}

__device__ __forceinline__ int DSubWrapI32(int a, int b)
{
    unsigned int ua = static_cast<unsigned int>(a);
    unsigned int ub = static_cast<unsigned int>(b);
    return static_cast<int>(ua - ub);
}

