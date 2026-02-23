#include "dsp_cuda_galaxy.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <utility>
#include <vector>

namespace
{
constexpr int kMBig = 2147483647;
constexpr int kMSeed = 161803398;
constexpr unsigned long long kFnvOffset = 14695981039346656037ull;
constexpr unsigned long long kFnvPrime = 1099511628211ull;

inline int AddWrapI32(int a, int b)
{
    unsigned int ua = static_cast<unsigned int>(a);
    unsigned int ub = static_cast<unsigned int>(b);
    return static_cast<int>(ua + ub);
}

inline int SubWrapI32(int a, int b)
{
    unsigned int ua = static_cast<unsigned int>(a);
    unsigned int ub = static_cast<unsigned int>(b);
    return static_cast<int>(ua - ub);
}

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
} // namespace

extern "C" int dsp_cuda_mix_signatures_from_seeds_f32(
    const int* galaxy_seeds,
    int seed_count,
    int star_count,
    int collision_fp64,
    int use_fp32_prob_compare,
    int vein_len,
    int device_id,
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
    int spectr_x,
    int planet_type_gas,
    int planet_type_ocean,
    int planet_type_vocano,
    int planet_type_desert,
    int planet_type_ice,
    int theme_distribute_default,
    int theme_distribute_birth,
    int theme_distribute_interstellar,
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
    int theme_count,
    unsigned long long* out_galaxy_sigs,
    unsigned long long* out_planet_sigs,
    unsigned long long* out_vein_sigs,
    unsigned long long* out_pipeline_sigs)
{
    bool debug_enter = false;
    if (const char* de = std::getenv("DSP_NATIVE_SIG_DEBUG_ENTER"))
        debug_enter = std::atoi(de) != 0;
    if (debug_enter)
    {
        std::fprintf(stderr, "[native-sig-enter] seed_count=%d star_count=%d vein_len=%d theme_count=%d device_id=%d\n",
            seed_count, star_count, vein_len, theme_count, device_id);
        std::fflush(stderr);
    }
    if (galaxy_seeds == nullptr || seed_count <= 0 || star_count <= 0 ||
        out_galaxy_sigs == nullptr || out_planet_sigs == nullptr || out_vein_sigs == nullptr || out_pipeline_sigs == nullptr)
    {
        if (const char* st = std::getenv("DSP_NATIVE_SIG_STAGE_TIMING"))
        {
            if (std::atoi(st) != 0)
                std::fprintf(stderr, "[native-sig-error] invalid-args: seeds=%p seed_count=%d star_count=%d out=%p/%p/%p/%p\n",
                    static_cast<const void*>(galaxy_seeds), seed_count, star_count,
                    static_cast<void*>(out_galaxy_sigs), static_cast<void*>(out_planet_sigs), static_cast<void*>(out_vein_sigs), static_cast<void*>(out_pipeline_sigs));
                std::fflush(stderr);
        }
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    }
    if (theme_count <= 0 || theme_ids == nullptr || theme_planet_types == nullptr || theme_temperatures == nullptr ||
        theme_distributes == nullptr || theme_water_item_ids == nullptr ||
        theme_vein_spot_offsets == nullptr || theme_rare_vein_offsets == nullptr || theme_rare_settings_offsets == nullptr)
    {
        if (const char* st = std::getenv("DSP_NATIVE_SIG_STAGE_TIMING"))
        {
            if (std::atoi(st) != 0)
                std::fprintf(stderr, "[native-sig-error] invalid-theme-args: theme_count=%d ids=%p ptypes=%p temps=%p dist=%p water=%p voff=%p roff=%p rsoff=%p\n",
                    theme_count,
                    static_cast<const void*>(theme_ids),
                    static_cast<const void*>(theme_planet_types),
                    static_cast<const void*>(theme_temperatures),
                    static_cast<const void*>(theme_distributes),
                    static_cast<const void*>(theme_water_item_ids),
                    static_cast<const void*>(theme_vein_spot_offsets),
                    static_cast<const void*>(theme_rare_vein_offsets),
                    static_cast<const void*>(theme_rare_settings_offsets));
                std::fflush(stderr);
        }
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    }
    if (vein_len <= 1 || vein_len > 32)
    {
        if (const char* st = std::getenv("DSP_NATIVE_SIG_STAGE_TIMING"))
        {
            if (std::atoi(st) != 0)
                std::fprintf(stderr, "[native-sig-error] invalid-vein-len=%d\n", vein_len);
                std::fflush(stderr);
        }
        return DSP_CUDA_ERR_INVALID_ARGUMENT;
    }
    const int cuda_device_id = device_id >= 0 ? device_id : 0;

    EnumMap em{
        star_type_main_seq, star_type_giant, star_type_white_dwarf, star_type_neutron_star, star_type_black_hole,
        spectr_m, spectr_k, spectr_g, spectr_f, spectr_a, spectr_b, spectr_o, spectr_x,
        planet_type_gas, planet_type_ocean, planet_type_vocano, planet_type_desert, planet_type_ice,
        theme_distribute_default, theme_distribute_birth, theme_distribute_interstellar};

    std::vector<ThemeLite> themes;
    themes.resize(theme_count);
    int max_planet_type = 0;
    for (int i = 0; i < theme_count; ++i)
        max_planet_type = std::max(max_planet_type, theme_planet_types[i]);
    max_planet_type = std::max(max_planet_type, em.planet_type_gas);
    max_planet_type = std::max(max_planet_type, em.planet_type_ocean);
    max_planet_type = std::max(max_planet_type, em.planet_type_vocano);
    max_planet_type = std::max(max_planet_type, em.planet_type_desert);
    max_planet_type = std::max(max_planet_type, em.planet_type_ice);
    std::vector<std::vector<int>> themes_by_planet_type(static_cast<size_t>(max_planet_type + 1));
    for (int i = 0; i < theme_count; ++i)
    {
        ThemeLite t{};
        t.id = theme_ids[i];
        t.planet_type = theme_planet_types[i];
        t.temperature = theme_temperatures[i];
        t.distribute = theme_distributes[i];
        t.water_item_id = theme_water_item_ids[i];
        t.vein_begin = theme_vein_spot_offsets[i];
        t.vein_end = theme_vein_spot_offsets[i + 1];
        t.rare_begin = theme_rare_vein_offsets[i];
        t.rare_end = theme_rare_vein_offsets[i + 1];
        t.rare_settings_begin = theme_rare_settings_offsets[i];
        t.rare_settings_end = theme_rare_settings_offsets[i + 1];
        themes[i] = t;
        if (t.planet_type >= 0 && t.planet_type <= max_planet_type)
            themes_by_planet_type[t.planet_type].push_back(i);
    }

    const int iter_count = 4;
    const int max_pose_count = star_count * iter_count;
    const int group_cap = ResolveGroupSize(seed_count, star_count, iter_count);
    bool debug_dump = false;
    if (const char* dbg = std::getenv("DSP_NATIVE_SIG_DEBUG"))
        debug_dump = std::atoi(dbg) != 0;
    bool stage_timing = false;
    if (const char* st = std::getenv("DSP_NATIVE_SIG_STAGE_TIMING"))
        stage_timing = std::atoi(st) != 0;
    auto now_ms = []() -> double {
        using clock = std::chrono::steady_clock;
        return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
    };
    double stage_pose_ms = 0.0;
    double stage_seed_build_ms = 0.0;
    double stage_core_pack_ms = 0.0;
    double stage_core_kernel_ms = 0.0;
    double stage_core_unpack_ms = 0.0;
    double stage_theme_ms = 0.0;
    double stage_vein_pack_ms = 0.0;
    double stage_vein_kernel_ms = 0.0;
    double stage_hash_ms = 0.0;
    double stage_total_begin_ms = stage_timing ? now_ms() : 0.0;
    int host_threads = 1;
    if (const char* hs = std::getenv("DSP_NATIVE_SIG_HOST_THREADS"))
    {
        host_threads = std::atoi(hs);
        if (host_threads < 1)
            host_threads = 1;
    }
    else
    {
        unsigned hc = std::thread::hardware_concurrency();
        if (hc > 1)
            host_threads = static_cast<int>(std::min<unsigned>(hc, 16));
    }

    for (int group_base = 0; group_base < seed_count; group_base += group_cap)
    {
        int group_count = std::min(group_cap, seed_count - group_base);
        std::vector<int> pose_seeds(group_count);
        std::vector<int> pose_raw_counts(group_count, 0);
        std::vector<dsp_vec3d_t> pose_raw(static_cast<size_t>(group_count) * static_cast<size_t>(max_pose_count));

        for (int gi = 0; gi < group_count; ++gi)
        {
            DotNet35RandomHost rng(galaxy_seeds[group_base + gi]);
            pose_seeds[gi] = rng.Next();
        }

        double stage0 = stage_timing ? now_ms() : 0.0;
        int rc = dsp_cuda_generate_temp_poses_params_fp64_batch(
            pose_seeds.data(),
            group_count,
            max_pose_count,
            2.0,
            2.3,
            3.5,
            0.18,
            collision_fp64,
            cuda_device_id,
            pose_raw.data(),
            max_pose_count,
            pose_raw_counts.data());
        if (stage_timing)
            stage_pose_ms += now_ms() - stage0;
        if (rc != DSP_CUDA_OK)
        {
            if (stage_timing || debug_dump || debug_enter)
            {
                std::fprintf(stderr, "[native-sig-error] pose-batch rc=%d group_count=%d max_pose_count=%d\n", rc, group_count, max_pose_count);
                std::fflush(stderr);
            }
            return rc;
        }

        stage0 = stage_timing ? now_ms() : 0.0;
        std::vector<SeedCtx> seeds;
        seeds.resize(group_count);
        std::vector<PlanetRef> primary_refs;
        std::vector<PlanetRef> secondary_refs;
        primary_refs.reserve(static_cast<size_t>(group_count) * 200);
        secondary_refs.reserve(static_cast<size_t>(group_count) * 80);
        int worker_count = std::min(host_threads, group_count);
        if (worker_count < 1)
            worker_count = 1;
        std::vector<std::vector<PlanetRef>> primary_refs_tls(static_cast<size_t>(worker_count));
        std::vector<std::vector<PlanetRef>> secondary_refs_tls(static_cast<size_t>(worker_count));
        auto build_seed_range = [&](int begin, int end, int worker_idx) {
            auto& pri = primary_refs_tls[worker_idx];
            auto& sec = secondary_refs_tls[worker_idx];
            pri.reserve(static_cast<size_t>(std::max(0, end - begin)) * 200);
            sec.reserve(static_cast<size_t>(std::max(0, end - begin)) * 80);
            for (int gi = begin; gi < end; ++gi)
            {
                SeedCtx& seed_ctx = seeds[gi];
                seed_ctx.galaxy_seed = galaxy_seeds[group_base + gi];
                seed_ctx.birth_star_id = 0;
                seed_ctx.birth_planet_id = 0;

                const int raw_count = pose_raw_counts[gi];
                const dsp_vec3d_t* raw_base = pose_raw.data() + static_cast<size_t>(gi) * static_cast<size_t>(max_pose_count);
                std::vector<dsp_vec3d_t> poses;
                poses.reserve(star_count);
                for (int i = 0; i < raw_count; ++i)
                {
                    if (i % iter_count == 0)
                        poses.push_back(raw_base[i]);
                }
                if (static_cast<int>(poses.size()) > star_count)
                    poses.resize(star_count);

                seed_ctx.star_count = static_cast<int>(poses.size());
                seed_ctx.stars.resize(seed_ctx.star_count);
                if (seed_ctx.star_count <= 0)
                    continue;

                DotNet35RandomHost galaxy_rng(seed_ctx.galaxy_seed);
                galaxy_rng.Next(); // consumed by pose seed

                float num1 = static_cast<float>(galaxy_rng.NextDouble());
                float num2 = static_cast<float>(galaxy_rng.NextDouble());
                float num3 = static_cast<float>(galaxy_rng.NextDouble());
                float num4 = static_cast<float>(galaxy_rng.NextDouble());

                int num5 = static_cast<int>(std::ceil(0.01 * seed_ctx.star_count + num1 * 0.300000011920929));
                int num6 = static_cast<int>(std::ceil(0.01 * seed_ctx.star_count + num2 * 0.300000011920929));
                int num7 = static_cast<int>(std::ceil(0.0160000007599592 * seed_ctx.star_count + num3 * 0.400000005960464));
                int num8 = static_cast<int>(std::ceil(0.0130000002682209 * seed_ctx.star_count + num4 * 1.39999997615814));
                int num9 = seed_ctx.star_count - num5;
                int num10 = num9 - num6;
                int num11 = num10 - num7;
                int num12 = (num11 - 1) / num8;
                int num13 = num12 / 2;

                for (int si = 0; si < seed_ctx.star_count; ++si)
                {
                    int star_seed = galaxy_rng.Next();
                    StarCtx sc{};
                    if (si == 0)
                    {
                        sc.star = CreateBirthStarLite(seed_ctx.star_count, star_seed, em);
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

                        sc.star = CreateStarLite(seed_ctx.star_count, poses[si], si + 1, star_seed, need_type, need_spectr, em);
                    }
                    BuildStarPlanetPlans(sc, em);
                    for (size_t pi = 0; pi < sc.planets.size(); ++pi)
                    {
                        PlanetRef pr{gi, si, static_cast<int>(pi)};
                        if (sc.planets[pi].orbit_around == 0)
                            pri.push_back(pr);
                        else
                            sec.push_back(pr);
                    }
                    seed_ctx.stars[si] = std::move(sc);
                }
            }
        };

        if (worker_count <= 1)
        {
            build_seed_range(0, group_count, 0);
        }
        else
        {
            std::vector<std::thread> workers;
            workers.reserve(static_cast<size_t>(worker_count - 1));
            int chunk_size = (group_count + worker_count - 1) / worker_count;
            for (int w = 1; w < worker_count; ++w)
            {
                int begin = w * chunk_size;
                int end = std::min(group_count, begin + chunk_size);
                if (begin >= end)
                    break;
                workers.emplace_back([&, begin, end, w]() { build_seed_range(begin, end, w); });
            }
            int begin0 = 0;
            int end0 = std::min(group_count, chunk_size);
            build_seed_range(begin0, end0, 0);
            for (auto& th : workers)
                th.join();
        }
        for (int w = 0; w < worker_count; ++w)
        {
            primary_refs.insert(primary_refs.end(), primary_refs_tls[w].begin(), primary_refs_tls[w].end());
            secondary_refs.insert(secondary_refs.end(), secondary_refs_tls[w].begin(), secondary_refs_tls[w].end());
        }
        if (stage_timing)
            stage_seed_build_ms += now_ms() - stage0;

        auto run_group_parallel = [&](const auto& fn) {
            if (worker_count <= 1 || group_count <= 1)
            {
                fn(0, group_count, 0);
                return;
            }
            int chunk_size = (group_count + worker_count - 1) / worker_count;
            std::vector<std::thread> workers;
            workers.reserve(static_cast<size_t>(worker_count - 1));
            for (int w = 1; w < worker_count; ++w)
            {
                int begin = w * chunk_size;
                int end = std::min(group_count, begin + chunk_size);
                if (begin >= end)
                    break;
                workers.emplace_back([&, begin, end, w]() { fn(begin, end, w); });
            }
            int begin0 = 0;
            int end0 = std::min(group_count, chunk_size);
            fn(begin0, end0, 0);
            for (auto& th : workers)
                th.join();
        };

        auto run_core_phase = [&](const std::vector<PlanetRef>& refs, bool secondary_phase) -> int {
            if (refs.empty())
                return DSP_CUDA_OK;

            const int n = static_cast<int>(refs.size());
            double t_pack = stage_timing ? now_ms() : 0.0;
            std::vector<int> info_seeds(n);
            std::vector<int> orbit_arounds(n);
            std::vector<int> orbit_indexes(n);
            std::vector<int> gas_giants(n);
            std::vector<int> star_indexes(n);
            std::vector<int> galaxy_star_counts(n);
            std::vector<int> galaxy_habitable_counts(n, 0);
            std::vector<int> boost_inclination_ns(n);
            std::vector<int> compact_type_cases(n);
            std::vector<float> star_orbit_scalers(n);
            std::vector<double> star_masses(n);
            std::vector<float> star_habitable_radiuses(n);
            std::vector<float> star_light_balance_radiuses(n);
            std::vector<float> around_real_radiuses(n, 0.0f);
            std::vector<float> around_orbit_radiuses(n, 0.0f);
            std::vector<double> around_orbital_periods(n, 0.0);
            std::vector<dsp_planet_core_f32_out_t> out(n);

            for (int i = 0; i < n; ++i)
            {
                const PlanetRef& ref = refs[i];
                SeedCtx& seed_ctx = seeds[ref.seed_idx];
                StarCtx& star_ctx = seed_ctx.stars[ref.star_idx];
                PlanetPlanLite& pp = star_ctx.planets[ref.planet_idx];

                info_seeds[i] = pp.info_seed;
                orbit_arounds[i] = pp.orbit_around;
                orbit_indexes[i] = pp.orbit_index;
                gas_giants[i] = pp.gas_giant ? 1 : 0;
                star_indexes[i] = star_ctx.star.index;
                galaxy_star_counts[i] = seed_ctx.star_count;
                boost_inclination_ns[i] = BoostInclinationNsByStarType(star_ctx.star.type, em) ? 1 : 0;
                compact_type_cases[i] = CompactTypeCaseByStarType(star_ctx.star.type, em);
                star_orbit_scalers[i] = star_ctx.star.orbit_scaler;
                star_masses[i] = star_ctx.star.mass;
                star_habitable_radiuses[i] = star_ctx.star.habitable_radius;
                star_light_balance_radiuses[i] = star_ctx.star.light_balance_radius;
                if (secondary_phase)
                {
                    if (pp.parent_planet_index < 0 || pp.parent_planet_index >= static_cast<int>(star_ctx.planets.size()))
                    {
                        if (stage_timing || debug_dump)
                        {
                            std::printf(
                                "[native-sig-error] secondary-parent-index invalid seedIdx=%d starIdx=%d planetIdx=%d parent=%d starPlanets=%d\n",
                                ref.seed_idx, ref.star_idx, ref.planet_idx, pp.parent_planet_index, static_cast<int>(star_ctx.planets.size()));
                            std::fflush(stdout);
                        }
                        return DSP_CUDA_ERR_INVALID_ARGUMENT;
                    }
                    const PlanetPlanLite& parent = star_ctx.planets[pp.parent_planet_index];
                    around_real_radiuses[i] = parent.core.radius * parent.core.scale;
                    around_orbit_radiuses[i] = parent.core.orbit_radius;
                    around_orbital_periods[i] = parent.core.orbital_period;
                }
            }
            if (stage_timing)
                stage_core_pack_ms += now_ms() - t_pack;

            double t_kernel = stage_timing ? now_ms() : 0.0;
            int rc = dsp_cuda_planet_eval_core_f32_batch(
                info_seeds.data(),
                orbit_arounds.data(),
                orbit_indexes.data(),
                gas_giants.data(),
                star_indexes.data(),
                galaxy_star_counts.data(),
                galaxy_habitable_counts.data(),
                boost_inclination_ns.data(),
                compact_type_cases.data(),
                star_orbit_scalers.data(),
                star_masses.data(),
                star_habitable_radiuses.data(),
                star_light_balance_radiuses.data(),
                around_real_radiuses.data(),
                around_orbit_radiuses.data(),
                around_orbital_periods.data(),
                n,
                cuda_device_id,
                out.data());
            if (stage_timing)
                stage_core_kernel_ms += now_ms() - t_kernel;
            if (rc != DSP_CUDA_OK)
            {
                if (stage_timing || debug_dump)
                {
                    std::printf("[native-sig-error] core-batch rc=%d secondary=%d n=%d\n", rc, secondary_phase ? 1 : 0, n);
                    std::fflush(stdout);
                }
                return rc;
            }

            double t_unpack = stage_timing ? now_ms() : 0.0;
            for (int i = 0; i < n; ++i)
            {
                const PlanetRef& ref = refs[i];
                PlanetPlanLite& pp = seeds[ref.seed_idx].stars[ref.star_idx].planets[ref.planet_idx];
                pp.core.orbit_radius = out[i].orbit_radius;
                pp.core.orbital_period = out[i].orbital_period;
                pp.core.scale = out[i].scale;
                pp.core.radius = out[i].radius;
                pp.core.habitable_bias = out[i].habitable_bias;
                pp.core.sun_distance = out[i].sun_distance;
                pp.core.temperature_bias = out[i].temperature_bias;
                pp.core.num13 = out[i].num13;
                pp.core.num14 = out[i].num14;
                pp.core.rand1 = out[i].rand1;
                pp.core.theme_seed = out[i].theme_seed;
            }
            if (stage_timing)
                stage_core_unpack_ms += now_ms() - t_unpack;
            return DSP_CUDA_OK;
        };

        rc = run_core_phase(primary_refs, false);
        if (rc != DSP_CUDA_OK)
        {
            if (stage_timing || debug_dump)
            {
                std::printf("[native-sig-error] run_core_phase primary rc=%d refs=%zu\n", rc, primary_refs.size());
                std::fflush(stdout);
            }
            return rc;
        }
        rc = run_core_phase(secondary_refs, true);
        if (rc != DSP_CUDA_OK)
        {
            if (stage_timing || debug_dump)
            {
                std::printf("[native-sig-error] run_core_phase secondary rc=%d refs=%zu\n", rc, secondary_refs.size());
                std::fflush(stdout);
            }
            return rc;
        }

        stage0 = stage_timing ? now_ms() : 0.0;
        std::atomic<int> theme_rc(DSP_CUDA_OK);
        run_group_parallel([&](int begin, int end, int) {
            if (theme_rc.load(std::memory_order_relaxed) != DSP_CUDA_OK)
                return;
            for (int gi = begin; gi < end; ++gi)
            {
                SeedCtx& seed_ctx = seeds[gi];
                seed_ctx.birth_planet_id = 0;
                seed_ctx.birth_star_id = 0;
                int habitable_count = 0;
                for (size_t si = 0; si < seed_ctx.stars.size(); ++si)
                {
                    StarCtx& star_ctx = seed_ctx.stars[si];
                    std::vector<int> assigned;
                    assigned.assign(star_ctx.star.planet_count > 0 ? star_ctx.star.planet_count : 0, 0);
                    for (size_t pi = 0; pi < star_ctx.planets.size(); ++pi)
                    {
                        PlanetPlanLite& pp = star_ctx.planets[pi];
                        int type_case = 0;
                        int habitable_delta = 0;
                        if (pp.gas_giant)
                        {
                            type_case = 0;
                        }
                        else
                        {
                            float num18 = std::ceil(seed_ctx.star_count * 0.29f);
                            if (num18 < 11.0f) num18 = 11.0f;
                            float num19 = num18 - habitable_count;
                            float num20 = seed_ctx.star_count - star_ctx.star.index;
                            float num24 = Clamp((num19 / std::max(1.0f, num20)) * 0.5f + 0.175f, 0.08f, 0.8f);
                            float num25 = std::pow(Clamp01(pp.core.habitable_bias / num24), num24 * 10.0f);
                            float f2 = 1000.0f;
                            if (star_ctx.star.habitable_radius > 0.0f && pp.core.sun_distance > 0.0f)
                                f2 = pp.core.sun_distance / star_ctx.star.habitable_radius;

                            if ((pp.core.num13 > num25 && star_ctx.star.index > 0) ||
                                (pp.orbit_around > 0 && pp.orbit_index == 1 && star_ctx.star.index == 0))
                            {
                                type_case = 1;
                                habitable_delta = 1;
                            }
                            else if (f2 < 0.833333f)
                            {
                                float num26 = std::max(0.15f, f2 * 2.5f - 0.85f);
                                type_case = pp.core.num14 >= num26 ? 2 : 3;
                            }
                            else if (f2 < 1.2f)
                            {
                                type_case = 3;
                            }
                            else
                            {
                                float num27 = 0.9f / f2 - 0.1f;
                                type_case = pp.core.num14 >= num27 ? 4 : 3;
                            }
                        }

                        int provisional_type = MapTypeCaseToPlanetType(type_case, em);
                        if (habitable_delta > 0)
                            ++habitable_count;

                        auto pick_theme_index = [&]() -> int {
                            thread_local std::vector<int> candidates;
                            candidates.clear();
                            if (candidates.capacity() < 64)
                                candidates.reserve(64);
                            auto append_unique = [&](int ti) {
                                int pidx = pp.index;
                                bool ok = true;
                                for (int j = 0; j < pidx; ++j)
                                {
                                    if (j < static_cast<int>(assigned.size()) && assigned[j] == themes[ti].id)
                                    {
                                        ok = false;
                                        break;
                                    }
                                }
                                if (ok)
                                    candidates.push_back(ti);
                            };

                            if (provisional_type >= 0 && provisional_type <= max_planet_type)
                            {
                                const auto& type_themes = themes_by_planet_type[provisional_type];
                                for (int ti : type_themes)
                                {
                                    const ThemeLite& t = themes[ti];
                                    bool ok = false;
                                    if (star_ctx.star.index == 0 && provisional_type == em.planet_type_ocean)
                                    {
                                        ok = t.distribute == em.theme_distribute_birth;
                                    }
                                    else
                                    {
                                        const double temp_gate = -0.10000000149011612;
                                        const double temp_gate_eps = 2e-8;
                                        double temp_prod = static_cast<double>(t.temperature) * static_cast<double>(pp.core.temperature_bias);
                                        bool temp_ok = temp_prod >= (temp_gate + temp_gate_eps);
                                        if (std::fabs(t.temperature) < 0.5f && t.planet_type == em.planet_type_desert)
                                            temp_ok = std::fabs(pp.core.temperature_bias) < std::fabs(t.temperature) + 0.1f;

                                        if (t.planet_type == provisional_type && temp_ok)
                                        {
                                            if (star_ctx.star.index == 0)
                                                ok = t.distribute == em.theme_distribute_default;
                                            else
                                                ok = t.distribute == em.theme_distribute_default || t.distribute == em.theme_distribute_interstellar;
                                        }
                                    }
                                    if (ok)
                                        append_unique(ti);
                                }
                            }

                            if (candidates.empty())
                            {
                                if (em.planet_type_desert >= 0 && em.planet_type_desert <= max_planet_type)
                                {
                                    const auto& desert_themes = themes_by_planet_type[em.planet_type_desert];
                                    for (int ti : desert_themes)
                                        append_unique(ti);
                                }
                            }
                            if (candidates.empty())
                            {
                                if (em.planet_type_desert >= 0 && em.planet_type_desert <= max_planet_type)
                                {
                                    const auto& desert_themes = themes_by_planet_type[em.planet_type_desert];
                                    for (int ti : desert_themes)
                                        candidates.push_back(ti);
                                }
                            }
                            if (candidates.empty())
                                return 0;

                            int pick = static_cast<int>(pp.core.rand1 * static_cast<double>(candidates.size()));
                            if (pick < 0) pick = 0;
                            pick %= static_cast<int>(candidates.size());
                            return candidates[pick];
                        };

                        int theme_idx = pick_theme_index();
                        if (theme_idx < 0 || theme_idx >= theme_count)
                        {
                            theme_rc.store(DSP_CUDA_ERR_INVALID_ARGUMENT, std::memory_order_relaxed);
                            return;
                        }
                        const ThemeLite& t = themes[theme_idx];
                        pp.theme = t.id;
                        pp.theme_index = theme_idx;
                        pp.water_item_id = t.water_item_id;
                        pp.type = t.planet_type;
                        pp.is_gas_final = (pp.type == em.planet_type_gas);
                        if (pp.index >= 0 && pp.index < static_cast<int>(assigned.size()))
                            assigned[pp.index] = pp.theme;

                        if (seed_ctx.birth_planet_id == 0 && star_ctx.star.index == 0 && t.distribute == em.theme_distribute_birth)
                        {
                            seed_ctx.birth_planet_id = pp.id;
                            seed_ctx.birth_star_id = star_ctx.star.id;
                        }
                    }
                }
            }
        });
        if (theme_rc.load(std::memory_order_relaxed) != DSP_CUDA_OK)
            return theme_rc.load(std::memory_order_relaxed);
        if (stage_timing)
            stage_theme_ms += now_ms() - stage0;

        struct SolidRef
        {
            int seed_idx;
            int star_idx;
            int planet_idx;
        };
        std::vector<SolidRef> solids;
        solids.reserve(static_cast<size_t>(group_count) * 240);
        std::vector<std::vector<SolidRef>> solids_tls(static_cast<size_t>(worker_count));
        run_group_parallel([&](int begin, int end, int worker_idx) {
            auto& local = solids_tls[worker_idx];
            for (int gi = begin; gi < end; ++gi)
            {
                const SeedCtx& seed_ctx = seeds[gi];
                for (size_t si = 0; si < seed_ctx.stars.size(); ++si)
                {
                    const StarCtx& star_ctx = seed_ctx.stars[si];
                    for (size_t pi = 0; pi < star_ctx.planets.size(); ++pi)
                    {
                        if (!star_ctx.planets[pi].is_gas_final)
                            local.push_back(SolidRef{gi, static_cast<int>(si), static_cast<int>(pi)});
                    }
                }
            }
        });
        for (int w = 0; w < worker_count; ++w)
            solids.insert(solids.end(), solids_tls[w].begin(), solids_tls[w].end());

        std::vector<int> vein_counts;
        if (!solids.empty())
        {
            stage0 = stage_timing ? now_ms() : 0.0;
            int solid_count = static_cast<int>(solids.size());
            std::vector<int> planet_seeds(solid_count);
            std::vector<float> p_values(solid_count);
            std::vector<int> bonus_cases(solid_count);
            std::vector<int> is_birth_stars(solid_count);
            std::vector<int> theme_indexes(solid_count);
            vein_counts.resize(static_cast<size_t>(solid_count) * static_cast<size_t>(vein_len), 0);

            for (int i = 0; i < solid_count; ++i)
            {
                const SolidRef& sr = solids[i];
                const SeedCtx& seed_ctx = seeds[sr.seed_idx];
                const StarCtx& star_ctx = seed_ctx.stars[sr.star_idx];
                const PlanetPlanLite& pp = star_ctx.planets[sr.planet_idx];

                planet_seeds[i] = pp.gen_seed;
                p_values[i] = CalcPAndBonusCase(star_ctx.star.type, star_ctx.star.spectr, em, bonus_cases[i]);
                is_birth_stars[i] = star_ctx.star.index == 0 ? 1 : 0;

                if (pp.theme_index < 0 || pp.theme_index >= theme_count)
                {
                    if (stage_timing || debug_dump)
                    {
                        std::printf(
                            "[native-sig-error] solid-theme-index invalid seedIdx=%d starIdx=%d planetIdx=%d theme_index=%d theme_count=%d\n",
                            sr.seed_idx, sr.star_idx, sr.planet_idx, pp.theme_index, theme_count);
                        std::fflush(stdout);
                    }
                    return DSP_CUDA_ERR_INVALID_ARGUMENT;
                }
                theme_indexes[i] = pp.theme_index;
            }
            if (stage_timing)
                stage_vein_pack_ms += now_ms() - stage0;

            double t_vein = stage_timing ? now_ms() : 0.0;
            rc = dsp_cuda_mix_chunk_eval_veins_by_theme_f32(
                planet_seeds.data(),
                p_values.data(),
                bonus_cases.data(),
                is_birth_stars.data(),
                theme_indexes.data(),
                solid_count,
                vein_len,
                use_fp32_prob_compare,
                cuda_device_id,
                theme_vein_spot_offsets,
                theme_vein_spot_values,
                theme_rare_vein_offsets,
                theme_rare_vein_values,
                theme_rare_settings_offsets,
                theme_rare_settings_values,
                theme_count,
                vein_counts.data());
            if (stage_timing)
                stage_vein_kernel_ms += now_ms() - t_vein;
            if (rc != DSP_CUDA_OK)
            {
                if (stage_timing || debug_dump)
                {
                    int tv_end = theme_vein_spot_offsets != nullptr ? theme_vein_spot_offsets[theme_count] : -1;
                    int tr_end = theme_rare_vein_offsets != nullptr ? theme_rare_vein_offsets[theme_count] : -1;
                    int ts_end = theme_rare_settings_offsets != nullptr ? theme_rare_settings_offsets[theme_count] : -1;
                    std::printf(
                        "[native-sig-error] vein_by_theme rc=%d solid=%d themeCount=%d veinTotal=%d rareTotal=%d settingsTotal=%d\n",
                        rc, solid_count, theme_count, tv_end, tr_end, ts_end);
                    std::fflush(stdout);
                }
                return rc;
            }
        }

        stage0 = stage_timing ? now_ms() : 0.0;
        int solid_cursor = 0;
        for (int gi = 0; gi < group_count; ++gi)
        {
            const SeedCtx& seed_ctx = seeds[gi];
            unsigned long long h_galaxy = kFnvOffset;
            unsigned long long h_planet = kFnvOffset;
            unsigned long long h_vein = kFnvOffset;
            unsigned long long h_pipeline = kFnvOffset;

            h_galaxy = MixHash(h_galaxy, seed_ctx.star_count);
            h_planet = MixHash(h_planet, seed_ctx.star_count);
            h_planet = MixHash(h_planet, seed_ctx.birth_star_id);
            h_planet = MixHash(h_planet, seed_ctx.birth_planet_id);
            h_vein = MixHash(h_vein, seed_ctx.star_count);
            h_pipeline = MixHash(h_pipeline, seed_ctx.star_count);
            h_pipeline = MixHash(h_pipeline, seed_ctx.birth_star_id);
            h_pipeline = MixHash(h_pipeline, seed_ctx.birth_planet_id);

            for (const StarCtx& star_ctx : seed_ctx.stars)
            {
                h_galaxy = MixHash(h_galaxy, star_ctx.star.id);
                h_galaxy = MixHash(h_galaxy, star_ctx.star.type);
                h_galaxy = MixHash(h_galaxy, star_ctx.star.spectr);
                h_galaxy = MixHash(h_galaxy, star_ctx.star.planet_count);
                h_galaxy = MixHash(h_galaxy, star_ctx.star.pos_qx);
                h_galaxy = MixHash(h_galaxy, star_ctx.star.pos_qy);
                h_galaxy = MixHash(h_galaxy, star_ctx.star.pos_qz);

                h_planet = MixHash(h_planet, star_ctx.star.id);
                h_planet = MixHash(h_planet, star_ctx.star.type);
                h_planet = MixHash(h_planet, star_ctx.star.spectr);
                h_planet = MixHash(h_planet, star_ctx.star.planet_count);
                h_planet = MixHash(h_planet, star_ctx.star.pos_qx);
                h_planet = MixHash(h_planet, star_ctx.star.pos_qy);
                h_planet = MixHash(h_planet, star_ctx.star.pos_qz);

                h_vein = MixHash(h_vein, star_ctx.star.id);

                h_pipeline = MixHash(h_pipeline, star_ctx.star.id);
                h_pipeline = MixHash(h_pipeline, star_ctx.star.type);
                h_pipeline = MixHash(h_pipeline, star_ctx.star.spectr);
                h_pipeline = MixHash(h_pipeline, star_ctx.star.planet_count);
                h_pipeline = MixHash(h_pipeline, star_ctx.star.pos_qx);
                h_pipeline = MixHash(h_pipeline, star_ctx.star.pos_qy);
                h_pipeline = MixHash(h_pipeline, star_ctx.star.pos_qz);

                for (const PlanetPlanLite& pp : star_ctx.planets)
                {
                    h_planet = MixHash(h_planet, pp.type);
                    h_planet = MixHash(h_planet, pp.theme);
                    h_planet = MixHash(h_planet, pp.water_item_id);
                    h_planet = MixHash(h_planet, pp.orbit_index);
                    h_planet = MixHash(h_planet, pp.orbit_around);

                    h_vein = MixHash(h_vein, pp.id);

                    h_pipeline = MixHash(h_pipeline, pp.type);
                    h_pipeline = MixHash(h_pipeline, pp.theme);
                    h_pipeline = MixHash(h_pipeline, pp.water_item_id);
                    h_pipeline = MixHash(h_pipeline, pp.orbit_index);
                    h_pipeline = MixHash(h_pipeline, pp.orbit_around);

                    if (pp.is_gas_final)
                    {
                        h_vein = MixHash(h_vein, 0);
                        continue;
                    }

                    int vmax = vein_len < 32 ? vein_len : 32;
                    for (int vid = 1; vid < vmax; ++vid)
                    {
                        int vv = vein_counts[static_cast<size_t>(solid_cursor) * static_cast<size_t>(vein_len) + static_cast<size_t>(vid)];
                        h_vein = MixHash(h_vein, vv);
                        h_pipeline = MixHash(h_pipeline, vv);
                    }
                    ++solid_cursor;
                }
            }

            out_galaxy_sigs[group_base + gi] = h_galaxy;
            out_planet_sigs[group_base + gi] = h_planet;
            out_vein_sigs[group_base + gi] = h_vein;
            out_pipeline_sigs[group_base + gi] = h_pipeline;

            if (debug_dump && (group_base + gi) == 0)
            {
                std::printf(
                    "[native-sig-debug] seed=%d stars=%d birthStar=%d birthPlanet=%d "
                    "gal=0x%016llX pl=0x%016llX ve=0x%016llX pipe=0x%016llX\n",
                    seed_ctx.galaxy_seed,
                    seed_ctx.star_count,
                    seed_ctx.birth_star_id,
                    seed_ctx.birth_planet_id,
                    static_cast<unsigned long long>(h_galaxy),
                    static_cast<unsigned long long>(h_planet),
                    static_cast<unsigned long long>(h_vein),
                    static_cast<unsigned long long>(h_pipeline));
                if (!seed_ctx.stars.empty())
                {
                    int sshow = std::min(4, static_cast<int>(seed_ctx.stars.size()));
                    for (int si = 0; si < sshow; ++si)
                    {
                        std::printf(
                            "[native-sig-debug] star%d id=%d seed=%d type=%d spectr=%d pc=%d posQ=(%d,%d,%d) coreInputs(mass=%.9g orbitScaler=%.9g hab=%.9g light=%.9g)\n",
                            si,
                            seed_ctx.stars[si].star.id,
                            seed_ctx.stars[si].star.seed,
                            seed_ctx.stars[si].star.type,
                            seed_ctx.stars[si].star.spectr,
                            seed_ctx.stars[si].star.planet_count,
                            seed_ctx.stars[si].star.pos_qx,
                            seed_ctx.stars[si].star.pos_qy,
                            seed_ctx.stars[si].star.pos_qz,
                            static_cast<double>(seed_ctx.stars[si].star.mass),
                            static_cast<double>(seed_ctx.stars[si].star.orbit_scaler),
                            static_cast<double>(seed_ctx.stars[si].star.habitable_radius),
                            static_cast<double>(seed_ctx.stars[si].star.light_balance_radius));
                        int pshow = std::min(2, static_cast<int>(seed_ctx.stars[si].planets.size()));
                        for (int pi = 0; pi < pshow; ++pi)
                        {
                            const PlanetPlanLite& p = seed_ctx.stars[si].planets[pi];
                            std::printf(
                                "[native-sig-debug] star%d.p%d id=%d infoSeed=%d genSeed=%d type=%d theme=%d water=%d orbit=(%d,%d) gas=%d core(orbit=%.9g scale=%.9g radius=%.9g temp=%.9g)\n",
                                si, pi, p.id, p.info_seed, p.gen_seed, p.type, p.theme, p.water_item_id, p.orbit_index, p.orbit_around, p.is_gas_final ? 1 : 0,
                                static_cast<double>(p.core.orbit_radius),
                                static_cast<double>(p.core.scale),
                                static_cast<double>(p.core.radius),
                                static_cast<double>(p.core.temperature_bias));
                        }
                    }
                    std::fflush(stdout);
                }
            }
        }
        if (stage_timing)
            stage_hash_ms += now_ms() - stage0;
    }

    if (stage_timing)
    {
        const double total_ms = now_ms() - stage_total_begin_ms;
        std::printf(
            "[native-sig-timing] seeds=%d stars=%d groupCap=%d totalMs=%.3f "
            "poseMs=%.3f seedBuildMs=%.3f corePackMs=%.3f coreKernelMs=%.3f coreUnpackMs=%.3f "
            "themeMs=%.3f veinPackMs=%.3f veinKernelMs=%.3f hashMs=%.3f\n",
            seed_count, star_count, group_cap, total_ms,
            stage_pose_ms, stage_seed_build_ms, stage_core_pack_ms, stage_core_kernel_ms, stage_core_unpack_ms,
            stage_theme_ms, stage_vein_pack_ms, stage_vein_kernel_ms, stage_hash_ms);
        std::fflush(stdout);
    }

    return DSP_CUDA_OK;
}
