using System;
using System.Collections.Generic;
using UnityEngine;

namespace SeedCli
{
    /// <summary>
    /// 实验用途：行星生成路径的“更多 FP32 化”版本。
    /// - RNG 仍然通过 DotNet35Random.NextDouble() 消费（序列一致）
    /// - 但多数随机值/中间计算提前截断到 float，再用 Mathf/float 运算推进
    /// 目标是观察与原版 PlanetGen 的差异（并不追求与游戏一致）。
    /// </summary>
    internal static class PlanetGenF32
    {
        // 复用原工程的 gasCoef（由 UniverseGen 按资源倍率设置）
        private static float GasCoef => global::DspFindSeed.PlanetGen.gasCoef;

        private static List<int> _tmpTheme;

        public static PlanetData CreatePlanet(
            GalaxyData galaxy,
            StarData star,
            int[] themeIds,
            int index,
            int orbitAround,
            int orbitIndex,
            int number,
            bool gasGiant,
            int info_seed,
            int gen_seed)
        {
            PlanetData planet = new PlanetData();
            DotNet35Random rng = new DotNet35Random(info_seed);
            planet.index = index;
            planet.galaxy = star.galaxy;
            planet.star = star;
            planet.seed = gen_seed;
            planet.infoSeed = info_seed;
            planet.orbitAround = orbitAround;
            planet.orbitIndex = orbitIndex;
            planet.number = number;
            planet.id = star.id * 100 + index + 1;

            // 这一段基本不影响搜索，但保持结构一致
            if (orbitAround > 0)
            {
                for (int i = 0; i < star.planetCount; ++i)
                {
                    if (orbitAround == star.planets[i].number && star.planets[i].orbitAround == 0)
                    {
                        planet.orbitAroundPlanet = star.planets[i];
                        if (orbitIndex > 1)
                        {
                            planet.orbitAroundPlanet.singularity |= EPlanetSingularity.MultipleSatellites;
                            break;
                        }
                        break;
                    }
                }
                Assert.NotNull(planet.orbitAroundPlanet);
            }

            string roman = star.planetCount > 20 ? (index + 1).ToString() : NameGen.roman[index + 1];
            planet.name = star.name + " " + roman + "号星";

            if (CudaPlanetNative.TryEvalPlanetCoreF32(
                info_seed,
                orbitAround,
                orbitIndex,
                gasGiant,
                star,
                planet.orbitAroundPlanet,
                star.galaxy.habitableCount,
                out var core))
            {
                planet.orbitRadius = core.orbit_radius;
                planet.orbitInclination = core.orbit_inclination;
                planet.orbitLongitude = core.orbit_longitude;
                planet.orbitalPeriod = core.orbital_period;
                planet.orbitPhase = core.orbit_phase;
                planet.obliquity = core.obliquity;
                planet.rotationPeriod = core.rotation_period;
                planet.rotationPhase = core.rotation_phase;
                planet.sunDistance = core.sun_distance;
                planet.scale = core.scale;
                planet.habitableBias = core.habitable_bias;
                planet.temperatureBias = core.temperature_bias;
                planet.radius = core.radius;
                planet.luminosity = core.luminosity;
                planet.precision = core.precision;
                planet.segment = core.segment;
                planet.singularity |= CudaPlanetNative.BuildSingularityFromFlags(core.singularity_flags);
                planet.type = (EPlanetType)CudaPlanetNative.MapTypeCaseToPlanetType(core.type_case);

                if (core.habitable_count_delta > 0)
                    ++star.galaxy.habitableCount;

                SetPlanetTheme(planet, themeIds, core.rand1, core.rand2, core.rand3, core.rand4, core.theme_seed);
                star.galaxy.astrosData[planet.id].uRadius = planet.realRadius;
                return planet;
            }

            // 随机输入保持 double，避免在阈值分支前提前截断导致主题/类型漂移。
            double num3  = rng.NextDouble();
            double num4  = rng.NextDouble();
            double num5  = rng.NextDouble();
            double num6  = rng.NextDouble();
            double num7  = rng.NextDouble();
            double num8  = rng.NextDouble();
            double num9  = rng.NextDouble();
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
            int theme_seed = rng.Next();

            float a = Mathf.Pow(1.2f, (float)(num3 * (num4 - 0.5) * 0.5));

            float orbitRadius;
            if (orbitAround == 0)
            {
                float b = global::DspFindSeed.StarGen.orbitRadius[orbitIndex] * star.orbitScaler;
                float num16 = (float)(((double)a - 1.0) / (double)Mathf.Max(1f, b) + 1.0);
                orbitRadius = b * num16;
            }
            else
            {
                // 这里原版含 double 运算，这里尽量走 float
                orbitRadius =
                    (float)((1600.0 * orbitIndex + 200.0) * Mathf.Pow(star.orbitScaler, 0.3f) * Mathf.Lerp(a, 1f, 0.5f)
                            + planet.orbitAroundPlanet.realRadius) / 40000f;
            }

            planet.orbitRadius = orbitRadius;
            planet.orbitInclination = (float)(num5 * 16.0 - 8.0);
            if (orbitAround > 0) planet.orbitInclination *= 2.2f;
            planet.orbitLongitude = (float)(num6 * 360.0);

            if (star.type >= EStarType.NeutronStar)
                planet.orbitInclination += planet.orbitInclination > 0f ? 3f : -3f;

            // orbitalPeriod 在 PlanetData 里是 double（原版如此），这里仍用 double sqrt，但输入尽量来自 float（更接近 FP32）
            double f1 = orbitRadius;
            double f1_3 = f1 * f1 * f1;
            planet.orbitalPeriod = planet.orbitAroundPlanet != null
                ? Math.Sqrt(39.4784176043574 * f1_3 / 1.08308421068537E-08)
                : Math.Sqrt(39.4784176043574 * f1_3 / (global::DspFindSeed.PlanetGen.GRAVITY * star.mass));

            planet.orbitPhase = (float)(num7 * 360.0);

            if (num15 < 0.0399999991059303)
            {
                planet.obliquity = (float)(num8 * (num9 - 0.5) * 39.9);
                planet.obliquity += planet.obliquity < 0f ? -70f : 70f;
                planet.singularity |= EPlanetSingularity.LaySide;
            }
            else if (num15 < 0.100000001490116)
            {
                planet.obliquity = (float)(num8 * (num9 - 0.5) * 80.0);
                planet.obliquity += planet.obliquity < 0f ? -30f : 30f;
            }
            else
            {
                planet.obliquity = (float)(num8 * (num9 - 0.5) * 60.0);
            }

            // rotationPeriod 在 PlanetData 里是 double（原版如此）
            double rotation = (num10 * num11 * 1000.0 + 400.0) * (orbitAround == 0 ? (double)Mathf.Pow(orbitRadius, 0.25f) : 1.0) * (gasGiant ? 0.2 : 1.0);
            if (!gasGiant)
            {
                if (star.type == EStarType.WhiteDwarf) rotation *= 0.5;
                else if (star.type == EStarType.NeutronStar) rotation *= 0.2;
                else if (star.type == EStarType.BlackHole) rotation *= 0.15;
            }
            planet.rotationPeriod = rotation;
            planet.rotationPhase = (float)(num12 * 360.0);
            planet.sunDistance = orbitAround == 0 ? planet.orbitRadius : planet.orbitAroundPlanet.orbitRadius;
            planet.scale = 1f;

            double num17 = orbitAround == 0 ? planet.orbitalPeriod : planet.orbitAroundPlanet.orbitalPeriod;
            planet.rotationPeriod = 1.0 / (1.0 / num17 + 1.0 / planet.rotationPeriod);

            if (orbitAround == 0 && orbitIndex <= 4 && !gasGiant)
            {
                if (num15 > 0.959999978542328)
                {
                    planet.obliquity *= 0.01f;
                    planet.rotationPeriod = planet.orbitalPeriod;
                    planet.singularity |= EPlanetSingularity.TidalLocked;
                }
                else if (num15 > 0.930000007152557)
                {
                    planet.obliquity *= 0.1f;
                    planet.rotationPeriod = planet.orbitalPeriod * 0.5;
                    planet.singularity |= EPlanetSingularity.TidalLocked2;
                }
                else if (num15 > 0.899999976158142)
                {
                    planet.obliquity *= 0.2f;
                    planet.rotationPeriod = planet.orbitalPeriod * 0.25;
                    planet.singularity |= EPlanetSingularity.TidalLocked4;
                }
            }

            if (num15 > 0.85f && num15 <= 0.9f)
            {
                planet.rotationPeriod = -planet.rotationPeriod;
                planet.singularity |= EPlanetSingularity.ClockwiseRotate;
            }

            float habitableRadius = star.habitableRadius;
            if (gasGiant)
            {
                planet.type = EPlanetType.Gas;
                planet.radius = 80f;
                planet.scale = 10f;
                planet.habitableBias = 100f;
            }
            else
            {
                float num18 = Mathf.Ceil(star.galaxy.starCount * 0.29f);
                if (num18 < 11f) num18 = 11f;
                float num19 = num18 - star.galaxy.habitableCount;
                float num20 = (star.galaxy.starCount - star.index);
                float sunDistance = planet.sunDistance;
                float num21 = 1000f;
                float f2 = 1000f;
                if (habitableRadius > 0f && sunDistance > 0f)
                {
                    f2 = sunDistance / habitableRadius;
                    num21 = Mathf.Abs(Mathf.Log(f2));
                }

                float num22 = Mathf.Clamp(Mathf.Sqrt(habitableRadius), 1f, 2f) - 0.04f;
                float num24 = Mathf.Clamp(Mathf.Lerp(num19 / Math.Max(1f, num20), 0.35f, 0.5f), 0.08f, 0.8f);

                planet.habitableBias = num21 * num22;
                planet.temperatureBias = (float)(1.20000004768372 / (f2 + 0.2f) - 1.0);

                float num25 = Mathf.Pow(Mathf.Clamp01(planet.habitableBias / num24), num24 * 10f);
                if (num13 > num25 && star.index > 0 || planet.orbitAround > 0 && planet.orbitIndex == 1 && star.index == 0)
                {
                    planet.type = EPlanetType.Ocean;
                    ++star.galaxy.habitableCount;
                }
                else if (f2 < 0.833333f)
                {
                    float num26 = Mathf.Max(0.15f, f2 * 2.5f - 0.85f);
                    planet.type = num14 >= num26 ? EPlanetType.Vocano : EPlanetType.Desert;
                }
                else if (f2 < 1.2f)
                {
                    planet.type = EPlanetType.Desert;
                }
                else
                {
                    float num27 = 0.9f / f2 - 0.1f;
                    planet.type = num14 >= num27 ? EPlanetType.Ice : EPlanetType.Desert;
                }
                planet.radius = 200f;
            }

            if (planet.type != EPlanetType.Gas && planet.type != EPlanetType.None)
            {
                planet.precision = 200;
                planet.segment = 5;
            }
            else
            {
                planet.precision = 64;
                planet.segment = 2;
            }

            planet.luminosity = Mathf.Pow(planet.star.lightBalanceRadius / (planet.sunDistance + 0.01f), 0.6f);
            if (planet.luminosity > 1f)
            {
                planet.luminosity = Mathf.Log(planet.luminosity) + 1f;
                planet.luminosity = Mathf.Log(planet.luminosity) + 1f;
                planet.luminosity = Mathf.Log(planet.luminosity) + 1f;
            }
            planet.luminosity = Mathf.Round(planet.luminosity * 100f) / 100f;

            SetPlanetTheme(planet, themeIds, rand1, rand2, rand3, rand4, theme_seed);
            star.galaxy.astrosData[planet.id].uRadius = planet.realRadius;
            return planet;
        }

        public static void SetPlanetTheme(
            PlanetData planet,
            int[] themeIds,
            double rand1,
            double rand2,
            double rand3,
            double rand4,
            int theme_seed)
        {
            if (_tmpTheme == null) _tmpTheme = new List<int>();
            else _tmpTheme.Clear();

            if (themeIds == null)
                themeIds = global::DspFindSeed.ThemeProto.themeIds;

            int length1 = themeIds.Length;
            for (int i = 0; i < length1; ++i)
            {
                var themeProto = global::DspFindSeed.LDB.themes.Select(themeIds[i]);
                bool ok = false;
                if (planet.star.index == 0 && planet.type == EPlanetType.Ocean)
                {
                    ok = themeProto.Distribute == EThemeDistribute.Birth;
                }
                else
                {
                    bool tempOk = (themeProto.Temperature * planet.temperatureBias) >= -0.1f;
                    if (Mathf.Abs(themeProto.Temperature) < 0.5f && themeProto.PlanetType == EPlanetType.Desert)
                        tempOk = Mathf.Abs(planet.temperatureBias) < Mathf.Abs(themeProto.Temperature) + 0.1f;

                    if ((int)themeProto.PlanetType == (int)planet.type && tempOk)
                    {
                        if (planet.star.index == 0)
                            ok = themeProto.Distribute == EThemeDistribute.Default;
                        else
                            ok = themeProto.Distribute == EThemeDistribute.Default || themeProto.Distribute == EThemeDistribute.Interstellar;
                    }
                }

                if (ok)
                {
                    for (int j = 0; j < planet.index; ++j)
                    {
                        if (planet.star.planets[j].theme == themeProto.ID)
                        {
                            ok = false;
                            break;
                        }
                    }
                }

                if (ok) _tmpTheme.Add(themeProto.ID);
            }

            if (_tmpTheme.Count == 0)
            {
                for (int i = 0; i < length1; ++i)
                {
                    var themeProto = global::DspFindSeed.LDB.themes.Select(themeIds[i]);
                    bool ok = themeProto.PlanetType == EPlanetType.Desert;
                    if (ok)
                    {
                        for (int j = 0; j < planet.index; ++j)
                        {
                            if (planet.star.planets[j].theme == themeProto.ID)
                            {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if (ok) _tmpTheme.Add(themeProto.ID);
                }
            }

            if (_tmpTheme.Count == 0)
            {
                for (int i = 0; i < length1; ++i)
                {
                    var themeProto = global::DspFindSeed.LDB.themes.Select(themeIds[i]);
                    if (themeProto.PlanetType == EPlanetType.Desert)
                        _tmpTheme.Add(themeProto.ID);
                }
            }

            planet.theme = _tmpTheme[(int)(rand1 * _tmpTheme.Count) % _tmpTheme.Count];

            var themeProto1 = global::DspFindSeed.LDB.themes.Select(planet.theme);
            planet.algoId = 0;
            if (themeProto1 != null && themeProto1.Algos != null && themeProto1.Algos.Length != 0)
            {
                planet.algoId = themeProto1.Algos[(int)(rand2 * themeProto1.Algos.Length) % themeProto1.Algos.Length];
                planet.mod_x = themeProto1.ModX.x + rand3 * (themeProto1.ModX.y - themeProto1.ModX.x);
                planet.mod_y = themeProto1.ModY.x + rand4 * (themeProto1.ModY.y - themeProto1.ModY.x);
            }

            if (themeProto1 == null) return;

            planet.style = theme_seed % 60;
            planet.type = (EPlanetType)themeProto1.PlanetType.GetHashCode();
            planet.ionHeight = themeProto1.IonHeight;
            planet.windStrength = themeProto1.Wind;
            planet.waterHeight = themeProto1.WaterHeight;
            planet.waterItemId = themeProto1.WaterItemId;
            planet.levelized = themeProto1.UseHeightForBuild;

            if (planet.type != EPlanetType.Gas) return;

            int lenItems = themeProto1.GasItems.Length;
            int lenSpeeds = themeProto1.GasSpeeds.Length;
            int[] gasItems = new int[lenItems];
            float[] gasSpeeds = new float[lenSpeeds];
            float[] gasHeatValues = new float[lenItems];
            for (int i = 0; i < lenItems; ++i) gasItems[i] = themeProto1.GasItems[i];

            double totalHeat = 0.0;
            for (int i = 0; i < lenItems; ++i)
            {
                var itemProto = global::DspFindSeed.LDB.items.Select(gasItems[i]);
                gasHeatValues[i] = (float)itemProto.HeatValue;
            }

            bool usedCudaGas = CudaPlanetNative.TryEvalPlanetGasDetailsF32(
                theme_seed,
                GasCoef,
                planet.star.resourceCoef,
                themeProto1.GasSpeeds,
                gasHeatValues,
                gasSpeeds,
                out totalHeat);

            if (!usedCudaGas)
            {
                DotNet35Random r = new DotNet35Random(theme_seed);
                int gasLoop = Math.Min(lenSpeeds, gasHeatValues.Length);
                for (int i = 0; i < gasLoop; ++i)
                {
                    float speed = themeProto1.GasSpeeds[i] * (float)(r.NextDouble() * 0.190909147262573 + 0.909090876579285) * GasCoef;
                    gasSpeeds[i] = speed * Mathf.Pow(planet.star.resourceCoef, 0.3f);
                    totalHeat += gasHeatValues[i] * gasSpeeds[i];
                }
            }

            planet.gasItems = gasItems;
            planet.gasSpeeds = gasSpeeds;
            planet.gasHeatValues = gasHeatValues;
            planet.gasTotalHeat = totalHeat;
        }
    }
}
