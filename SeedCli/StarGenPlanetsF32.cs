using System;
using UnityEngine;

namespace SeedCli
{
    /// <summary>
    /// 实验用途：复制原 StarGen.CreateStarPlanets 的控制流，
    /// 但把 PlanetGen.CreatePlanet 调用替换为 PlanetGenF32.CreatePlanet。
    /// </summary>
    internal static class StarGenPlanetsF32
    {
        // 复用原 StarGen.pGas 表（定义在 DspFindSeed.StarGen）
        public static void CreateStarPlanetsF32(GalaxyData galaxy, StarData star, global::DspFindSeed.GameDesc gameDesc)
        {
            DotNet35Random rng1 = new DotNet35Random(star.seed);
            rng1.Next(); rng1.Next(); rng1.Next();
            DotNet35Random rng2 = new DotNet35Random(rng1.Next());

            // 随机输入仍消费 NextDouble，但截断成 float，模拟“更 FP32”
            float num1 = (float)rng2.NextDouble();
            float num2 = (float)rng2.NextDouble();
            float num3 = (float)rng2.NextDouble();
            float num4 = (float)rng2.NextDouble();
            float num5 = (float)rng2.NextDouble();
            float num6 = (float)rng2.NextDouble() * 0.2f + 0.9f;
            float num7 = (float)rng2.NextDouble() * 0.2f + 0.9f;

            if (star.type == EStarType.BlackHole || star.type == EStarType.NeutronStar)
            {
                star.planetCount = 1;
                star.planets = new PlanetData[1];
                int info_seed = rng2.Next();
                int gen_seed = rng2.Next();
                star.planets[0] = PlanetGenF32.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 0, 0, 3, 1, false, info_seed, gen_seed);
            }
            else if (star.type == EStarType.WhiteDwarf)
            {
                if (num1 < 0.7f)
                {
                    star.planetCount = 1;
                    star.planets = new PlanetData[1];
                    int info_seed = rng2.Next();
                    int gen_seed = rng2.Next();
                    star.planets[0] = PlanetGenF32.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 0, 0, 3, 1, false, info_seed, gen_seed);
                }
                else
                {
                    star.planetCount = 2;
                    star.planets = new PlanetData[2];
                    if (num2 < 0.30000002f)
                    {
                        int info_seed1 = rng2.Next();
                        int gen_seed1 = rng2.Next();
                        star.planets[0] = PlanetGenF32.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 0, 0, 3, 1, false, info_seed1, gen_seed1);
                        int info_seed2 = rng2.Next();
                        int gen_seed2 = rng2.Next();
                        star.planets[1] = PlanetGenF32.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 1, 0, 4, 2, false, info_seed2, gen_seed2);
                    }
                    else
                    {
                        int info_seed3 = rng2.Next();
                        int gen_seed3 = rng2.Next();
                        star.planets[0] = PlanetGenF32.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 0, 0, 4, 1, true, info_seed3, gen_seed3);
                        int info_seed4 = rng2.Next();
                        int gen_seed4 = rng2.Next();
                        star.planets[1] = PlanetGenF32.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 1, 1, 1, 1, false, info_seed4, gen_seed4);
                    }
                }
            }
            else if (star.type == EStarType.GiantStar)
            {
                if (num1 < 0.30000002f)
                {
                    star.planetCount = 1;
                    star.planets = new PlanetData[1];
                    int info_seed = rng2.Next();
                    int gen_seed = rng2.Next();
                    star.planets[0] = PlanetGenF32.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 0, 0, num3 > 0.5f ? 3 : 2, 1, false, info_seed, gen_seed);
                }
                else if (num1 < 0.80000002f)
                {
                    star.planetCount = 2;
                    star.planets = new PlanetData[2];
                    if (num2 < 0.25f)
                    {
                        int info_seed5 = rng2.Next();
                        int gen_seed5 = rng2.Next();
                        star.planets[0] = PlanetGenF32.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 0, 0, num3 > 0.5f ? 3 : 2, 1, false, info_seed5, gen_seed5);
                        int info_seed6 = rng2.Next();
                        int gen_seed6 = rng2.Next();
                        star.planets[1] = PlanetGenF32.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 1, 0, num3 > 0.5f ? 4 : 3, 2, false, info_seed6, gen_seed6);
                    }
                    else
                    {
                        int info_seed7 = rng2.Next();
                        int gen_seed7 = rng2.Next();
                        star.planets[0] = PlanetGenF32.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 0, 0, 3, 1, true, info_seed7, gen_seed7);
                        int info_seed8 = rng2.Next();
                        int gen_seed8 = rng2.Next();
                        star.planets[1] = PlanetGenF32.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 1, 1, 1, 1, false, info_seed8, gen_seed8);
                    }
                }
                else
                {
                    star.planetCount = 3;
                    star.planets = new PlanetData[3];
                    if (num2 < 0.15000001f)
                    {
                        int info_seed9 = rng2.Next();
                        int gen_seed9 = rng2.Next();
                        star.planets[0] = PlanetGenF32.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 0, 0, num3 > 0.5f ? 3 : 2, 1, false, info_seed9, gen_seed9);
                        int info_seed10 = rng2.Next();
                        int gen_seed10 = rng2.Next();
                        star.planets[1] = PlanetGenF32.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 1, 0, num3 > 0.5f ? 4 : 3, 2, false, info_seed10, gen_seed10);
                        int info_seed11 = rng2.Next();
                        int gen_seed11 = rng2.Next();
                        star.planets[2] = PlanetGenF32.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 2, 0, num3 > 0.5f ? 5 : 4, 3, false, info_seed11, gen_seed11);
                    }
                    else if (num2 < 0.75f)
                    {
                        int info_seed12 = rng2.Next();
                        int gen_seed12 = rng2.Next();
                        star.planets[0] = PlanetGenF32.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 0, 0, num3 > 0.5f ? 3 : 2, 1, false, info_seed12, gen_seed12);
                        int info_seed13 = rng2.Next();
                        int gen_seed13 = rng2.Next();
                        star.planets[1] = PlanetGenF32.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 1, 0, 4, 2, true, info_seed13, gen_seed13);
                        int info_seed14 = rng2.Next();
                        int gen_seed14 = rng2.Next();
                        star.planets[2] = PlanetGenF32.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 2, 2, 1, 1, false, info_seed14, gen_seed14);
                    }
                    else
                    {
                        int info_seed15 = rng2.Next();
                        int gen_seed15 = rng2.Next();
                        star.planets[0] = PlanetGenF32.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 0, 0, num3 > 0.5f ? 4 : 3, 1, true, info_seed15, gen_seed15);
                        int info_seed16 = rng2.Next();
                        int gen_seed16 = rng2.Next();
                        star.planets[1] = PlanetGenF32.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 1, 1, 1, 1, false, info_seed16, gen_seed16);
                        int info_seed17 = rng2.Next();
                        int gen_seed17 = rng2.Next();
                        star.planets[2] = PlanetGenF32.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 2, 1, 2, 2, false, info_seed17, gen_seed17);
                    }
                }
            }
            else
            {
                // 原版使用 StarGen 内部 private 的 pGas。这里用本地数组复制同样的概率表。
                double[] pGas = new double[10];
                Array.Clear(pGas, 0, pGas.Length);
                if (star.index == 0)
                {
                    star.planetCount = 4;
                    pGas[0] = 0.0;
                    pGas[1] = 0.0;
                    pGas[2] = 0.0;
                }
                else if (star.spectr == ESpectrType.M)
                {
                    star.planetCount = num1 >= 0.1f ? (num1 >= 0.3f ? (num1 >= 0.8f ? 4 : 3) : 2) : 1;
                    if (star.planetCount <= 3)
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
                else if (star.spectr == ESpectrType.K)
                {
                    star.planetCount = num1 >= 0.1f ? (num1 >= 0.2f ? (num1 >= 0.7f ? (num1 >= 0.95f ? 5 : 4) : 3) : 2) : 1;
                    if (star.planetCount <= 3)
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
                else if (star.spectr == ESpectrType.G)
                {
                    star.planetCount = num1 >= 0.4f ? (num1 >= 0.9f ? 5 : 4) : 3;
                    if (star.planetCount <= 3)
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
                else if (star.spectr == ESpectrType.F)
                {
                    star.planetCount = num1 >= 0.35f ? (num1 >= 0.8f ? 5 : 4) : 3;
                    if (star.planetCount <= 3)
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
                else if (star.spectr == ESpectrType.A)
                {
                    star.planetCount = num1 >= 0.3f ? (num1 >= 0.75f ? 5 : 4) : 3;
                    if (star.planetCount <= 3)
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
                else if (star.spectr == ESpectrType.B)
                {
                    star.planetCount = num1 >= 0.3f ? (num1 >= 0.75f ? 6 : 5) : 4;
                    if (star.planetCount <= 3)
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
                else if (star.spectr == ESpectrType.O)
                {
                    star.planetCount = num1 >= 0.5f ? 6 : 5;
                    pGas[0] = 0.1;
                    pGas[1] = 0.2;
                    pGas[2] = 0.25;
                    pGas[3] = 0.3;
                    pGas[4] = 0.32;
                    pGas[5] = 0.35;
                }
                else
                {
                    star.planetCount = 1;
                }

                star.planets = new PlanetData[star.planetCount];
                int num8 = 0;
                int num9 = 0;
                int orbitAround = 0;
                int num10 = 1;
                for (int index = 0; index < star.planetCount; ++index)
                {
                    int info_seed = rng2.Next();
                    int gen_seed = rng2.Next();
                    float num11 = (float)rng2.NextDouble();
                    float num12 = (float)rng2.NextDouble();
                    bool gasGiant = false;
                    if (orbitAround == 0)
                    {
                        ++num8;
                        if (index < star.planetCount - 1 && num11 < (float)pGas[index])
                        {
                            gasGiant = true;
                            if (num10 < 3) num10 = 3;
                        }
                        for (; star.index != 0 || num10 != 3; ++num10)
                        {
                            int left = star.planetCount - index;
                            int slots = 9 - num10;
                            if (slots > left)
                            {
                                float a = (float)left / (float)slots;
                                float prob = num10 <= 3 ? Mathf.Lerp(a, 1f, 0.15f) + 0.01f : Mathf.Lerp(a, 1f, 0.45f) + 0.01f;
                                if ((float)rng2.NextDouble() < prob) goto label_62;
                            }
                            else goto label_62;
                        }
                        gasGiant = true;
                    }
                    else
                    {
                        ++num9;
                        gasGiant = false;
                    }
                label_62:
                    star.planets[index] = PlanetGenF32.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, index, orbitAround, orbitAround == 0 ? num10 : num9, orbitAround == 0 ? num8 : num9, gasGiant, info_seed, gen_seed);
                    ++num10;
                    if (gasGiant)
                    {
                        orbitAround = num8;
                        num9 = 0;
                    }
                    if (num9 >= 1 && num12 < 0.8f)
                    {
                        orbitAround = 0;
                        num9 = 0;
                    }
                }
            }

            // 小行星带（保持原版 float 逻辑）
            int gasOrbit = 0;
            int lastOrbit = 0;
            int belt1 = 0;
            for (int i = 0; i < star.planetCount; ++i)
            {
                if (star.planets[i].type == EPlanetType.Gas)
                {
                    gasOrbit = star.planets[i].orbitIndex;
                    break;
                }
            }
            for (int i = 0; i < star.planetCount; ++i)
            {
                if (star.planets[i].orbitAround == 0)
                    lastOrbit = star.planets[i].orbitIndex;
            }
            if (gasOrbit > 0)
            {
                int idx = gasOrbit - 1;
                bool ok = true;
                for (int i = 0; i < star.planetCount; ++i)
                {
                    if (star.planets[i].orbitAround == 0 && star.planets[i].orbitIndex == gasOrbit - 1)
                    {
                        ok = false;
                        break;
                    }
                }
                if (ok && num4 < 0.2f + idx * 0.2f)
                    belt1 = idx;
            }

            int belt2 = num5 >= 0.2f ? (num5 >= 0.4f ? (num5 >= 0.8f ? 0 : lastOrbit + 1) : lastOrbit + 2) : lastOrbit + 3;
            if (belt2 != 0 && belt2 < 5) belt2 = 5;
            star.asterBelt1OrbitIndex = belt1;
            star.asterBelt2OrbitIndex = belt2;
            if (belt1 > 0)
                star.asterBelt1Radius = global::DspFindSeed.StarGen.orbitRadius[belt1] * num6 * star.orbitScaler;
            if (belt2 > 0)
                star.asterBelt2Radius = global::DspFindSeed.StarGen.orbitRadius[belt2] * num7 * star.orbitScaler;
        }
    }
}

