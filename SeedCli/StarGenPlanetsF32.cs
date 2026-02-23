using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine;

namespace SeedCli
{
    /// <summary>
    /// 实验用途：复制原 StarGen.CreateStarPlanets 的控制流，
    /// 但把 PlanetGen.CreatePlanet 调用替换为 PlanetGenF32.CreatePlanet。
    /// </summary>
    internal static class StarGenPlanetsF32
    {
        private sealed class ReferenceEqualityComparer<T> : IEqualityComparer<T>
            where T : class
        {
            public static readonly ReferenceEqualityComparer<T> Instance = new ReferenceEqualityComparer<T>();

            public bool Equals(T x, T y)
            {
                return ReferenceEquals(x, y);
            }

            public int GetHashCode(T obj)
            {
                return obj == null ? 0 : RuntimeHelpers.GetHashCode(obj);
            }
        }

        private sealed class GalaxyBatchGroup
        {
            public GalaxyData Galaxy;
            public List<PlanetBuildContext> Contexts = new List<PlanetBuildContext>(96);
        }

        private struct PlanetBuildPlan
        {
            public int index;
            public int orbitAround;
            public int orbitIndex;
            public int number;
            public bool gasGiant;
            public int infoSeed;
            public int genSeed;
        }

        private sealed class PlanetBuildContext
        {
            public GalaxyData galaxy;
            public StarData star;
            public int[] themeIds;
            public bool batchEnabled;
            public List<PlanetBuildPlan> plans;
            public double num4;
            public double num5;
            public double num6;
            public double num7;
        }

        [ThreadStatic] private static List<PlanetBuildContext> _pendingGalaxyBatch;

        public static void BeginGalaxyBatch()
        {
            _pendingGalaxyBatch = new List<PlanetBuildContext>(96);
        }

        public static void FlushGalaxyBatch()
        {
            var pending = _pendingGalaxyBatch;
            _pendingGalaxyBatch = null;
            if (pending == null || pending.Count == 0)
                return;
            FinalizeBufferedPlanetsBatch(pending);
        }

        // 复用原 StarGen.pGas 表（定义在 DspFindSeed.StarGen）
        public static void CreateStarPlanetsF32(GalaxyData galaxy, StarData star, global::DspFindSeed.GameDesc gameDesc)
        {
            var buildCtx = new PlanetBuildContext
            {
                galaxy = galaxy,
                star = star,
                themeIds = gameDesc.savedThemeIds,
                batchEnabled = CudaPlanetNative.IsCoreEnabled(),
                plans = new List<PlanetBuildPlan>(8)
            };

            DotNet35Random rng1 = new DotNet35Random(star.seed);
            rng1.Next(); rng1.Next(); rng1.Next();
            DotNet35Random rng2 = new DotNet35Random(rng1.Next());

            // 关键门限判断改回 FP64（与原版一致），避免边界分支翻转。
            double num1 = rng2.NextDouble();
            double num2 = rng2.NextDouble();
            double num3 = rng2.NextDouble();
            double num4 = rng2.NextDouble();
            double num5 = rng2.NextDouble();
            double num6 = rng2.NextDouble() * 0.2 + 0.9;
            double num7 = rng2.NextDouble() * 0.2 + 0.9;

            if (star.type == EStarType.BlackHole || star.type == EStarType.NeutronStar)
            {
                star.planetCount = 1;
                star.planets = new PlanetData[1];
                int info_seed = rng2.Next();
                int gen_seed = rng2.Next();
                star.planets[0] = CreatePlanetBuffered(buildCtx, 0, 0, 3, 1, false, info_seed, gen_seed);
            }
            else if (star.type == EStarType.WhiteDwarf)
            {
                if (num1 < 0.7)
                {
                    star.planetCount = 1;
                    star.planets = new PlanetData[1];
                    int info_seed = rng2.Next();
                    int gen_seed = rng2.Next();
                    star.planets[0] = CreatePlanetBuffered(buildCtx, 0, 0, 3, 1, false, info_seed, gen_seed);
                }
                else
                {
                    star.planetCount = 2;
                    star.planets = new PlanetData[2];
                    if (num2 < 0.30000001192092896)
                    {
                        int info_seed1 = rng2.Next();
                        int gen_seed1 = rng2.Next();
                        star.planets[0] = CreatePlanetBuffered(buildCtx, 0, 0, 3, 1, false, info_seed1, gen_seed1);
                        int info_seed2 = rng2.Next();
                        int gen_seed2 = rng2.Next();
                        star.planets[1] = CreatePlanetBuffered(buildCtx, 1, 0, 4, 2, false, info_seed2, gen_seed2);
                    }
                    else
                    {
                        int info_seed3 = rng2.Next();
                        int gen_seed3 = rng2.Next();
                        star.planets[0] = CreatePlanetBuffered(buildCtx, 0, 0, 4, 1, true, info_seed3, gen_seed3);
                        int info_seed4 = rng2.Next();
                        int gen_seed4 = rng2.Next();
                        star.planets[1] = CreatePlanetBuffered(buildCtx, 1, 1, 1, 1, false, info_seed4, gen_seed4);
                    }
                }
            }
            else if (star.type == EStarType.GiantStar)
            {
                if (num1 < 0.30000001192092896)
                {
                    star.planetCount = 1;
                    star.planets = new PlanetData[1];
                    int info_seed = rng2.Next();
                    int gen_seed = rng2.Next();
                    star.planets[0] = CreatePlanetBuffered(buildCtx, 0, 0, num3 > 0.5 ? 3 : 2, 1, false, info_seed, gen_seed);
                }
                else if (num1 < 0.800000011920929)
                {
                    star.planetCount = 2;
                    star.planets = new PlanetData[2];
                    if (num2 < 0.25)
                    {
                        int info_seed5 = rng2.Next();
                        int gen_seed5 = rng2.Next();
                        star.planets[0] = CreatePlanetBuffered(buildCtx, 0, 0, num3 > 0.5 ? 3 : 2, 1, false, info_seed5, gen_seed5);
                        int info_seed6 = rng2.Next();
                        int gen_seed6 = rng2.Next();
                        star.planets[1] = CreatePlanetBuffered(buildCtx, 1, 0, num3 > 0.5 ? 4 : 3, 2, false, info_seed6, gen_seed6);
                    }
                    else
                    {
                        int info_seed7 = rng2.Next();
                        int gen_seed7 = rng2.Next();
                        star.planets[0] = CreatePlanetBuffered(buildCtx, 0, 0, 3, 1, true, info_seed7, gen_seed7);
                        int info_seed8 = rng2.Next();
                        int gen_seed8 = rng2.Next();
                        star.planets[1] = CreatePlanetBuffered(buildCtx, 1, 1, 1, 1, false, info_seed8, gen_seed8);
                    }
                }
                else
                {
                    star.planetCount = 3;
                    star.planets = new PlanetData[3];
                    if (num2 < 0.15000000596046448)
                    {
                        int info_seed9 = rng2.Next();
                        int gen_seed9 = rng2.Next();
                        star.planets[0] = CreatePlanetBuffered(buildCtx, 0, 0, num3 > 0.5 ? 3 : 2, 1, false, info_seed9, gen_seed9);
                        int info_seed10 = rng2.Next();
                        int gen_seed10 = rng2.Next();
                        star.planets[1] = CreatePlanetBuffered(buildCtx, 1, 0, num3 > 0.5 ? 4 : 3, 2, false, info_seed10, gen_seed10);
                        int info_seed11 = rng2.Next();
                        int gen_seed11 = rng2.Next();
                        star.planets[2] = CreatePlanetBuffered(buildCtx, 2, 0, num3 > 0.5 ? 5 : 4, 3, false, info_seed11, gen_seed11);
                    }
                    else if (num2 < 0.75)
                    {
                        int info_seed12 = rng2.Next();
                        int gen_seed12 = rng2.Next();
                        star.planets[0] = CreatePlanetBuffered(buildCtx, 0, 0, num3 > 0.5 ? 3 : 2, 1, false, info_seed12, gen_seed12);
                        int info_seed13 = rng2.Next();
                        int gen_seed13 = rng2.Next();
                        star.planets[1] = CreatePlanetBuffered(buildCtx, 1, 0, 4, 2, true, info_seed13, gen_seed13);
                        int info_seed14 = rng2.Next();
                        int gen_seed14 = rng2.Next();
                        star.planets[2] = CreatePlanetBuffered(buildCtx, 2, 2, 1, 1, false, info_seed14, gen_seed14);
                    }
                    else
                    {
                        int info_seed15 = rng2.Next();
                        int gen_seed15 = rng2.Next();
                        star.planets[0] = CreatePlanetBuffered(buildCtx, 0, 0, num3 > 0.5 ? 4 : 3, 1, true, info_seed15, gen_seed15);
                        int info_seed16 = rng2.Next();
                        int gen_seed16 = rng2.Next();
                        star.planets[1] = CreatePlanetBuffered(buildCtx, 1, 1, 1, 1, false, info_seed16, gen_seed16);
                        int info_seed17 = rng2.Next();
                        int gen_seed17 = rng2.Next();
                        star.planets[2] = CreatePlanetBuffered(buildCtx, 2, 1, 2, 2, false, info_seed17, gen_seed17);
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
                    star.planetCount = num1 >= 0.1 ? (num1 >= 0.3 ? (num1 >= 0.8 ? 4 : 3) : 2) : 1;
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
                    star.planetCount = num1 >= 0.1 ? (num1 >= 0.2 ? (num1 >= 0.7 ? (num1 >= 0.95 ? 5 : 4) : 3) : 2) : 1;
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
                    star.planetCount = num1 >= 0.4 ? (num1 >= 0.9 ? 5 : 4) : 3;
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
                    star.planetCount = num1 >= 0.35 ? (num1 >= 0.8 ? 5 : 4) : 3;
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
                    star.planetCount = num1 >= 0.3 ? (num1 >= 0.75 ? 5 : 4) : 3;
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
                    star.planetCount = num1 >= 0.3 ? (num1 >= 0.75 ? 6 : 5) : 4;
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
                    star.planetCount = num1 >= 0.5 ? 6 : 5;
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
                    double num11 = rng2.NextDouble();
                    double num12 = rng2.NextDouble();
                    bool gasGiant = false;
                    if (orbitAround == 0)
                    {
                        ++num8;
                        if (index < star.planetCount - 1 && num11 < pGas[index])
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
                                double prob = num10 <= 3
                                    ? Mathf.Lerp(a, 1f, 0.15f) + 0.01f
                                    : Mathf.Lerp(a, 1f, 0.45f) + 0.01f;
                                if (rng2.NextDouble() < prob) goto label_62;
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
                    star.planets[index] = CreatePlanetBuffered(buildCtx, index, orbitAround, orbitAround == 0 ? num10 : num9, orbitAround == 0 ? num8 : num9, gasGiant, info_seed, gen_seed);
                    ++num10;
                    if (gasGiant)
                    {
                        orbitAround = num8;
                        num9 = 0;
                    }
                    if (num9 >= 1 && num12 < 0.8)
                    {
                        orbitAround = 0;
                        num9 = 0;
                    }
                }
            }

            buildCtx.num4 = num4;
            buildCtx.num5 = num5;
            buildCtx.num6 = num6;
            buildCtx.num7 = num7;
            if (buildCtx.batchEnabled && _pendingGalaxyBatch != null)
            {
                _pendingGalaxyBatch.Add(buildCtx);
                return;
            }

            FinalizeBufferedPlanets(buildCtx);
            ApplyAsteroidBelts(star, num4, num5, num6, num7);
        }

        private static PlanetData CreatePlanetBuffered(
            PlanetBuildContext ctx,
            int index,
            int orbitAround,
            int orbitIndex,
            int number,
            bool gasGiant,
            int infoSeed,
            int genSeed)
        {
            if (!ctx.batchEnabled)
                return PlanetGenF32.CreatePlanet(ctx.galaxy, ctx.star, ctx.themeIds, index, orbitAround, orbitIndex, number, gasGiant, infoSeed, genSeed);

            var planet = PlanetGenF32.CreatePlanetShell(ctx.galaxy, ctx.star, index, orbitAround, orbitIndex, number, genSeed, infoSeed);
            ctx.plans.Add(new PlanetBuildPlan
            {
                index = index,
                orbitAround = orbitAround,
                orbitIndex = orbitIndex,
                number = number,
                gasGiant = gasGiant,
                infoSeed = infoSeed,
                genSeed = genSeed
            });
            return planet;
        }

        private static CudaPlanetNative.PlanetCoreBatchInput BuildCoreBatchInput(
            PlanetBuildContext ctx,
            PlanetData planet,
            in PlanetBuildPlan plan)
        {
            int compactTypeCase = 0;
            if (ctx.star.type == EStarType.WhiteDwarf) compactTypeCase = 1;
            else if (ctx.star.type == EStarType.NeutronStar) compactTypeCase = 2;
            else if (ctx.star.type == EStarType.BlackHole) compactTypeCase = 3;

            return new CudaPlanetNative.PlanetCoreBatchInput
            {
                infoSeed = plan.infoSeed,
                orbitAround = plan.orbitAround,
                orbitIndex = plan.orbitIndex,
                gasGiant = plan.gasGiant ? 1 : 0,
                starIndex = ctx.star.index,
                galaxyStarCount = ctx.star.galaxy.starCount,
                // 彻底批处理路径中，类型/宜居增量在 CPU 顺序重算，GPU 仅负责重数值段。
                galaxyHabitableCount = 0,
                boostInclinationNs = ctx.star.type >= EStarType.NeutronStar ? 1 : 0,
                compactTypeCase = compactTypeCase,
                starOrbitScaler = ctx.star.orbitScaler,
                starMass = ctx.star.mass,
                starHabitableRadius = ctx.star.habitableRadius,
                starLightBalanceRadius = ctx.star.lightBalanceRadius,
                orbitAroundPlanetRealRadius = planet.orbitAroundPlanet != null ? planet.orbitAroundPlanet.realRadius : 0f,
                orbitAroundPlanetOrbitRadius = planet.orbitAroundPlanet != null ? planet.orbitAroundPlanet.orbitRadius : 0f,
                orbitAroundPlanetOrbitalPeriod = planet.orbitAroundPlanet != null ? planet.orbitAroundPlanet.orbitalPeriod : 0.0
            };
        }

        private static void ApplyAsteroidBelts(StarData star, double num4, double num5, double num6, double num7)
        {
            if (MixRuntimeFlags.SignatureOnlyFastPath)
                return;

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
                if (ok && num4 < 0.2 + idx * 0.2)
                    belt1 = idx;
            }

            int belt2 = num5 >= 0.2 ? (num5 >= 0.4 ? (num5 >= 0.8 ? 0 : lastOrbit + 1) : lastOrbit + 2) : lastOrbit + 3;
            if (belt2 != 0 && belt2 < 5) belt2 = 5;
            star.asterBelt1OrbitIndex = belt1;
            star.asterBelt2OrbitIndex = belt2;
            if (belt1 > 0)
                star.asterBelt1Radius = (float)(global::DspFindSeed.StarGen.orbitRadius[belt1] * num6 * star.orbitScaler);
            if (belt2 > 0)
                star.asterBelt2Radius = (float)(global::DspFindSeed.StarGen.orbitRadius[belt2] * num7 * star.orbitScaler);
        }

        private static void FinalizeBufferedPlanetsBatch(IList<PlanetBuildContext> contexts)
        {
            if (contexts == null || contexts.Count == 0)
                return;

            var coreByCtx = new Dictionary<PlanetBuildContext, CudaPlanetNative.PlanetCoreF32Out[]>(
                contexts.Count,
                ReferenceEqualityComparer<PlanetBuildContext>.Instance);
            var groupMap = new Dictionary<GalaxyData, GalaxyBatchGroup>(ReferenceEqualityComparer<GalaxyData>.Instance);
            var groups = new List<GalaxyBatchGroup>(Math.Min(contexts.Count, 64));
            for (int i = 0; i < contexts.Count; ++i)
            {
                var ctx = contexts[i];
                if (ctx?.plans == null || ctx.plans.Count == 0)
                    continue;
                coreByCtx[ctx] = new CudaPlanetNative.PlanetCoreF32Out[ctx.plans.Count];

                var galaxy = ctx.star?.galaxy;
                if (galaxy == null)
                    continue;
                if (!groupMap.TryGetValue(galaxy, out var group))
                {
                    group = new GalaxyBatchGroup { Galaxy = galaxy };
                    groupMap[galaxy] = group;
                    groups.Add(group);
                }
                group.Contexts.Add(ctx);
            }

            var phaseInputs = new List<CudaPlanetNative.PlanetCoreBatchInput>(4096);
            var phaseCtx = new List<PlanetBuildContext>(4096);
            var phasePlanIdx = new List<int>(4096);

            void ResetPhase()
            {
                phaseInputs.Clear();
                phaseCtx.Clear();
                phasePlanIdx.Clear();
            }

            bool CollectPhase(bool primary)
            {
                ResetPhase();
                for (int ci = 0; ci < contexts.Count; ++ci)
                {
                    var ctx = contexts[ci];
                    if (ctx == null || ctx.plans == null || ctx.plans.Count == 0)
                        continue;
                    for (int pi = 0; pi < ctx.plans.Count; ++pi)
                    {
                        var plan = ctx.plans[pi];
                        bool isPrimary = plan.orbitAround == 0;
                        if (primary != isPrimary)
                            continue;
                        var planet = ctx.star.planets[plan.index];
                        phaseInputs.Add(BuildCoreBatchInput(ctx, planet, plan));
                        phaseCtx.Add(ctx);
                        phasePlanIdx.Add(pi);
                    }
                }
                return phaseInputs.Count > 0;
            }

            void FallbackAll()
            {
                for (int gi = 0; gi < groups.Count; ++gi)
                {
                    var group = groups[gi];
                    int runningHabitable = group.Galaxy.habitableCount;
                    for (int ci = 0; ci < group.Contexts.Count; ++ci)
                    {
                        var ctx = group.Contexts[ci];
                        if (ctx == null || ctx.plans == null || ctx.plans.Count == 0)
                            continue;
                        ctx.star.galaxy.habitableCount = runningHabitable;
                        RebuildFallback(ctx);
                        runningHabitable = ctx.star.galaxy.habitableCount;
                        ApplyAsteroidBelts(ctx.star, ctx.num4, ctx.num5, ctx.num6, ctx.num7);
                    }
                }
            }

            if (CollectPhase(primary: true))
            {
                var phaseOut = new CudaPlanetNative.PlanetCoreF32Out[phaseInputs.Count];
                if (!CudaPlanetNative.TryEvalPlanetCoreF32Batch(phaseInputs, phaseOut))
                {
                    FallbackAll();
                    return;
                }
                for (int i = 0; i < phaseOut.Length; ++i)
                {
                    var ctx = phaseCtx[i];
                    int planIdx = phasePlanIdx[i];
                    coreByCtx[ctx][planIdx] = phaseOut[i];
                    var p = ctx.star.planets[ctx.plans[planIdx].index];
                    PlanetGenF32.ApplyCoreOrbitState(p, phaseOut[i]);
                }
            }

            if (CollectPhase(primary: false))
            {
                var phaseOut = new CudaPlanetNative.PlanetCoreF32Out[phaseInputs.Count];
                if (!CudaPlanetNative.TryEvalPlanetCoreF32Batch(phaseInputs, phaseOut))
                {
                    FallbackAll();
                    return;
                }
                for (int i = 0; i < phaseOut.Length; ++i)
                {
                    var ctx = phaseCtx[i];
                    int planIdx = phasePlanIdx[i];
                    coreByCtx[ctx][planIdx] = phaseOut[i];
                }
            }

            for (int gi = 0; gi < groups.Count; ++gi)
            {
                var group = groups[gi];
                int runningHabitableFinal = group.Galaxy.habitableCount;
                for (int ci = 0; ci < group.Contexts.Count; ++ci)
                {
                    var ctx = group.Contexts[ci];
                    if (ctx == null || ctx.plans == null || ctx.plans.Count == 0)
                        continue;

                    ctx.star.galaxy.habitableCount = runningHabitableFinal;
                    var coreResults = coreByCtx[ctx];
                    for (int i = 0; i < ctx.plans.Count; ++i)
                    {
                        var plan = ctx.plans[i];
                        var p = ctx.star.planets[plan.index];
                        var core = coreResults[i];
                        PlanetGenF32.RecomputeCoreTypeByHabitableCount(
                            ref core,
                            plan.infoSeed,
                            plan.orbitAround,
                            plan.orbitIndex,
                            plan.gasGiant,
                            ctx.star,
                            ctx.star.galaxy.habitableCount);
                        PlanetGenF32.ApplyCoreAndTheme(p, ctx.themeIds, ref core);
                    }

                    runningHabitableFinal = ctx.star.galaxy.habitableCount;
                    ApplyAsteroidBelts(ctx.star, ctx.num4, ctx.num5, ctx.num6, ctx.num7);
                }
            }
        }

        private static void FinalizeBufferedPlanets(PlanetBuildContext ctx)
        {
            if (!ctx.batchEnabled || ctx.plans == null || ctx.plans.Count == 0)
                return;

            int habitableBefore = ctx.star.galaxy.habitableCount;
            int n = ctx.plans.Count;
            var coreResults = new CudaPlanetNative.PlanetCoreF32Out[n];

            var phaseInputs = new List<CudaPlanetNative.PlanetCoreBatchInput>(n);
            var phaseMap = new List<int>(n);

            // Phase-1: 主轨行星，先拿到父星体轨道/半径供卫星请求使用。
            for (int i = 0; i < n; ++i)
            {
                var plan = ctx.plans[i];
                if (plan.orbitAround != 0)
                    continue;
                var planet = ctx.star.planets[plan.index];
                phaseInputs.Add(BuildCoreBatchInput(ctx, planet, plan));
                phaseMap.Add(i);
            }

            if (phaseInputs.Count > 0)
            {
                var phaseOut = new CudaPlanetNative.PlanetCoreF32Out[phaseInputs.Count];
                if (!CudaPlanetNative.TryEvalPlanetCoreF32Batch(phaseInputs, phaseOut))
                {
                    ctx.star.galaxy.habitableCount = habitableBefore;
                    RebuildFallback(ctx);
                    return;
                }
                for (int i = 0; i < phaseMap.Count; ++i)
                    coreResults[phaseMap[i]] = phaseOut[i];
            }

            for (int i = 0; i < phaseMap.Count; ++i)
            {
                int pi = phaseMap[i];
                var plan = ctx.plans[pi];
                var p = ctx.star.planets[plan.index];
                PlanetGenF32.ApplyCoreOrbitState(p, coreResults[pi]);
            }

            // Phase-2: 卫星行星（依赖父行星 realRadius/orbit 信息）
            phaseInputs.Clear();
            phaseMap.Clear();
            for (int i = 0; i < n; ++i)
            {
                var plan = ctx.plans[i];
                if (plan.orbitAround == 0)
                    continue;
                var planet = ctx.star.planets[plan.index];
                phaseInputs.Add(BuildCoreBatchInput(ctx, planet, plan));
                phaseMap.Add(i);
            }

            if (phaseInputs.Count > 0)
            {
                var phaseOut = new CudaPlanetNative.PlanetCoreF32Out[phaseInputs.Count];
                if (!CudaPlanetNative.TryEvalPlanetCoreF32Batch(phaseInputs, phaseOut))
                {
                    ctx.star.galaxy.habitableCount = habitableBefore;
                    RebuildFallback(ctx);
                    return;
                }
                for (int i = 0; i < phaseMap.Count; ++i)
                    coreResults[phaseMap[i]] = phaseOut[i];
            }

            // 顺序回放：按原创建顺序重算类型/宜居分支，保证主题去重与 habitableCount 链式一致。
            ctx.star.galaxy.habitableCount = habitableBefore;
            for (int i = 0; i < n; ++i)
            {
                var plan = ctx.plans[i];
                var p = ctx.star.planets[plan.index];
                var core = coreResults[i];
                PlanetGenF32.RecomputeCoreTypeByHabitableCount(
                    ref core,
                    plan.infoSeed,
                    plan.orbitAround,
                    plan.orbitIndex,
                    plan.gasGiant,
                    ctx.star,
                    ctx.star.galaxy.habitableCount);
                PlanetGenF32.ApplyCoreAndTheme(p, ctx.themeIds, ref core);
            }
        }

        private static void RebuildFallback(PlanetBuildContext ctx)
        {
            for (int i = 0; i < ctx.plans.Count; ++i)
            {
                var plan = ctx.plans[i];
                ctx.star.planets[plan.index] = PlanetGenF32.CreatePlanet(
                    ctx.galaxy,
                    ctx.star,
                    ctx.themeIds,
                    plan.index,
                    plan.orbitAround,
                    plan.orbitIndex,
                    plan.number,
                    plan.gasGiant,
                    plan.infoSeed,
                    plan.genSeed);
            }
        }
    }
}
