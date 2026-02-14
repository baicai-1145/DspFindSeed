using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;

namespace SeedCli
{
    internal static class Program
    {
        // 用法示例：
        //   SeedCli.exe --seed 12345678 --stars 64
        //   SeedCli.exe --seed 12345678 --stars 32 --resource 1
        private static int Main(string[] args)
        {
            if (args == null) args = Array.Empty<string>();

            int seed = GetIntArg(args, "--seed", 0);
            int stars = GetIntArg(args, "--stars", 64);
            int count = GetIntArg(args, "--count", 1);
            bool comparePlanetsF32 = HasFlag(args, "--compare-planets-f32");
            bool compareVeinsF32 = HasFlag(args, "--compare-veins-f32");
            bool comparePipelineF32 = HasFlag(args, "--compare-pipeline-f32");
            bool comparePipelineMix = HasFlag(args, "--compare-pipeline-mix");
            bool comparePipelineMixVeinsF32 = HasFlag(args, "--compare-pipeline-mix-veins-f32");
            bool compareF32 = HasFlag(args, "--compare-f32");
            bool compareF32Pt64 = HasFlag(args, "--compare-f32-pt64");
            bool compareF32Pt64Coll64 = HasFlag(args, "--compare-f32-pt64-coll64");
            bool compareF32Pt64Rand64 = HasFlag(args, "--compare-f32-pt64-rand64");
            bool compareF32Pt64Rand64Coll64 = HasFlag(args, "--compare-f32-pt64-rand64-coll64");
            bool compareF32Pt64Rand64Params64 = HasFlag(args, "--compare-f32-pt64-rand64-params64");
            bool compareF32Pt64Rand64Params64Coll64 = HasFlag(args, "--compare-f32-pt64-rand64-params64-coll64");
            bool benchGalaxyOnly = HasFlag(args, "--bench-galaxy-only");
            bool collisionFp64 = HasFlag(args, "--collision-fp64");
            bool useCudaGalaxy = HasFlag(args, "--use-cuda-galaxy");
            int batchSize = GetIntArg(args, "--batch-size", 1024);
            int showMismatches = GetIntArg(args, "--show-mismatches", 0);

            if (seed <= 0)
            {
                Console.WriteLine("用法：SeedCli.exe --seed <种子ID> [--stars <星区数量>]");
                Console.WriteLine("说明：会优先从 Release/Prototypes 或 Prototypes 读取 *ProtoSet.json，再回退 xml。");
                Console.WriteLine("对比实验：SeedCli.exe --seed <起始种子> --stars <星区数量> --count <个数> --compare-f32");
                Console.WriteLine("行星 FP32 实验：SeedCli.exe --seed <起始种子> --stars <星区数量> --count <个数> --compare-planets-f32");
                Console.WriteLine("矿堆数 FP32 实验：SeedCli.exe --seed <起始种子> --stars <星区数量> --count <个数> --compare-veins-f32");
                Console.WriteLine("端到端 FP32 实验：SeedCli.exe --seed <起始种子> --stars <星区数量> --count <个数> --compare-pipeline-f32");
                Console.WriteLine("端到端 Mix 实验（星系 0 mismatch 变体）：SeedCli.exe --seed <起始种子> --stars <星区数量> --count <个数> --compare-pipeline-mix");
                Console.WriteLine("端到端 Mix+矿 FP32 实验：SeedCli.exe --seed <起始种子> --stars <星区数量> --count <个数> --compare-pipeline-mix-veins-f32");
                Console.WriteLine("启用 CUDA 星系点位（仅 ParamsFp64 路线）：追加 --use-cuda-galaxy（或环境变量 DSP_USE_CUDA_GALAXY=1）");
                Console.WriteLine("仅星系生成并行基准：SeedCli.exe --seed <起始种子> --stars <星区数量> --count <个数> --bench-galaxy-only [--batch-size 1024] [--collision-fp64]");
                Console.WriteLine("差异打印：SeedCli.exe --seed <起始种子> --stars <星区数量> --count <个数> --compare-f32 --show-mismatches <最多打印条数>");
                return 2;
            }

            try
            {
                if (benchGalaxyOnly)
                {
                    CudaGalaxyNative.EnableByCli(true);
                    BenchGalaxyOnly(seed, stars, count, batchSize, collisionFp64);
                    return 0;
                }

                CudaGalaxyNative.EnableByCli(useCudaGalaxy);
                InitForSearch();
                if (comparePipelineMix || comparePipelineMixVeinsF32)
                {
                    ComparePipelineMix(seed, stars, count, showMismatches, useFp32Veins: comparePipelineMixVeinsF32);
                }
                else if (comparePipelineF32)
                {
                    ComparePipelineF32(seed, stars, count, showMismatches);
                }
                else if (compareVeinsF32)
                {
                    CompareVeinsF32(seed, stars, count, showMismatches);
                }
                else if (comparePlanetsF32)
                {
                    ComparePlanetsF32(seed, stars, count, showMismatches);
                }
                else if (compareF32Pt64)
                {
                    CompareF32(seed, stars, count, showMismatches, usePtFp64Variant: true);
                }
                else if (compareF32Pt64Coll64)
                {
                    CompareF32(seed, stars, count, showMismatches, usePtFp64Variant: true, useCollFp64Variant: true);
                }
                else if (compareF32Pt64Rand64)
                {
                    CompareF32(seed, stars, count, showMismatches, usePtFp64Variant: true, useCollFp64Variant: false, useRandFp64Variant: true);
                }
                else if (compareF32Pt64Rand64Coll64)
                {
                    CompareF32(seed, stars, count, showMismatches, usePtFp64Variant: true, useCollFp64Variant: true, useRandFp64Variant: true);
                }
                else if (compareF32Pt64Rand64Params64)
                {
                    CompareF32(seed, stars, count, showMismatches, usePtFp64Variant: true, useCollFp64Variant: false, useRandFp64Variant: true, useParamsFp64Variant: true);
                }
                else if (compareF32Pt64Rand64Params64Coll64)
                {
                    CompareF32(seed, stars, count, showMismatches, usePtFp64Variant: true, useCollFp64Variant: true, useRandFp64Variant: true, useParamsFp64Variant: true);
                }
                else if (compareF32)
                {
                    CompareF32(seed, stars, count, showMismatches, usePtFp64Variant: false);
                }
                else
                {
                    DumpSeed(seed, stars);
                }
                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine("运行失败：");
                Console.WriteLine(ex);
                return 1;
            }
        }

        private static void BenchGalaxyOnly(int startSeed, int starCount, int count, int batchSize, bool collisionFp64)
        {
            if (count < 1) count = 1;
            if (batchSize < 1) batchSize = 1;

            const ulong FNV_OFFSET = 14695981039346656037UL;
            const ulong FNV_PRIME = 1099511628211UL;

            ulong cpuAgg = FNV_OFFSET;
            ulong gpuAgg = FNV_OFFSET;
            int total = 0;
            int mismatch = 0;
            bool gpuAvailable = true;

            long cpuTicks = 0;
            long gpuTicks = 0;

            int maxCount = Math.Max(1, starCount * 4);
            var cpuPoses = new List<VectorLF3>(maxCount);
            int endSeed = startSeed + count;

            for (int seedBase = startSeed; seedBase < endSeed; seedBase += batchSize)
            {
                int chunk = Math.Min(batchSize, endSeed - seedBase);
                var galaxySeeds = new int[chunk];
                var poseSeeds = new int[chunk];
                var cpuSig = new ulong[chunk];

                for (int i = 0; i < chunk; ++i)
                {
                    int galaxySeed = seedBase + i;
                    galaxySeeds[i] = galaxySeed;
                    poseSeeds[i] = CalcTempPoseSeed(galaxySeed);
                }

                for (int i = 0; i < chunk; ++i)
                {
                    long t0 = Stopwatch.GetTimestamp();
                    int poseCount = UniverseGenF32.GenerateTempPosesOnly_ParamsFp64_Cpu(galaxySeeds[i], starCount, collisionFp64, cpuPoses);
                    long t1 = Stopwatch.GetTimestamp();
                    cpuTicks += (t1 - t0);

                    ulong sig = PoseSignature(cpuPoses, poseCount);
                    cpuSig[i] = sig;
                    cpuAgg ^= sig;
                    cpuAgg *= FNV_PRIME;
                }

                if (gpuAvailable)
                {
                    long t0 = Stopwatch.GetTimestamp();
                    bool ok = CudaGalaxyNative.TryGenerateRandomPosesParamsFp64Batch(
                        poseSeeds,
                        maxCount,
                        minDist: 2.0,
                        minStepLen: 2.3,
                        maxStepLen: 3.5,
                        flatten: 0.18,
                        collisionFp64: collisionFp64,
                        out var gpuPoses,
                        out var gpuCounts,
                        out var outStride);
                    long t1 = Stopwatch.GetTimestamp();
                    gpuTicks += (t1 - t0);

                    if (!ok || gpuPoses == null || gpuCounts == null)
                    {
                        gpuAvailable = false;
                    }
                    else
                    {
                        for (int i = 0; i < chunk; ++i)
                        {
                            int pc = gpuCounts[i];
                            ulong sig = PoseSignatureAfterCpuTrim(
                                gpuPoses,
                                i * outStride,
                                pc,
                                starCount,
                                iterCount: 4);
                            gpuAgg ^= sig;
                            gpuAgg *= FNV_PRIME;
                            if (sig != cpuSig[i])
                                mismatch++;
                        }
                    }
                }

                total += chunk;
            }

            double cpuMs = cpuTicks * 1000.0 / Stopwatch.Frequency;
            double gpuMs = gpuTicks * 1000.0 / Stopwatch.Frequency;
            Console.WriteLine($"bench-galaxy-only startSeed={startSeed} stars={starCount} count={count} batchSize={batchSize} collisionFp64={collisionFp64}");
            Console.WriteLine($"cpuTimeMs={cpuMs:F3}");
            if (gpuAvailable)
            {
                Console.WriteLine($"gpuTimeMs={gpuMs:F3}");
                Console.WriteLine($"speedup={((gpuMs > 0.0) ? (cpuMs / gpuMs) : 0.0):F3}x");
                Console.WriteLine($"poseMismatch={mismatch}/{total} ({(total > 0 ? mismatch * 100.0 / total : 0):F6}%)");
                Console.WriteLine($"cpuPoseSig=0x{cpuAgg:X16} gpuPoseSig=0x{gpuAgg:X16}");
            }
            else
            {
                Console.WriteLine("gpuTimeMs=N/A (CUDA path unavailable, fallback/disabled)");
            }
        }

        private static int CalcTempPoseSeed(int galaxySeed)
        {
            var rng = new DotNet35Random(galaxySeed);
            return rng.Next();
        }

        private static ulong PoseSignature(List<VectorLF3> poses, int count)
        {
            unchecked
            {
                const ulong FNV_OFFSET = 14695981039346656037UL;
                const ulong FNV_PRIME = 1099511628211UL;
                ulong h = FNV_OFFSET;
                h ^= (uint)count;
                h *= FNV_PRIME;
                int c = Math.Min(count, poses != null ? poses.Count : 0);
                for (int i = 0; i < c; ++i)
                {
                    var p = poses[i];
                    h ^= (ulong)BitConverter.DoubleToInt64Bits(p.x);
                    h *= FNV_PRIME;
                    h ^= (ulong)BitConverter.DoubleToInt64Bits(p.y);
                    h *= FNV_PRIME;
                    h ^= (ulong)BitConverter.DoubleToInt64Bits(p.z);
                    h *= FNV_PRIME;
                }
                return h;
            }
        }

        private static ulong PoseSignature(CudaGalaxyNative.NativeVec3d[] poses, int offset, int count)
        {
            unchecked
            {
                const ulong FNV_OFFSET = 14695981039346656037UL;
                const ulong FNV_PRIME = 1099511628211UL;
                ulong h = FNV_OFFSET;
                h ^= (uint)count;
                h *= FNV_PRIME;
                if (poses == null || offset < 0 || count < 0 || offset >= poses.Length)
                    return h;
                int end = Math.Min(offset + count, poses.Length);
                for (int i = offset; i < end; ++i)
                {
                    var p = poses[i];
                    h ^= (ulong)BitConverter.DoubleToInt64Bits(p.x);
                    h *= FNV_PRIME;
                    h ^= (ulong)BitConverter.DoubleToInt64Bits(p.y);
                    h *= FNV_PRIME;
                    h ^= (ulong)BitConverter.DoubleToInt64Bits(p.z);
                    h *= FNV_PRIME;
                }
                return h;
            }
        }

        private static ulong PoseSignatureAfterCpuTrim(
            CudaGalaxyNative.NativeVec3d[] poses,
            int offset,
            int rawCount,
            int targetCount,
            int iterCount)
        {
            if (iterCount < 1) iterCount = 1;
            if (iterCount > 16) iterCount = 16;
            if (targetCount < 0) targetCount = 0;

            if (poses == null || offset < 0 || rawCount <= 0 || offset >= poses.Length)
                return PoseSignature(poses, 0, 0);

            int end = Math.Min(offset + rawCount, poses.Length);
            var tmp = new List<CudaGalaxyNative.NativeVec3d>(end - offset);
            for (int i = offset; i < end; ++i)
                tmp.Add(poses[i]);

            for (int i = tmp.Count - 1; i >= 0; --i)
            {
                if (i % iterCount != 0)
                    tmp.RemoveAt(i);
                if (tmp.Count <= targetCount)
                    break;
            }

            unchecked
            {
                const ulong FNV_OFFSET = 14695981039346656037UL;
                const ulong FNV_PRIME = 1099511628211UL;
                ulong h = FNV_OFFSET;
                h ^= (uint)tmp.Count;
                h *= FNV_PRIME;
                for (int i = 0; i < tmp.Count; ++i)
                {
                    var p = tmp[i];
                    h ^= (ulong)BitConverter.DoubleToInt64Bits(p.x);
                    h *= FNV_PRIME;
                    h ^= (ulong)BitConverter.DoubleToInt64Bits(p.y);
                    h *= FNV_PRIME;
                    h ^= (ulong)BitConverter.DoubleToInt64Bits(p.z);
                    h *= FNV_PRIME;
                }
                return h;
            }
        }

        private static void InitForSearch()
        {
            // MainWindow 构造里做的初始化，这里复制最小集合
            RandomTable.Init();
            LoadThemeIds();
            global::DspFindSeed.PlanetModelingManager.Start();
        }

        private static void LoadThemeIds()
        {
            // 显式使用 global::DspFindSeed.LDB，避免误用游戏原版 LDB（它依赖 Unity 的 Resources.Load）
            var themeArr = global::DspFindSeed.LDB.themes.dataArray;
            global::DspFindSeed.ThemeProto.themeIds = new int[themeArr.Length];
            for (int i = 0; i < themeArr.Length; i++)
                global::DspFindSeed.ThemeProto.themeIds[i] = themeArr[i].ID;
        }

        private static void CompareF32(int startSeed, int starCount, int count, int showMismatches, bool usePtFp64Variant, bool useCollFp64Variant = false, bool useRandFp64Variant = false, bool useParamsFp64Variant = false)
        {
            if (count < 1) count = 1;

            int mismatch = 0;
            int total = 0;
            int shown = 0;

            // 统计“第一处差异”分布
            var firstDiffKindCount = new Dictionary<string, int>(StringComparer.Ordinal);
            var firstDiffStarIndexCount = new Dictionary<int, int>();

            for (int s = startSeed; s < startSeed + count; s++)
            {
                var gd = new global::DspFindSeed.GameDesc();
                gd.SetForNewGame(global::DspFindSeed.UniverseGen.algoVersion, s, starCount, 1, 1f);

                var g64 = global::DspFindSeed.UniverseGen.CreateGalaxy(gd);
                GalaxyData g32;
                if (useParamsFp64Variant && useRandFp64Variant && useCollFp64Variant)
                    g32 = UniverseGenF32.CreateGalaxy_PtFp64_RandFp64_ParamsFp64_CollFp64(gd);
                else if (useParamsFp64Variant && useRandFp64Variant)
                    g32 = UniverseGenF32.CreateGalaxy_PtFp64_RandFp64_ParamsFp64(gd);
                else if (useRandFp64Variant && useCollFp64Variant)
                    g32 = UniverseGenF32.CreateGalaxy_PtFp64_RandFp64_CollFp64(gd);
                else if (useRandFp64Variant)
                    g32 = UniverseGenF32.CreateGalaxy_PtFp64_RandFp64(gd);
                else if (useCollFp64Variant)
                    g32 = UniverseGenF32.CreateGalaxy_PtFp64_CollFp64(gd);
                else
                    g32 = usePtFp64Variant ? UniverseGenF32.CreateGalaxy_PtFp64(gd) : UniverseGenF32.CreateGalaxy(gd);

                ulong h64 = Signature(g64);
                ulong h32 = Signature(g32);

                total++;
                if (h64 != h32)
                {
                    mismatch++;

                    var diff = FirstDifference(g64, g32);
                    if (!string.IsNullOrEmpty(diff.Kind))
                    {
                        if (!firstDiffKindCount.TryGetValue(diff.Kind, out var kc)) kc = 0;
                        firstDiffKindCount[diff.Kind] = kc + 1;
                    }
                    if (diff.StarIndex >= 0)
                    {
                        if (!firstDiffStarIndexCount.TryGetValue(diff.StarIndex, out var sc)) sc = 0;
                        firstDiffStarIndexCount[diff.StarIndex] = sc + 1;
                    }

                    if (showMismatches > 0 && shown < showMismatches)
                    {
                        shown++;
                        Console.WriteLine();
                        Console.WriteLine($"--- mismatch #{shown} seed={s} stars={starCount} ---");
                        Console.WriteLine($"sig64=0x{h64:X16} sig32=0x{h32:X16}");
                        Console.WriteLine(DescribeFirstDifference(g64, g32));
                    }
                }
            }

            var label =
                (useParamsFp64Variant && useRandFp64Variant && useCollFp64Variant) ? "compare-f32-pt64-rand64-params64-coll64"
                : (useParamsFp64Variant && useRandFp64Variant) ? "compare-f32-pt64-rand64-params64"
                : (useRandFp64Variant && useCollFp64Variant) ? "compare-f32-pt64-rand64-coll64"
                : (useRandFp64Variant ? "compare-f32-pt64-rand64"
                : (useCollFp64Variant ? "compare-f32-pt64-coll64" : (usePtFp64Variant ? "compare-f32-pt64" : "compare-f32")));
            Console.WriteLine($"{label} startSeed={startSeed} stars={starCount} count={count}");
            Console.WriteLine($"mismatch={mismatch}/{total} ({(total > 0 ? (mismatch * 100.0 / total) : 0):F2}%)");
            Console.WriteLine("说明：mismatch 表示“星系/行星结构签名不同”，实际筛选条件的命中差异通常也会随之显著变化。");

            if (mismatch > 0)
            {
                Console.WriteLine();
                Console.WriteLine("第一处差异字段分布（Top 10）：");
                foreach (var kv in firstDiffKindCount.OrderByDescending(k => k.Value).ThenBy(k => k.Key).Take(10))
                    Console.WriteLine($"  {kv.Key}: {kv.Value}");

                Console.WriteLine("第一处差异 Star 索引分布（Top 10）：");
                foreach (var kv in firstDiffStarIndexCount.OrderByDescending(k => k.Value).ThenBy(k => k.Key).Take(10))
                    Console.WriteLine($"  Star[{kv.Key}]: {kv.Value}");
            }
        }

        private static void ComparePlanetsF32(int startSeed, int starCount, int count, int showMismatches)
        {
            if (count < 1) count = 1;

            int mismatch = 0;
            int total = 0;
            int shown = 0;

            var firstDiffKindCount = new Dictionary<string, int>(StringComparer.Ordinal);
            var firstDiffStarIndexCount = new Dictionary<int, int>();

            for (int s = startSeed; s < startSeed + count; s++)
            {
                var gd = new global::DspFindSeed.GameDesc();
                gd.SetForNewGame(global::DspFindSeed.UniverseGen.algoVersion, s, starCount, 1, 1f);

                var g64 = global::DspFindSeed.UniverseGen.CreateGalaxy(gd);
                var gF32 = UniverseGenPlanetsF32.CreateGalaxy(gd);

                ulong h64 = Signature(g64);
                ulong h32 = Signature(gF32);

                total++;
                if (h64 != h32)
                {
                    mismatch++;

                    var diff = FirstDifference(g64, gF32);
                    if (!string.IsNullOrEmpty(diff.Kind))
                    {
                        if (!firstDiffKindCount.TryGetValue(diff.Kind, out var kc)) kc = 0;
                        firstDiffKindCount[diff.Kind] = kc + 1;
                    }
                    if (diff.StarIndex >= 0)
                    {
                        if (!firstDiffStarIndexCount.TryGetValue(diff.StarIndex, out var sc)) sc = 0;
                        firstDiffStarIndexCount[diff.StarIndex] = sc + 1;
                    }

                    if (showMismatches > 0 && shown < showMismatches)
                    {
                        shown++;
                        Console.WriteLine();
                        Console.WriteLine($"--- mismatch #{shown} seed={s} stars={starCount} ---");
                        Console.WriteLine($"sig64=0x{h64:X16} sigF32=0x{h32:X16}");
                        Console.WriteLine(DescribeFirstDifference(g64, gF32));
                    }
                }
            }

            Console.WriteLine($"compare-planets-f32 startSeed={startSeed} stars={starCount} count={count}");
            Console.WriteLine($"mismatch={mismatch}/{total} ({(total > 0 ? (mismatch * 100.0 / total) : 0):F2}%)");
            Console.WriteLine("说明：该实验只替换“行星生成链路”为更 FP32 的变体，恒星点位仍按原版生成；mismatch 表示结构签名不同。");

            if (mismatch > 0)
            {
                Console.WriteLine();
                Console.WriteLine("第一处差异字段分布（Top 10）：");
                foreach (var kv in firstDiffKindCount.OrderByDescending(k => k.Value).ThenBy(k => k.Key).Take(10))
                    Console.WriteLine($"  {kv.Key}: {kv.Value}");

                Console.WriteLine("第一处差异 Star 索引分布（Top 10）：");
                foreach (var kv in firstDiffStarIndexCount.OrderByDescending(k => k.Value).ThenBy(k => k.Key).Take(10))
                    Console.WriteLine($"  Star[{kv.Key}]: {kv.Value}");
            }
        }

        private static void CompareVeinsF32(int startSeed, int starCount, int count, int showMismatches)
        {
            if (count < 1) count = 1;

            int seedMismatch = 0;
            int seedTotal = 0;
            int shown = 0;

            int planetMismatch = 0;
            int planetTotal = 0;

            for (int s = startSeed; s < startSeed + count; s++)
            {
                var gd = new global::DspFindSeed.GameDesc();
                gd.SetForNewGame(global::DspFindSeed.UniverseGen.algoVersion, s, starCount, 1, 1f);

                var galaxy = global::DspFindSeed.UniverseGen.CreateGalaxy(gd);
                bool seedDiff = false;
                string firstDiff = null;

                if (galaxy != null && galaxy.stars != null)
                {
                    for (int si = 0; si < galaxy.stars.Length && !seedDiff; si++)
                    {
                        var star = galaxy.stars[si];
                        if (star?.planets == null) continue;

                        for (int pi = 0; pi < star.planets.Length; pi++)
                        {
                            var p = star.planets[pi];
                            if (p == null || p.type == EPlanetType.Gas) continue;
                            planetTotal++;

                            var c64 = global::DspFindSeed.PlanetModelingManager.RefreshPlanetData(p);
                            var c32 = global::DspFindSeed.PlanetModelingManager.RefreshPlanetData_F32(p);
                            if (c64 == null || c32 == null)
                                continue;

                            int len = Math.Min(c64.Length, c32.Length);
                            int diffVeinId = -1, a = 0, b = 0;
                            for (int vid = 1; vid < len; vid++)
                            {
                                if (c64[vid] != c32[vid])
                                {
                                    diffVeinId = vid;
                                    a = c64[vid];
                                    b = c32[vid];
                                    break;
                                }
                            }

                            if (diffVeinId >= 0)
                            {
                                planetMismatch++;
                                seedDiff = true;

                                string veinName = null;
                                var vp = global::DspFindSeed.LDB.veins.Select(diffVeinId);
                                if (vp != null) veinName = vp.Name;
                                if (string.IsNullOrEmpty(veinName)) veinName = $"veinId={diffVeinId}";

                                firstDiff = $"Star[{si}].Planet[{pi}] id={p.id} theme={p.theme} type={p.type} veinSpot[{veinName}] FP64={a} FP32={b}";
                                break;
                            }
                        }
                    }
                }

                seedTotal++;
                if (seedDiff)
                {
                    seedMismatch++;
                    if (showMismatches > 0 && shown < showMismatches)
                    {
                        shown++;
                        Console.WriteLine();
                        Console.WriteLine($"--- mismatch #{shown} seed={s} stars={starCount} ---");
                        Console.WriteLine(firstDiff ?? "（未找到具体差异，但判定为 mismatch）");
                    }
                }
            }

            Console.WriteLine($"compare-veins-f32 startSeed={startSeed} stars={starCount} count={count}");
            Console.WriteLine($"seedMismatch={seedMismatch}/{seedTotal} ({(seedTotal > 0 ? (seedMismatch * 100.0 / seedTotal) : 0):F6}%)");
            Console.WriteLine($"planetMismatch={planetMismatch}/{planetTotal} ({(planetTotal > 0 ? (planetMismatch * 100.0 / planetTotal) : 0):F6}%)");
            Console.WriteLine("说明：FP64=严格版（NextDouble 与阈值比较）；FP32=将 NextDouble 截断成 float 后再比较。该实验仅用于评估差异比例。");
        }

        private static void ComparePipelineF32(int startSeed, int starCount, int count, int showMismatches)
        {
            if (count < 1) count = 1;

            int mismatchGalaxy = 0;
            int mismatchPlanets = 0;
            int mismatchVeins = 0;
            int mismatchPipeline = 0;
            int total = 0;
            int shown = 0;

            // 第一处差异统计
            var firstDiffKindCount = new Dictionary<string, int>(StringComparer.Ordinal);

            for (int s = startSeed; s < startSeed + count; s++)
            {
                var gd = new global::DspFindSeed.GameDesc();
                gd.SetForNewGame(global::DspFindSeed.UniverseGen.algoVersion, s, starCount, 1, 1f);

                var g64 = global::DspFindSeed.UniverseGen.CreateGalaxy(gd);
                var g32 = UniverseGenPipelineF32.CreateGalaxy(gd);

                ulong hg64 = SignatureGalaxyOnly(g64);
                ulong hg32 = SignatureGalaxyOnly(g32);
                if (hg64 != hg32) mismatchGalaxy++;

                ulong hp64 = Signature(g64);
                ulong hp32 = Signature(g32);
                if (hp64 != hp32) mismatchPlanets++;

                ulong hv64 = SignatureVeinsOnly(g64, useFp32Veins: false);
                ulong hv32 = SignatureVeinsOnly(g32, useFp32Veins: true);
                if (hv64 != hv32) mismatchVeins++;

                ulong hall64 = SignaturePipeline(g64, useFp32Veins: false);
                ulong hall32 = SignaturePipeline(g32, useFp32Veins: true);
                if (hall64 != hall32) mismatchPipeline++;

                total++;
                if (hall64 != hall32)
                {
                    string diff = DescribeFirstDifferenceWithVeins(g64, g32);
                    string kind = ExtractKind(diff);
                    if (!string.IsNullOrEmpty(kind))
                    {
                        if (!firstDiffKindCount.TryGetValue(kind, out var c)) c = 0;
                        firstDiffKindCount[kind] = c + 1;
                    }

                    if (showMismatches > 0 && shown < showMismatches)
                    {
                        shown++;
                        Console.WriteLine();
                        Console.WriteLine($"--- mismatch #{shown} seed={s} stars={starCount} ---");
                        Console.WriteLine($"galaxySig64=0x{hg64:X16} galaxySig32=0x{hg32:X16}");
                        Console.WriteLine($"planetSig64=0x{hp64:X16} planetSig32=0x{hp32:X16}");
                        Console.WriteLine($"veinSig64=0x{hv64:X16} veinSig32=0x{hv32:X16}");
                        Console.WriteLine($"pipeSig64=0x{hall64:X16} pipeSig32=0x{hall32:X16}");
                        Console.WriteLine(diff);
                    }
                }
            }

            Console.WriteLine($"compare-pipeline-f32 startSeed={startSeed} stars={starCount} count={count}");
            Console.WriteLine($"galaxyMismatch={mismatchGalaxy}/{total} ({(total > 0 ? (mismatchGalaxy * 100.0 / total) : 0):F6}%)");
            Console.WriteLine($"planetMismatch={mismatchPlanets}/{total} ({(total > 0 ? (mismatchPlanets * 100.0 / total) : 0):F6}%)");
            Console.WriteLine($"veinMismatch={mismatchVeins}/{total} ({(total > 0 ? (mismatchVeins * 100.0 / total) : 0):F6}%)");
            Console.WriteLine($"pipelineMismatch={mismatchPipeline}/{total} ({(total > 0 ? (mismatchPipeline * 100.0 / total) : 0):F6}%)");
            Console.WriteLine("说明：FP64=UniverseGen(原版) + RefreshPlanetData(严格)；FP32=UniverseGenF32(星系点位) + PlanetGenF32(行星) + RefreshPlanetData_F32(矿堆数概率比较也 FP32)。");

            if (mismatchPipeline > 0)
            {
                Console.WriteLine();
                Console.WriteLine("第一处差异分布（Top 10）：");
                foreach (var kv in firstDiffKindCount.OrderByDescending(k => k.Value).ThenBy(k => k.Key).Take(10))
                    Console.WriteLine($"  {kv.Key}: {kv.Value}");
            }
        }

        private static void ComparePipelineMix(int startSeed, int starCount, int count, int showMismatches, bool useFp32Veins)
        {
            if (count < 1) count = 1;

            int mismatchGalaxy = 0;
            int mismatchPlanets = 0;
            int mismatchVeins = 0;
            int mismatchPipeline = 0;
            int total = 0;
            int shown = 0;

            for (int s = startSeed; s < startSeed + count; s++)
            {
                var gd = new global::DspFindSeed.GameDesc();
                gd.SetForNewGame(global::DspFindSeed.UniverseGen.algoVersion, s, starCount, 1, 1f);

                var g64 = global::DspFindSeed.UniverseGen.CreateGalaxy(gd);
                var gMix = UniverseGenPipelineMix.CreateGalaxy(gd);

                ulong hg64 = SignatureGalaxyOnly(g64);
                ulong hg32 = SignatureGalaxyOnly(gMix);
                if (hg64 != hg32) mismatchGalaxy++;

                ulong hp64 = Signature(g64);
                ulong hp32 = Signature(gMix);
                if (hp64 != hp32) mismatchPlanets++;

                ulong hv64 = SignatureVeinsOnly(g64, useFp32Veins: false);
                ulong hv32 = SignatureVeinsOnly(gMix, useFp32Veins: useFp32Veins);
                if (hv64 != hv32) mismatchVeins++;

                ulong hall64 = SignaturePipeline(g64, useFp32Veins: false);
                ulong hall32 = SignaturePipeline(gMix, useFp32Veins: useFp32Veins);
                if (hall64 != hall32) mismatchPipeline++;

                total++;
                if (showMismatches > 0 && shown < showMismatches && hall64 != hall32)
                {
                    shown++;
                    Console.WriteLine();
                    Console.WriteLine($"--- mismatch #{shown} seed={s} stars={starCount} ---");
                    Console.WriteLine($"galaxySig64=0x{hg64:X16} galaxySigMix=0x{hg32:X16}");
                    Console.WriteLine($"planetSig64=0x{hp64:X16} planetSigMix=0x{hp32:X16}");
                    Console.WriteLine($"veinSig64=0x{hv64:X16} veinSigMix=0x{hv32:X16}");
                    Console.WriteLine($"pipeSig64=0x{hall64:X16} pipeSigMix=0x{hall32:X16}");
                    Console.WriteLine(DescribeFirstDifferenceWithVeins(g64, gMix));
                }
            }

            var label = useFp32Veins ? "compare-pipeline-mix-veins-f32" : "compare-pipeline-mix";
            Console.WriteLine($"{label} startSeed={startSeed} stars={starCount} count={count}");
            Console.WriteLine($"galaxyMismatch={mismatchGalaxy}/{total} ({(total > 0 ? (mismatchGalaxy * 100.0 / total) : 0):F6}%)");
            Console.WriteLine($"planetMismatch={mismatchPlanets}/{total} ({(total > 0 ? (mismatchPlanets * 100.0 / total) : 0):F6}%)");
            Console.WriteLine($"veinMismatch={mismatchVeins}/{total} ({(total > 0 ? (mismatchVeins * 100.0 / total) : 0):F6}%)");
            Console.WriteLine($"pipelineMismatch={mismatchPipeline}/{total} ({(total > 0 ? (mismatchPipeline * 100.0 / total) : 0):F6}%)");
            Console.WriteLine("说明：Mix=星系点位使用 pt/rand/params 回到 FP64 的变体（你之前验证过 0 mismatch），行星用 PlanetGenF32，矿堆数可选严格或 FP32。");
        }

        private static string ExtractKind(string diff)
        {
            if (string.IsNullOrEmpty(diff)) return null;
            // 约定：DescribeFirstDifference* 的第一段会包含类似 "galaxy.xxx" / "star.xxx" / "planet.xxx" / "veinSpot.xxx"
            int sp = diff.IndexOf(' ');
            if (sp <= 0) return diff;
            return diff.Substring(0, sp);
        }

        private static ulong SignaturePipeline(GalaxyData g, bool useFp32Veins)
        {
            unchecked
            {
                const ulong FNV_OFFSET = 14695981039346656037UL;
                const ulong FNV_PRIME = 1099511628211UL;
                ulong h = FNV_OFFSET;

                void Mix(int v)
                {
                    h ^= (uint)v;
                    h *= FNV_PRIME;
                }

                if (g == null || g.stars == null)
                    return 0;

                Mix(g.starCount);
                Mix(g.birthStarId);
                Mix(g.birthPlanetId);

                for (int i = 0; i < g.stars.Length; i++)
                {
                    var star = g.stars[i];
                    if (star == null) { Mix(-1); continue; }
                    Mix(star.id);
                    Mix((int)star.type);
                    Mix((int)star.spectr);
                    Mix(star.planetCount);

                    Mix((int)Math.Round(star.uPosition.x * 0.001));
                    Mix((int)Math.Round(star.uPosition.y * 0.001));
                    Mix((int)Math.Round(star.uPosition.z * 0.001));

                    if (star.planets == null) continue;
                    for (int j = 0; j < star.planets.Length; j++)
                    {
                        var p = star.planets[j];
                        if (p == null) { Mix(-2); continue; }
                        Mix((int)p.type);
                        Mix(p.theme);
                        Mix(p.waterItemId);
                        Mix(p.orbitIndex);
                        Mix(p.orbitAround);

                        if (p.type == EPlanetType.Gas) continue;
                        var counts = useFp32Veins
                            ? global::DspFindSeed.PlanetModelingManager.RefreshPlanetData_F32(p)
                            : global::DspFindSeed.PlanetModelingManager.RefreshPlanetData(p);
                        if (counts == null) { Mix(-3); continue; }
                        // 只混入前 32 个 id（已覆盖 DSP 常见矿种，且更快）
                        int max = Math.Min(counts.Length, 32);
                        for (int vid = 1; vid < max; vid++)
                            Mix(counts[vid]);
                    }
                }

                return h;
            }
        }

        private static ulong SignatureGalaxyOnly(GalaxyData g)
        {
            unchecked
            {
                const ulong FNV_OFFSET = 14695981039346656037UL;
                const ulong FNV_PRIME = 1099511628211UL;
                ulong h = FNV_OFFSET;
                void Mix(int v) { h ^= (uint)v; h *= FNV_PRIME; }

                if (g == null || g.stars == null) return 0;
                Mix(g.starCount);
                for (int i = 0; i < g.stars.Length; i++)
                {
                    var star = g.stars[i];
                    if (star == null) { Mix(-1); continue; }
                    Mix(star.id);
                    Mix((int)star.type);
                    Mix((int)star.spectr);
                    Mix(star.planetCount);
                    Mix((int)Math.Round(star.uPosition.x * 0.001));
                    Mix((int)Math.Round(star.uPosition.y * 0.001));
                    Mix((int)Math.Round(star.uPosition.z * 0.001));
                }
                return h;
            }
        }

        private static ulong SignatureVeinsOnly(GalaxyData g, bool useFp32Veins)
        {
            unchecked
            {
                const ulong FNV_OFFSET = 14695981039346656037UL;
                const ulong FNV_PRIME = 1099511628211UL;
                ulong h = FNV_OFFSET;
                void Mix(int v) { h ^= (uint)v; h *= FNV_PRIME; }

                if (g == null || g.stars == null) return 0;
                Mix(g.starCount);
                for (int si = 0; si < g.stars.Length; si++)
                {
                    var star = g.stars[si];
                    if (star?.planets == null) { Mix(-1); continue; }
                    Mix(star.id);
                    for (int pi = 0; pi < star.planets.Length; pi++)
                    {
                        var p = star.planets[pi];
                        if (p == null) { Mix(-2); continue; }
                        Mix(p.id);
                        if (p.type == EPlanetType.Gas) { Mix(0); continue; }
                        var counts = useFp32Veins
                            ? global::DspFindSeed.PlanetModelingManager.RefreshPlanetData_F32(p)
                            : global::DspFindSeed.PlanetModelingManager.RefreshPlanetData(p);
                        if (counts == null) { Mix(-3); continue; }
                        int max = Math.Min(counts.Length, 32);
                        for (int vid = 1; vid < max; vid++)
                            Mix(counts[vid]);
                    }
                }
                return h;
            }
        }

        private static string DescribeFirstDifferenceWithVeins(GalaxyData g64, GalaxyData g32)
        {
            // 先复用现有的“星系/行星”第一处差异描述
            string baseDiff = DescribeFirstDifference(g64, g32);
            if (!string.IsNullOrEmpty(baseDiff) && !baseDiff.Contains("未找到差异"))
                return baseDiff;

            // 如果星系/行星字段都一致，再比较矿堆数
            if (g64 == null || g32 == null || g64.stars == null || g32.stars == null)
                return "galaxy.null";
            int sc = Math.Min(g64.stars.Length, g32.stars.Length);
            for (int si = 0; si < sc; si++)
            {
                var s64 = g64.stars[si];
                var s32 = g32.stars[si];
                if (s64?.planets == null || s32?.planets == null) continue;
                int pc = Math.Min(s64.planets.Length, s32.planets.Length);
                for (int pi = 0; pi < pc; pi++)
                {
                    var p64 = s64.planets[pi];
                    var p32 = s32.planets[pi];
                    if (p64 == null || p32 == null) continue;
                    if (p64.type == EPlanetType.Gas || p32.type == EPlanetType.Gas) continue;

                    var c64 = global::DspFindSeed.PlanetModelingManager.RefreshPlanetData(p64);
                    var c32 = global::DspFindSeed.PlanetModelingManager.RefreshPlanetData_F32(p32);
                    if (c64 == null || c32 == null) continue;

                    int max = Math.Min(Math.Min(c64.Length, c32.Length), 32);
                    for (int vid = 1; vid < max; vid++)
                    {
                        if (c64[vid] != c32[vid])
                            return $"veinSpot.diff Star[{si}].Planet[{pi}] id64={p64.id} id32={p32.id} veinId={vid} FP64={c64[vid]} FP32={c32[vid]}";
                    }
                }
            }

            return "unknown";
        }

        private struct FirstDiff
        {
            public string Kind;
            public int StarIndex;
        }

        private static FirstDiff FirstDifference(GalaxyData g64, GalaxyData g32)
        {
            // 与 DescribeFirstDifference 一致，但返回结构化信息用于统计
            if (g64 == null || g32 == null)
                return new FirstDiff { Kind = "galaxy.null", StarIndex = -1 };

            if (g64.starCount != g32.starCount)
                return new FirstDiff { Kind = "galaxy.starCount", StarIndex = -1 };
            if (g64.birthStarId != g32.birthStarId)
                return new FirstDiff { Kind = "galaxy.birthStarId", StarIndex = -1 };
            if (g64.birthPlanetId != g32.birthPlanetId)
                return new FirstDiff { Kind = "galaxy.birthPlanetId", StarIndex = -1 };

            if (g64.stars == null || g32.stars == null || g64.stars.Length != g32.stars.Length)
                return new FirstDiff { Kind = "galaxy.stars", StarIndex = -1 };

            for (int i = 0; i < g64.stars.Length; i++)
            {
                var s64 = g64.stars[i];
                var s32 = g32.stars[i];
                if (s64 == null || s32 == null)
                    return new FirstDiff { Kind = "star.null", StarIndex = i };
                if (s64.id != s32.id)
                    return new FirstDiff { Kind = "star.id", StarIndex = i };
                if (s64.type != s32.type)
                    return new FirstDiff { Kind = "star.type", StarIndex = i };
                if (s64.spectr != s32.spectr)
                    return new FirstDiff { Kind = "star.spectr", StarIndex = i };
                if (s64.planetCount != s32.planetCount)
                    return new FirstDiff { Kind = "star.planetCount", StarIndex = i };

                var x64 = (int)Math.Round(s64.uPosition.x * 0.001);
                var y64 = (int)Math.Round(s64.uPosition.y * 0.001);
                var z64 = (int)Math.Round(s64.uPosition.z * 0.001);
                var x32 = (int)Math.Round(s32.uPosition.x * 0.001);
                var y32 = (int)Math.Round(s32.uPosition.y * 0.001);
                var z32 = (int)Math.Round(s32.uPosition.z * 0.001);
                if (x64 != x32 || y64 != y32 || z64 != z32)
                    return new FirstDiff { Kind = "star.uPosition.q1e-3", StarIndex = i };

                var p64 = s64.planets;
                var p32 = s32.planets;
                if (p64 == null || p32 == null || p64.Length != p32.Length)
                    return new FirstDiff { Kind = "star.planets", StarIndex = i };
                for (int j = 0; j < p64.Length; j++)
                {
                    var a = p64[j];
                    var b = p32[j];
                    if (a == null || b == null)
                        return new FirstDiff { Kind = "planet.null", StarIndex = i };
                    if (a.type != b.type)
                        return new FirstDiff { Kind = "planet.type", StarIndex = i };
                    if (a.theme != b.theme)
                        return new FirstDiff { Kind = "planet.theme", StarIndex = i };
                    if (a.waterItemId != b.waterItemId)
                        return new FirstDiff { Kind = "planet.waterItemId", StarIndex = i };
                    if (a.orbitIndex != b.orbitIndex)
                        return new FirstDiff { Kind = "planet.orbitIndex", StarIndex = i };
                    if (a.orbitAround != b.orbitAround)
                        return new FirstDiff { Kind = "planet.orbitAround", StarIndex = i };
                }
            }

            return new FirstDiff { Kind = "unknown", StarIndex = -1 };
        }

        private static string DescribeFirstDifference(GalaxyData g64, GalaxyData g32)
        {
            // 只输出“第一处差异”，避免日志爆炸
            if (g64 == null && g32 == null) return "(both null)";
            if (g64 == null) return "FP64 galaxy=null, FP32 galaxy!=null";
            if (g32 == null) return "FP64 galaxy!=null, FP32 galaxy=null";

            if (g64.starCount != g32.starCount)
                return $"galaxy.starCount 不同：FP64={g64.starCount} FP32={g32.starCount}";
            if (g64.birthStarId != g32.birthStarId)
                return $"galaxy.birthStarId 不同：FP64={g64.birthStarId} FP32={g32.birthStarId}";
            if (g64.birthPlanetId != g32.birthPlanetId)
                return $"galaxy.birthPlanetId 不同：FP64={g64.birthPlanetId} FP32={g32.birthPlanetId}";

            if (g64.stars == null && g32.stars == null) return "(both stars null)";
            if (g64.stars == null) return "FP64 stars=null, FP32 stars!=null";
            if (g32.stars == null) return "FP64 stars!=null, FP32 stars=null";
            if (g64.stars.Length != g32.stars.Length)
                return $"stars.Length 不同：FP64={g64.stars.Length} FP32={g32.stars.Length}";

            for (int i = 0; i < g64.stars.Length; i++)
            {
                var s64 = g64.stars[i];
                var s32 = g32.stars[i];

                if (s64 == null && s32 == null) continue;
                if (s64 == null) return $"Star[{i}] FP64=null FP32!=null";
                if (s32 == null) return $"Star[{i}] FP64!=null FP32=null";

                if (s64.id != s32.id)
                    return $"Star[{i}].id 不同：FP64={s64.id} FP32={s32.id}";
                if (s64.type != s32.type)
                    return $"Star[{i}].type 不同：FP64={s64.type} FP32={s32.type}";
                if (s64.spectr != s32.spectr)
                    return $"Star[{i}].spectr 不同：FP64={s64.spectr} FP32={s32.spectr}";
                if (s64.planetCount != s32.planetCount)
                    return $"Star[{i}].planetCount 不同：FP64={s64.planetCount} FP32={s32.planetCount}";

                // 位置量化（与签名一致），更容易阅读
                var x64 = (int)Math.Round(s64.uPosition.x * 0.001);
                var y64 = (int)Math.Round(s64.uPosition.y * 0.001);
                var z64 = (int)Math.Round(s64.uPosition.z * 0.001);
                var x32 = (int)Math.Round(s32.uPosition.x * 0.001);
                var y32 = (int)Math.Round(s32.uPosition.y * 0.001);
                var z32 = (int)Math.Round(s32.uPosition.z * 0.001);
                if (x64 != x32 || y64 != y32 || z64 != z32)
                    return $"Star[{i}].uPosition(×1e-3量化) 不同：FP64=({x64},{y64},{z64}) FP32=({x32},{y32},{z32})";

                var p64 = s64.planets;
                var p32 = s32.planets;
                if (p64 == null && p32 == null) continue;
                if (p64 == null) return $"Star[{i}].planets FP64=null FP32!=null";
                if (p32 == null) return $"Star[{i}].planets FP64!=null FP32=null";
                if (p64.Length != p32.Length)
                    return $"Star[{i}].planets.Length 不同：FP64={p64.Length} FP32={p32.Length}";

                for (int j = 0; j < p64.Length; j++)
                {
                    var a = p64[j];
                    var b = p32[j];
                    if (a == null && b == null) continue;
                    if (a == null) return $"Star[{i}].Planet[{j}] FP64=null FP32!=null";
                    if (b == null) return $"Star[{i}].Planet[{j}] FP64!=null FP32=null";

                    if (a.type != b.type)
                        return $"Star[{i}].Planet[{j}].type 不同：FP64={a.type} FP32={b.type}";
                    if (a.theme != b.theme)
                        return $"Star[{i}].Planet[{j}].theme 不同：FP64={a.theme} FP32={b.theme}";
                    if (a.waterItemId != b.waterItemId)
                        return $"Star[{i}].Planet[{j}].waterItemId 不同：FP64={a.waterItemId} FP32={b.waterItemId}";
                    if (a.orbitIndex != b.orbitIndex)
                        return $"Star[{i}].Planet[{j}].orbitIndex 不同：FP64={a.orbitIndex} FP32={b.orbitIndex}";
                    if (a.orbitAround != b.orbitAround)
                        return $"Star[{i}].Planet[{j}].orbitAround 不同：FP64={a.orbitAround} FP32={b.orbitAround}";
                }
            }

            return "（未找到差异：可能是签名未覆盖的字段变化）";
        }

        private static ulong Signature(GalaxyData g)
        {
            // 简单稳定签名：不追求加密安全，只要对结构变化敏感即可
            unchecked
            {
                const ulong FNV_OFFSET = 14695981039346656037UL;
                const ulong FNV_PRIME = 1099511628211UL;
                ulong h = FNV_OFFSET;

                void Mix(int v)
                {
                    h ^= (uint)v;
                    h *= FNV_PRIME;
                }

                if (g == null || g.stars == null)
                    return 0;

                Mix(g.starCount);
                Mix(g.birthStarId);
                Mix(g.birthPlanetId);

                for (int i = 0; i < g.stars.Length; i++)
                {
                    var star = g.stars[i];
                    if (star == null) { Mix(-1); continue; }
                    Mix(star.id);
                    Mix((int)star.type);
                    Mix((int)star.spectr);
                    Mix(star.planetCount);

                    // 量化位置到 1e-3，避免浮点微小噪声导致签名全变
                    Mix((int)Math.Round(star.uPosition.x * 0.001));
                    Mix((int)Math.Round(star.uPosition.y * 0.001));
                    Mix((int)Math.Round(star.uPosition.z * 0.001));

                    if (star.planets == null) continue;
                    for (int j = 0; j < star.planets.Length; j++)
                    {
                        var p = star.planets[j];
                        if (p == null) { Mix(-2); continue; }
                        Mix((int)p.type);
                        Mix(p.theme);
                        Mix(p.waterItemId);
                        Mix(p.orbitIndex);
                        Mix(p.orbitAround);
                    }
                }

                return h;
            }
        }

        private static void DumpSeed(int seed, int starCount)
        {
            var algoVersion = global::DspFindSeed.UniverseGen.algoVersion;
            var gd = new global::DspFindSeed.GameDesc();
            gd.SetForNewGame(algoVersion, seed, starCount, 1, 1f);
            var galaxy = global::DspFindSeed.UniverseGen.CreateGalaxy(gd);
            if (galaxy == null || galaxy.stars == null)
                throw new InvalidOperationException("CreateGalaxy 返回空。");

            Console.WriteLine($"seed={seed} stars={starCount} algo={algoVersion}");
            Console.WriteLine($"galaxy.starCount={galaxy.starCount} birthStarId={galaxy.birthStarId} birthPlanetId={galaxy.birthPlanetId}");
            Console.WriteLine();

            for (int si = 0; si < galaxy.stars.Length; si++)
            {
                var star = galaxy.stars[si];
                if (star == null) continue;

                Console.WriteLine($"[Star {si}] id={star.id} name={star.displayName ?? star.name} type={star.type} spectr={star.spectr} lumino={star.dysonLumino.ToString("F4", CultureInfo.InvariantCulture)} planets={star.planetCount}");

                if (star.planets == null)
                {
                    Console.WriteLine("  (no planets)");
                    continue;
                }

                for (int pi = 0; pi < star.planets.Length; pi++)
                {
                    var p = star.planets[pi];
                    if (p == null) continue;

                    var theme = global::DspFindSeed.LDB.themes.Select(p.theme);
                    string themeName = theme != null ? theme.DisplayName : $"themeId={p.theme}";

                    Console.WriteLine($"  [Planet {pi}] id={p.id} name={p.displayName ?? p.name} type={p.type} theme={themeName} orbitIndex={p.orbitIndex} orbitAround={p.orbitAround} singularity={p.singularity} waterItemId={p.waterItemId}");

                    if (p.type == EPlanetType.Gas && p.gasSpeeds != null)
                    {
                        // DSP 里通常 gasSpeeds[1] 是重氢速率（你的 UI 就取 [1]）
                        var max = p.gasSpeeds.Max();
                        Console.WriteLine($"    gasSpeeds=[{string.Join(", ", p.gasSpeeds.Select(x => x.ToString("F4", CultureInfo.InvariantCulture)))}] max={max.ToString("F4", CultureInfo.InvariantCulture)}");
                    }

                    var counts = global::DspFindSeed.PlanetModelingManager.RefreshPlanetData(p);
                    if (counts != null)
                    {
                        // counts 的 index 对齐 VeinProto.ID（0 号不用）
                        var veins = global::DspFindSeed.LDB.veins;
                        int shown = 0;
                        for (int vid = 1; vid < Math.Min(counts.Length, veins.dataArray.Length); vid++)
                        {
                            int c = counts[vid];
                            if (c <= 0) continue;
                            var vp = veins.Select(vid);
                            var name = vp != null ? vp.Name : $"veinId={vid}";
                            Console.WriteLine($"    veinSpot {name} (id={vid}) = {c}");
                            shown++;
                        }
                        if (shown == 0)
                            Console.WriteLine("    veinSpot: (none)");
                    }
                }

                Console.WriteLine();
            }
        }

        private static int GetIntArg(string[] args, string key, int defaultValue)
        {
            for (int i = 0; i < args.Length; i++)
            {
                if (!string.Equals(args[i], key, StringComparison.OrdinalIgnoreCase))
                    continue;
                if (i + 1 >= args.Length)
                    return defaultValue;
                if (int.TryParse(args[i + 1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int v))
                    return v;
                return defaultValue;
            }
            return defaultValue;
        }

        private static bool HasFlag(string[] args, string flag)
        {
            for (int i = 0; i < args.Length; i++)
            {
                if (string.Equals(args[i], flag, StringComparison.OrdinalIgnoreCase))
                    return true;
            }
            return false;
        }
    }
}
