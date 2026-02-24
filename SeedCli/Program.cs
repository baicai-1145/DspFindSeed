using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

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
            bool dumpCompareMix = HasFlag(args, "--dump-compare-mix");
            bool dumpCompareMixVeinsF32 = HasFlag(args, "--dump-compare-mix-veins-f32");
            bool debugVeinBranch = HasFlag(args, "--debug-vein-branch");
            bool mixCollisionFp64 = HasFlag(args, "--mix-collision-fp64");
            bool mixSpeedOnly = HasFlag(args, "--mix-speed-only");
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
            bool useCudaPlanet = HasFlag(args, "--use-cuda-planet");
            bool useCudaPlanetCore = HasFlag(args, "--use-cuda-planet-core");
            int batchSize = GetIntArg(args, "--batch-size", 1024);
            bool hasLegacySeedBatchArg = HasFlag(args, "--seed-batch-size");
            bool hasMixThreadsArg = HasFlag(args, "--mix-threads");
            int legacySeedBatchSize = GetIntArg(args, "--seed-batch-size", 0);
            int batchPerThread = GetIntArg(args, "--batch-per-thread", legacySeedBatchSize > 0 ? legacySeedBatchSize : 1);
            int showMismatches = GetIntArg(args, "--show-mismatches", 0);
            int threads = GetIntArg(args, "--threads", 1);
            int cpuThreads = GetIntArg(args, "--cpu-threads", threads);
            int gpuChunkSeeds = GetIntArg(args, "--gpu-chunk-seeds", 0);
            int gpuStreams = GetIntArg(args, "--gpu-streams", 1);
            int mixCoreGroupSeeds = GetIntArg(args, "--mix-core-group-seeds", 0);
            int planetId = GetIntArg(args, "--planet-id", 0);
            string cpuCacheFile = GetStringArg(args, "--cpu-cache-file", null);
            bool noCpuCache = HasFlag(args, "--no-cpu-cache");
            bool timingDebug = HasFlag(args, "--timing-debug");

            if (seed <= 0)
            {
                Console.WriteLine("用法：SeedCli.exe --seed <种子ID> [--stars <星区数量>]");
                Console.WriteLine("说明：会优先从 Release/Prototypes 或 Prototypes 读取 *ProtoSet.json，再回退 xml。");
                Console.WriteLine("对比实验：SeedCli.exe --seed <起始种子> --stars <星区数量> --count <个数> --compare-f32");
                Console.WriteLine("行星 FP32 实验：SeedCli.exe --seed <起始种子> --stars <星区数量> --count <个数> --compare-planets-f32");
                Console.WriteLine("矿堆数 FP32 实验：SeedCli.exe --seed <起始种子> --stars <星区数量> --count <个数> --compare-veins-f32");
                Console.WriteLine("端到端 FP32 实验：SeedCli.exe --seed <起始种子> --stars <星区数量> --count <个数> --compare-pipeline-f32");
                Console.WriteLine("端到端 Mix 实验（星系 0 mismatch 变体）：SeedCli.exe --seed <起始种子> --stars <星区数量> --count <个数> --compare-pipeline-mix [--cpu-threads 10] [--gpu-chunk-seeds 4096]");
                Console.WriteLine("端到端 Mix+矿 FP32 实验：SeedCli.exe --seed <起始种子> --stars <星区数量> --count <个数> --compare-pipeline-mix-veins-f32 [--cpu-threads 10] [--gpu-chunk-seeds 4096]");
                Console.WriteLine("Mix 纯测速（不做 FP64 对比）：追加 --mix-speed-only（建议配合 --compare-pipeline-mix-veins-f32）。");
                Console.WriteLine("CPU 线程数：追加 --cpu-threads <N>（兼容 --threads）");
                Console.WriteLine("GPU 真批大小：追加 --gpu-chunk-seeds <N>");
                Console.WriteLine("Mix 行星 core 分组大小：追加 --mix-core-group-seeds <N>（默认 0=自动按 chunk 大小）");
                Console.WriteLine("GPU 流数量：追加 --gpu-streams <N>（用于 native seed 签名双缓冲/多流并发，默认 1）");
                Console.WriteLine("兼容别名：--batch-per-thread / --seed-batch-size 仅用于推导 gpu-chunk-seeds（当未显式设置 --gpu-chunk-seeds）");
                Console.WriteLine("耗时统计：追加 --timing-debug（打印各阶段耗时与 native 调用统计）");
                Console.WriteLine("CPU FP64 缓存文件：追加 --cpu-cache-file <路径>（默认 logs/cpu_fp64_cache 自动命名）");
                Console.WriteLine("禁用 CPU FP64 缓存：追加 --no-cpu-cache");
                Console.WriteLine("Mix 路线强制星系碰撞 FP64：追加 --mix-collision-fp64");
                Console.WriteLine("单行星矿脉概率分支定位：SeedCli.exe --seed <种子ID> --stars <星区数量> --planet-id <行星ID> --debug-vein-branch");
                Console.WriteLine("单种子详细对照（FP64 vs Mix 严格矿）：SeedCli.exe --seed <种子ID> --stars <星区数量> --dump-compare-mix");
                Console.WriteLine("单种子详细对照（FP64 vs Mix+矿FP32）：SeedCli.exe --seed <种子ID> --stars <星区数量> --dump-compare-mix-veins-f32");
                Console.WriteLine("启用 CUDA 星系点位（仅 ParamsFp64 路线）：追加 --use-cuda-galaxy（或环境变量 DSP_USE_CUDA_GALAXY=1）");
                Console.WriteLine("启用 CUDA 行星矿堆数批量统计：追加 --use-cuda-planet（或环境变量 DSP_USE_CUDA_PLANET=1）");
                Console.WriteLine("启用 CUDA 行星核心计算（PlanetGenF32 主体）：追加 --use-cuda-planet-core（或环境变量 DSP_USE_CUDA_PLANET_CORE=1）");
                Console.WriteLine("仅星系生成并行基准：SeedCli.exe --seed <起始种子> --stars <星区数量> --count <个数> --bench-galaxy-only [--batch-size 1024] [--collision-fp64]");
                Console.WriteLine("差异打印：SeedCli.exe --seed <起始种子> --stars <星区数量> --count <个数> --compare-f32 --show-mismatches <最多打印条数>");
                return 2;
            }

            try
            {
                if (benchGalaxyOnly)
                {
                    CudaGalaxyNative.EnableByCli(true);
                    CudaPlanetNative.EnableByCli(useCudaPlanet);
                    CudaPlanetNative.EnableCoreByCli(useCudaPlanetCore);
                    BenchGalaxyOnly(seed, stars, count, batchSize, collisionFp64);
                    return 0;
                }

                CudaGalaxyNative.EnableByCli(useCudaGalaxy);
                CudaPlanetNative.EnableByCli(useCudaPlanet);
                CudaPlanetNative.EnableCoreByCli(useCudaPlanetCore);
                InitForSearch();
                if (dumpCompareMix || dumpCompareMixVeinsF32)
                {
                    DumpSeedCompareMix(seed, stars, useFp32Veins: dumpCompareMixVeinsF32, mixCollisionFp64: mixCollisionFp64);
                }
                else if (debugVeinBranch)
                {
                    DebugVeinBranch(seed, stars, planetId);
                }
                else if (comparePipelineMix || comparePipelineMixVeinsF32)
                {
                    if (hasMixThreadsArg)
                        Console.WriteLine("参数 --mix-threads 已弃用：请使用 --cpu-threads + --gpu-chunk-seeds。");
                    if (hasLegacySeedBatchArg && !HasFlag(args, "--batch-per-thread"))
                        Console.WriteLine("参数 --seed-batch-size 已作为 --batch-per-thread 兼容别名处理。");
                    if (HasFlag(args, "--threads") && !HasFlag(args, "--cpu-threads"))
                        Console.WriteLine("参数 --threads 已兼容映射到 --cpu-threads。");
                    if ((HasFlag(args, "--batch-per-thread") || hasLegacySeedBatchArg) && !HasFlag(args, "--gpu-chunk-seeds"))
                        Console.WriteLine("参数 --batch-per-thread/--seed-batch-size 仅用于推导 --gpu-chunk-seeds（未显式设置时）。");
                    if (mixSpeedOnly && showMismatches > 0)
                        Console.WriteLine("参数 --mix-speed-only 已启用：--show-mismatches 将被忽略。");

                    ComparePipelineMix(
                        seed,
                        stars,
                        count,
                        showMismatches,
                        useFp32Veins: comparePipelineMixVeinsF32,
                        batchPerThread: batchPerThread,
                        mixCollisionFp64: mixCollisionFp64,
                        cpuThreads: cpuThreads,
                        gpuChunkSeeds: gpuChunkSeeds,
                        gpuStreams: gpuStreams,
                        mixCoreGroupSeeds: mixCoreGroupSeeds,
                        mixSpeedOnly: mixSpeedOnly,
                        timingDebug: timingDebug,
                        cpuCacheFile: cpuCacheFile,
                        useCpuCache: !noCpuCache);
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
            var galaxySeeds = new int[batchSize];
            var poseSeeds = new int[batchSize];
            var cpuSig = new ulong[batchSize];
            var gpuPoses = new CudaGalaxyNative.NativeVec3d[batchSize * maxCount];
            var gpuCounts = new int[batchSize];
            int outStride = maxCount;
            int endSeed = startSeed + count;

            for (int seedBase = startSeed; seedBase < endSeed; seedBase += batchSize)
            {
                int chunk = Math.Min(batchSize, endSeed - seedBase);

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
                    bool ok = CudaGalaxyNative.TryGenerateRandomPosesParamsFp64BatchInto(
                        poseSeeds,
                        chunk,
                        maxCount,
                        minDist: 2.0,
                        minStepLen: 2.3,
                        maxStepLen: 3.5,
                        flatten: 0.18,
                        collisionFp64: collisionFp64,
                        poses: gpuPoses,
                        outStride: outStride,
                        counts: gpuCounts);
                    long t1 = Stopwatch.GetTimestamp();
                    gpuTicks += (t1 - t0);

                    if (!ok)
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

        private static void ComparePipelineMix(
            int startSeed,
            int starCount,
            int count,
            int showMismatches,
            bool useFp32Veins,
            int batchPerThread,
            bool mixCollisionFp64,
            int cpuThreads,
            int gpuChunkSeeds,
            int gpuStreams,
            int mixCoreGroupSeeds,
            bool mixSpeedOnly,
            bool timingDebug,
            string cpuCacheFile,
            bool useCpuCache)
        {
            if (count < 1) count = 1;
            if (batchPerThread < 1) batchPerThread = 1;
            if (cpuThreads < 1) cpuThreads = 1;
            if (gpuStreams < 1) gpuStreams = 1;
            long derivedChunkLong = (long)cpuThreads * (long)batchPerThread;
            if (derivedChunkLong < 1) derivedChunkLong = 1;
            if (derivedChunkLong > int.MaxValue) derivedChunkLong = int.MaxValue;
            int seedBatchSize = gpuChunkSeeds > 0 ? gpuChunkSeeds : (int)derivedChunkLong;
            if (seedBatchSize < 1) seedBatchSize = 1;
            int mixRunChunkSeeds = seedBatchSize;
            int envMixRunChunkSeeds = GetIntEnv("DSP_MIX_RUN_CHUNK_SEEDS", 0);
            if (envMixRunChunkSeeds > 0 && mixRunChunkSeeds > envMixRunChunkSeeds)
                mixRunChunkSeeds = envMixRunChunkSeeds;
            if (mixRunChunkSeeds < 1)
                mixRunChunkSeeds = 1;
            bool autoCoreGroupSeeds = mixCoreGroupSeeds <= 0;

            int mismatchGalaxy = 0;
            int mismatchPlanets = 0;
            int mismatchVeins = 0;
            int mismatchPipeline = 0;
            int total = 0;
            int shown = 0;
            long tAllStart = 0;
            long tCacheLoad = 0;
            long tCpuGen = 0;
            long tCacheSave = 0;
            long tMixTotal = 0;
            long tMixPrefetch = 0;
            long tMixCreateGalaxy = 0;
            long tMixVeinBatch = 0;
            long tMixCompare = 0;
            long tMixDetail = 0;
            long tMixChunkSubmit = 0;
            int mixChunkCount = 0;
            long mixChunkSeedTotal = 0;

            if (timingDebug)
            {
                tAllStart = Stopwatch.GetTimestamp();
                CudaGalaxyNative.ResetPerfStats();
                CudaPlanetNative.ResetPerfStats();
            }

            CpuFp64SigRow[] cpuRows = mixSpeedOnly ? null : new CpuFp64SigRow[count];
            string resolvedCpuCacheFile = null;
            bool cpuCacheLoaded = false;
            bool cpuCacheSaved = false;

            if (!mixSpeedOnly && useCpuCache)
            {
                long t0 = timingDebug ? Stopwatch.GetTimestamp() : 0;
                resolvedCpuCacheFile = ResolveCpuFp64CacheFile(cpuCacheFile, startSeed, starCount, count);
                cpuCacheLoaded = TryLoadCpuFp64SigCache(resolvedCpuCacheFile, startSeed, starCount, count, cpuRows);
                if (timingDebug)
                    tCacheLoad += Stopwatch.GetTimestamp() - t0;
            }

            if (!mixSpeedOnly && !cpuCacheLoaded)
            {
                long tCpuStart = timingDebug ? Stopwatch.GetTimestamp() : 0;
                if (cpuThreads > 1)
                {
                    var options = new ParallelOptions { MaxDegreeOfParallelism = cpuThreads };
                    Parallel.For(0, count, options, idx =>
                    {
                        int seed = startSeed + idx;
                        var gd = new global::DspFindSeed.GameDesc();
                        gd.SetForNewGame(global::DspFindSeed.UniverseGen.algoVersion, seed, starCount, 1, 1f);
                        var g64 = global::DspFindSeed.UniverseGen.CreateGalaxy(gd);
                        cpuRows[idx] = new CpuFp64SigRow
                        {
                            Seed = seed,
                            GalaxySig = SignatureGalaxyOnly(g64),
                            PlanetSig = Signature(g64),
                            VeinSig = SignatureVeinsOnly(g64, useFp32Veins: false, allowCudaPlanetCounts: false),
                            PipelineSig = SignaturePipeline(g64, useFp32Veins: false, allowCudaPlanetCounts: false)
                        };
                    });
                }
                else
                {
                    for (int idx = 0; idx < count; ++idx)
                    {
                        int seed = startSeed + idx;
                        var gd = new global::DspFindSeed.GameDesc();
                        gd.SetForNewGame(global::DspFindSeed.UniverseGen.algoVersion, seed, starCount, 1, 1f);
                        var g64 = global::DspFindSeed.UniverseGen.CreateGalaxy(gd);
                        cpuRows[idx] = new CpuFp64SigRow
                        {
                            Seed = seed,
                            GalaxySig = SignatureGalaxyOnly(g64),
                            PlanetSig = Signature(g64),
                            VeinSig = SignatureVeinsOnly(g64, useFp32Veins: false, allowCudaPlanetCounts: false),
                            PipelineSig = SignaturePipeline(g64, useFp32Veins: false, allowCudaPlanetCounts: false)
                        };
                    }
                }

                if (useCpuCache && !string.IsNullOrEmpty(resolvedCpuCacheFile))
                {
                    long tSave = timingDebug ? Stopwatch.GetTimestamp() : 0;
                    cpuCacheSaved = TrySaveCpuFp64SigCache(resolvedCpuCacheFile, startSeed, starCount, count, cpuRows);
                    if (timingDebug)
                        tCacheSave += Stopwatch.GetTimestamp() - tSave;
                }
                if (timingDebug)
                    tCpuGen += Stopwatch.GetTimestamp() - tCpuStart;
            }

            int mixChunkTotal = 0;
            int mixChunkCudaCountsOk = 0;
            int mixChunkNativeSigOk = 0;
            int mixChunkNativeSeedSigOk = 0;
            int mixChunkObjectFallback = 0;
            int endSeed = startSeed + count;
            int mixThreads = cpuThreads;
            int envMixThreadsMax = GetIntEnv("DSP_MIX_THREADS_MAX", 8);
            if (envMixThreadsMax > 0 && mixThreads > envMixThreadsMax)
                mixThreads = envMixThreadsMax;
            if (mixThreads < 1)
                mixThreads = 1;
            if (autoCoreGroupSeeds)
            {
                int rangeSize = (mixRunChunkSeeds + mixThreads - 1) / mixThreads;
                mixCoreGroupSeeds = rangeSize <= 1536 ? rangeSize : 1024;
            }
            if (mixCoreGroupSeeds < 1)
                mixCoreGroupSeeds = 1;
            if (mixCoreGroupSeeds > seedBatchSize)
                mixCoreGroupSeeds = seedBatchSize;
            MixRuntimeFlags.ChunkWideCoreBatchSeedGroup = mixCoreGroupSeeds;

            var chunkBuffers = new GpuChunkBuffers();
            var chunkRunner = new GpuChunkRunner(starCount, mixCollisionFp64, mixThreads);
            int nativeInFlightLimit = 0;
            long tMixStart = (timingDebug || mixSpeedOnly) ? Stopwatch.GetTimestamp() : 0;
            bool prevSkipNameGeneration = global::DspFindSeed.StarGen.SkipNameGeneration;
            string prevNativeSpeedOnly = null;
            string prevNativePoseDirect = null;
            string prevNativeThemeExperimental = null;
            string prevNativeThemeTrust = null;
            string prevNativeThemeUnsafe = null;
            MixRuntimeFlags.SignatureOnlyFastPath = true;
            global::DspFindSeed.StarGen.SkipNameGeneration = true;
            if (mixSpeedOnly)
            {
                prevNativeSpeedOnly = Environment.GetEnvironmentVariable("DSP_NATIVE_SIG_SPEED_ONLY");
                prevNativePoseDirect = Environment.GetEnvironmentVariable("DSP_NATIVE_SIG_GPU_POSE_DIRECT");
                prevNativeThemeExperimental = Environment.GetEnvironmentVariable("DSP_NATIVE_SIG_GPU_THEME_VEIN_HASH_EXPERIMENTAL");
                prevNativeThemeTrust = Environment.GetEnvironmentVariable("DSP_NATIVE_SIG_GPU_THEME_VEIN_HASH_TRUST");
                prevNativeThemeUnsafe = Environment.GetEnvironmentVariable("DSP_NATIVE_SIG_GPU_THEME_VEIN_HASH_UNSAFE");
                Environment.SetEnvironmentVariable("DSP_NATIVE_SIG_SPEED_ONLY", "1");
                Environment.SetEnvironmentVariable("DSP_NATIVE_SIG_GPU_POSE_DIRECT", "1");
                Environment.SetEnvironmentVariable("DSP_NATIVE_SIG_GPU_THEME_VEIN_HASH_EXPERIMENTAL", "1");
                Environment.SetEnvironmentVariable("DSP_NATIVE_SIG_GPU_THEME_VEIN_HASH_TRUST", "1");
                Environment.SetEnvironmentVariable("DSP_NATIVE_SIG_GPU_THEME_VEIN_HASH_UNSAFE", "1");
            }
            try
            {
                ulong[] CopySigSlice(ulong[] src, int len)
                {
                    if (src == null || len <= 0)
                        return null;
                    var dst = new ulong[len];
                    Array.Copy(src, 0, dst, 0, len);
                    return dst;
                }

                NativeSeedSigChunkResult RunNativeSeedSigChunk(int seedBase, int chunk)
                {
                    var result = new NativeSeedSigChunkResult
                    {
                        seedBase = seedBase,
                        chunkSize = chunk
                    };
                    long ts0 = timingDebug ? Stopwatch.GetTimestamp() : 0;
                    bool ok = CudaGalaxyNative.TryEvalMixSignaturesFromSeedRange(
                        seedBase,
                        chunk,
                        starCount,
                        collisionFp64: mixCollisionFp64,
                        useFp32ProbCompare: useFp32Veins,
                        out var chunkNativeGalaxySig,
                        out var chunkNativePlanetSig,
                        out var chunkNativeVeinSig,
                        out var chunkNativePipelineSig);
                    if (timingDebug)
                        result.submitTicks = Stopwatch.GetTimestamp() - ts0;
                    result.useNativeSeedSigs = ok;
                    if (ok && !mixSpeedOnly)
                    {
                        result.galaxySigs = CopySigSlice(chunkNativeGalaxySig, chunk);
                        result.planetSigs = CopySigSlice(chunkNativePlanetSig, chunk);
                        result.veinSigs = CopySigSlice(chunkNativeVeinSig, chunk);
                        result.pipelineSigs = CopySigSlice(chunkNativePipelineSig, chunk);
                    }
                    return result;
                }

                void ProcessNativeSeedSigChunkResult(NativeSeedSigChunkResult chunkResult)
                {
                    int seedBase = chunkResult.seedBase;
                    int chunk = chunkResult.chunkSize;
                    mixChunkNativeSeedSigOk++;
                    if (mixSpeedOnly)
                    {
                        total += chunk;
                        return;
                    }
                    long tc0 = timingDebug ? Stopwatch.GetTimestamp() : 0;
                    for (int i = 0; i < chunk; i++)
                    {
                        int s = seedBase + i;
                        int idx = s - startSeed;
                        var cpu = cpuRows[idx];

                        ulong hg64 = cpu.GalaxySig;
                        ulong hg32 = chunkResult.galaxySigs[i];
                        if (hg64 != hg32) mismatchGalaxy++;

                        ulong hp64 = cpu.PlanetSig;
                        ulong hp32 = chunkResult.planetSigs[i];
                        if (hp64 != hp32) mismatchPlanets++;

                        ulong hv64 = cpu.VeinSig;
                        ulong hv32 = chunkResult.veinSigs[i];
                        if (hv64 != hv32) mismatchVeins++;

                        ulong hall64 = cpu.PipelineSig;
                        ulong hall32 = chunkResult.pipelineSigs[i];
                        if (hall64 != hall32) mismatchPipeline++;

                        total++;
                        if (showMismatches > 0 && shown < showMismatches && hall64 != hall32)
                        {
                            long td0 = timingDebug ? Stopwatch.GetTimestamp() : 0;
                            shown++;
                            var gd64 = new global::DspFindSeed.GameDesc();
                            gd64.SetForNewGame(global::DspFindSeed.UniverseGen.algoVersion, s, starCount, 1, 1f);
                            var g64Detail = global::DspFindSeed.UniverseGen.CreateGalaxy(gd64);

                            var gdMix = new global::DspFindSeed.GameDesc();
                            gdMix.SetForNewGame(global::DspFindSeed.UniverseGen.algoVersion, s, starCount, 1, 1f);
                            var gMixDetail = UniverseGenPipelineMix.CreateGalaxy(gdMix, collisionFp64: mixCollisionFp64);

                            Console.WriteLine();
                            Console.WriteLine($"--- mismatch #{shown} seed={s} stars={starCount} ---");
                            Console.WriteLine($"galaxySig64=0x{hg64:X16} galaxySigMix=0x{hg32:X16}");
                            Console.WriteLine($"planetSig64=0x{hp64:X16} planetSigMix=0x{hp32:X16}");
                            Console.WriteLine($"veinSig64=0x{hv64:X16} veinSigMix=0x{hv32:X16}");
                            Console.WriteLine($"pipeSig64=0x{hall64:X16} pipeSigMix=0x{hall32:X16}");
                            Console.WriteLine(DescribeFirstDifferenceWithVeins(g64Detail, gMixDetail));
                            if (timingDebug)
                                tMixDetail += Stopwatch.GetTimestamp() - td0;
                        }
                    }
                    if (timingDebug)
                        tMixCompare += Stopwatch.GetTimestamp() - tc0;
                }

                void ProcessObjectFallbackChunk(int seedBase, int chunk)
                {
                    mixChunkObjectFallback++;
                    var chunkRun = chunkRunner.RunChunk(seedBase, chunk, enableBatchPrefetch: false, timingDebug, chunkBuffers);
                    if (timingDebug)
                        tMixCreateGalaxy += chunkRun.CreateGalaxyTicks;
                    if (mixSpeedOnly)
                    {
                        total += chunk;
                        MixObjectPool.ReleaseGalaxies(chunkBuffers.GalaxyBuffer, chunk);
                        return;
                    }

                    long tv0 = timingDebug ? Stopwatch.GetTimestamp() : 0;
                    bool useChunkCudaCounts = TryGetCudaPlanetCountsBatch(
                        chunkBuffers.GalaxyBuffer,
                        chunk,
                        useFp32Veins,
                        out var chunkCudaCounts,
                        out var chunkCudaStride,
                        out var chunkPlanetStartIdx);
                    if (timingDebug)
                        tMixVeinBatch += Stopwatch.GetTimestamp() - tv0;
                    if (useChunkCudaCounts)
                        mixChunkCudaCountsOk++;

                    bool useChunkNativeSig = false;
                    ulong[] chunkFallbackGalaxySig = null;
                    ulong[] chunkFallbackPlanetSig = null;
                    ulong[] chunkFallbackVeinSig = null;
                    ulong[] chunkFallbackPipelineSig = null;
                    if (useChunkCudaCounts)
                    {
                        useChunkNativeSig = CudaGalaxyNative.TryReduceMixChunkSignatures(
                            chunkBuffers.GalaxyBuffer,
                            chunk,
                            chunkCudaCounts,
                            chunkCudaStride,
                            chunkPlanetStartIdx,
                            out chunkFallbackGalaxySig,
                            out chunkFallbackPlanetSig,
                            out chunkFallbackVeinSig,
                            out chunkFallbackPipelineSig);
                        if (useChunkNativeSig)
                            mixChunkNativeSigOk++;
                    }

                    long tc1 = timingDebug ? Stopwatch.GetTimestamp() : 0;
                    for (int i = 0; i < chunk; i++)
                    {
                        int s = seedBase + i;
                        int idx = s - startSeed;
                        var gMix = chunkBuffers.GalaxyBuffer[i];
                        var cpu = cpuRows[idx];

                        ulong hg64 = cpu.GalaxySig;
                        ulong hg32 = useChunkNativeSig
                            ? chunkFallbackGalaxySig[i]
                            : SignatureGalaxyOnly(gMix);
                        if (hg64 != hg32) mismatchGalaxy++;

                        ulong hp64 = cpu.PlanetSig;
                        ulong hp32 = useChunkNativeSig
                            ? chunkFallbackPlanetSig[i]
                            : Signature(gMix);
                        if (hp64 != hp32) mismatchPlanets++;

                        ulong hv64 = cpu.VeinSig;
                        ulong hv32 = useChunkNativeSig
                            ? chunkFallbackVeinSig[i]
                            : (useChunkCudaCounts
                            ? SignatureVeinsOnly(gMix, useFp32Veins: useFp32Veins, cudaCountsFlat: chunkCudaCounts, cudaStride: chunkCudaStride, cudaPlanetStartIdx: chunkPlanetStartIdx[i])
                            : SignatureVeinsOnly(gMix, useFp32Veins: useFp32Veins));
                        if (hv64 != hv32) mismatchVeins++;

                        ulong hall64 = cpu.PipelineSig;
                        ulong hall32 = useChunkNativeSig
                            ? chunkFallbackPipelineSig[i]
                            : (useChunkCudaCounts
                            ? SignaturePipeline(gMix, useFp32Veins: useFp32Veins, cudaCountsFlat: chunkCudaCounts, cudaStride: chunkCudaStride, cudaPlanetStartIdx: chunkPlanetStartIdx[i])
                            : SignaturePipeline(gMix, useFp32Veins: useFp32Veins));
                        if (hall64 != hall32) mismatchPipeline++;

                        total++;
                        if (showMismatches > 0 && shown < showMismatches && hall64 != hall32)
                        {
                            long td0 = timingDebug ? Stopwatch.GetTimestamp() : 0;
                            shown++;
                            var gd64 = new global::DspFindSeed.GameDesc();
                            gd64.SetForNewGame(global::DspFindSeed.UniverseGen.algoVersion, s, starCount, 1, 1f);
                            var g64Detail = global::DspFindSeed.UniverseGen.CreateGalaxy(gd64);
                            Console.WriteLine();
                            Console.WriteLine($"--- mismatch #{shown} seed={s} stars={starCount} ---");
                            Console.WriteLine($"galaxySig64=0x{hg64:X16} galaxySigMix=0x{hg32:X16}");
                            Console.WriteLine($"planetSig64=0x{hp64:X16} planetSigMix=0x{hp32:X16}");
                            Console.WriteLine($"veinSig64=0x{hv64:X16} veinSigMix=0x{hv32:X16}");
                            Console.WriteLine($"pipeSig64=0x{hall64:X16} pipeSigMix=0x{hall32:X16}");
                            Console.WriteLine(DescribeFirstDifferenceWithVeins(g64Detail, gMix));
                            if (timingDebug)
                                tMixDetail += Stopwatch.GetTimestamp() - td0;
                        }
                    }
                    if (timingDebug)
                        tMixCompare += Stopwatch.GetTimestamp() - tc1;

                    MixObjectPool.ReleaseGalaxies(chunkBuffers.GalaxyBuffer, chunk);
                }

                nativeInFlightLimit = gpuStreams < 1 ? 1 : gpuStreams;
                int envNativeInFlight = GetIntEnv("DSP_MIX_NATIVE_INFLIGHT", 0);
                if (envNativeInFlight > 0)
                {
                    nativeInFlightLimit = envNativeInFlight;
                }
                else
                {
                    nativeInFlightLimit = nativeInFlightLimit * 2;
                    if (nativeInFlightLimit > 20)
                        nativeInFlightLimit = 20;
                }
                if (nativeInFlightLimit < 2)
                    nativeInFlightLimit = 2;
                var nativeInFlight = new List<Task<NativeSeedSigChunkResult>>(nativeInFlightLimit);
                bool nativeSeedSigDisabled = false;

                void DrainOneNativeTask()
                {
                    if (nativeInFlight.Count <= 0)
                        return;

                    Task<NativeSeedSigChunkResult> task = null;
                    int completedIdx = -1;
                    for (int i = 0; i < nativeInFlight.Count; i++)
                    {
                        if (nativeInFlight[i].IsCompleted)
                        {
                            completedIdx = i;
                            break;
                        }
                    }
                    if (completedIdx >= 0)
                    {
                        task = nativeInFlight[completedIdx];
                        nativeInFlight.RemoveAt(completedIdx);
                    }
                    else
                    {
                        task = Task.WhenAny(nativeInFlight).GetAwaiter().GetResult();
                        nativeInFlight.Remove(task);
                    }

                    var chunkResult = task.GetAwaiter().GetResult();
                    if (timingDebug)
                        tMixChunkSubmit += chunkResult.submitTicks;
                    if (chunkResult.useNativeSeedSigs)
                    {
                        ProcessNativeSeedSigChunkResult(chunkResult);
                        return;
                    }
                    nativeSeedSigDisabled = true;
                    ProcessObjectFallbackChunk(chunkResult.seedBase, chunkResult.chunkSize);
                }

                for (int seedBase = startSeed; seedBase < endSeed; seedBase += mixRunChunkSeeds)
                {
                    int chunk = Math.Min(mixRunChunkSeeds, endSeed - seedBase);
                    mixChunkCount++;
                    mixChunkSeedTotal += chunk;
                    mixChunkTotal++;

                    if (nativeSeedSigDisabled)
                    {
                        ProcessObjectFallbackChunk(seedBase, chunk);
                        continue;
                    }

                    int chunkSeedBase = seedBase;
                    int chunkSize = chunk;
                    nativeInFlight.Add(Task.Run(() => RunNativeSeedSigChunk(chunkSeedBase, chunkSize)));
                    if (nativeInFlight.Count >= nativeInFlightLimit)
                        DrainOneNativeTask();
                }

                while (nativeInFlight.Count > 0)
                    DrainOneNativeTask();
            }
            finally
            {
                if (mixSpeedOnly)
                {
                    Environment.SetEnvironmentVariable("DSP_NATIVE_SIG_SPEED_ONLY", prevNativeSpeedOnly);
                    Environment.SetEnvironmentVariable("DSP_NATIVE_SIG_GPU_POSE_DIRECT", prevNativePoseDirect);
                    Environment.SetEnvironmentVariable("DSP_NATIVE_SIG_GPU_THEME_VEIN_HASH_EXPERIMENTAL", prevNativeThemeExperimental);
                    Environment.SetEnvironmentVariable("DSP_NATIVE_SIG_GPU_THEME_VEIN_HASH_TRUST", prevNativeThemeTrust);
                    Environment.SetEnvironmentVariable("DSP_NATIVE_SIG_GPU_THEME_VEIN_HASH_UNSAFE", prevNativeThemeUnsafe);
                }
                global::DspFindSeed.StarGen.SkipNameGeneration = prevSkipNameGeneration;
                MixRuntimeFlags.SignatureOnlyFastPath = false;
            }
            if (timingDebug || mixSpeedOnly)
                tMixTotal += Stopwatch.GetTimestamp() - tMixStart;

            var label = useFp32Veins ? "compare-pipeline-mix-veins-f32" : "compare-pipeline-mix";
            Console.WriteLine($"{label} startSeed={startSeed} stars={starCount} count={count}");
            Console.WriteLine($"mixSpeedOnly={mixSpeedOnly}");
            Console.WriteLine($"mixCollisionFp64={mixCollisionFp64}");
            Console.WriteLine($"cpuThreads={cpuThreads} mixThreads={mixThreads} gpuChunkSeeds={seedBatchSize} gpuStreams={gpuStreams} cpuParallel={(cpuThreads > 1 ? "True" : "False")}");
            Console.WriteLine($"mixCoreGroupSeeds={mixCoreGroupSeeds}");
            if (mixSpeedOnly)
                Console.WriteLine("cpuFp64Cache=skipped (mix-speed-only)");
            else if (useCpuCache)
                Console.WriteLine($"cpuFp64Cache={(cpuCacheLoaded ? "loaded" : (cpuCacheSaved ? "saved" : "built-no-save"))} file={resolvedCpuCacheFile}");
            else
                Console.WriteLine("cpuFp64Cache=disabled");
            Console.WriteLine($"mixChunkSeeds={seedBatchSize} mixRunChunkSeeds={mixRunChunkSeeds} (derivedFromCpuThreads*batchPerThread={(gpuChunkSeeds > 0 ? "False" : "True")}) mixCudaPlanetBatchChunks={mixChunkCudaCountsOk}/{mixChunkTotal}");
            Console.WriteLine($"mixNativeSigChunks={mixChunkNativeSigOk}/{mixChunkTotal}");
            Console.WriteLine($"mixNativeSeedSigChunks={mixChunkNativeSeedSigOk}/{mixChunkTotal} mixObjectFallbackChunks={mixChunkObjectFallback}/{mixChunkTotal}");
            Console.WriteLine($"mixNativeInFlight={nativeInFlightLimit}");
            if (mixChunkCount > 0)
                Console.WriteLine($"mixChunkCount={mixChunkCount} mixChunkSizeAvg={mixChunkSeedTotal / (double)mixChunkCount:F2}");
            Console.WriteLine($"seedBatchSize={seedBatchSize} prefetchChunks=0/0");
            if (mixSpeedOnly)
            {
                double mixMs = tMixTotal * 1000.0 / Stopwatch.Frequency;
                double seedsPerSec = mixMs > 0.0 ? total * 1000.0 / mixMs : 0.0;
                Console.WriteLine($"speedOnlyTotalSeeds={total}");
                Console.WriteLine($"speedOnlyMixMs={mixMs:F3}");
                Console.WriteLine($"speedOnlySeedsPerSec={seedsPerSec:F2}");
                Console.WriteLine("说明：仅执行 Mix/GPU 路线用于测速，不生成 CPU FP64，不做正确率对比。");
            }
            else
            {
                Console.WriteLine($"galaxyMismatch={mismatchGalaxy}/{total} ({(total > 0 ? (mismatchGalaxy * 100.0 / total) : 0):F6}%)");
                Console.WriteLine($"planetMismatch={mismatchPlanets}/{total} ({(total > 0 ? (mismatchPlanets * 100.0 / total) : 0):F6}%)");
                Console.WriteLine($"veinMismatch={mismatchVeins}/{total} ({(total > 0 ? (mismatchVeins * 100.0 / total) : 0):F6}%)");
                Console.WriteLine($"pipelineMismatch={mismatchPipeline}/{total} ({(total > 0 ? (mismatchPipeline * 100.0 / total) : 0):F6}%)");
                Console.WriteLine("说明：先全量生成 CPU FP64 签名（可缓存复用），再全量生成 Mix 并对比。");
            }
            if (timingDebug)
            {
                double Ms(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;
                var cudaGalaxyPerf = CudaGalaxyNative.GetPerfStats();
                var cudaPlanetPerf = CudaPlanetNative.GetPerfStats();
                long tAll = Stopwatch.GetTimestamp() - tAllStart;
                Console.WriteLine("timingDebug=True");
                Console.WriteLine($"timing.totalMs={Ms(tAll):F3}");
                Console.WriteLine($"timing.cacheLoadMs={Ms(tCacheLoad):F3} cpuGenMs={Ms(tCpuGen):F3} cacheSaveMs={Ms(tCacheSave):F3}");
                Console.WriteLine($"timing.mixTotalMs={Ms(tMixTotal):F3} chunkSubmitMs={Ms(tMixChunkSubmit):F3} prefetchMs={Ms(tMixPrefetch):F3} createGalaxyMs={Ms(tMixCreateGalaxy):F3} veinBatchMs={Ms(tMixVeinBatch):F3} compareMs={Ms(tMixCompare):F3} detailMs={Ms(tMixDetail):F3}");
                Console.WriteLine($"timing.cudaGalaxy.singleCalls={cudaGalaxyPerf.singleCalls} singleMs={cudaGalaxyPerf.singleMs:F3} batchCalls={cudaGalaxyPerf.batchCalls} batchSeeds={cudaGalaxyPerf.batchSeeds} batchMs={cudaGalaxyPerf.batchMs:F3} fail={cudaGalaxyPerf.failCalls}");
                Console.WriteLine($"timing.cudaGalaxy.mixSeedSigCalls={cudaGalaxyPerf.mixSeedSigCalls} mixSeedSigSeeds={cudaGalaxyPerf.mixSeedSigSeeds} mixSeedSigMs={cudaGalaxyPerf.mixSeedSigMs:F3} mixSeedSigFail={cudaGalaxyPerf.mixSeedSigFailCalls}");
                Console.WriteLine($"timing.cudaPlanet.coreReq={cudaPlanetPerf.coreReqCount} coreReqWaitMs={cudaPlanetPerf.coreReqWaitMs:F3} coreBatchCalls={cudaPlanetPerf.coreBatchCalls} coreBatchItems={cudaPlanetPerf.coreBatchItems} coreBatchMs={cudaPlanetPerf.coreBatchMs:F3} coreBatchFallback={cudaPlanetPerf.coreBatchFallbackCalls} coreSingleCalls={cudaPlanetPerf.coreSingleCalls} coreSingleMs={cudaPlanetPerf.coreSingleMs:F3}");
                Console.WriteLine($"timing.cudaPlanet.veinBatchCalls={cudaPlanetPerf.veinBatchCalls} veinBatchPlanets={cudaPlanetPerf.veinBatchPlanets} veinBatchMs={cudaPlanetPerf.veinBatchMs:F3} fail={cudaPlanetPerf.veinBatchFailCalls}");
                double gpuApproxMs = cudaGalaxyPerf.singleMs + cudaGalaxyPerf.batchMs + cudaGalaxyPerf.mixSeedSigMs + cudaPlanetPerf.coreBatchMs + cudaPlanetPerf.coreSingleMs + cudaPlanetPerf.veinBatchMs;
                Console.WriteLine($"timing.gpuApproxMs={gpuApproxMs:F3} timing.gpuFallbackApprox={cudaGalaxyPerf.failCalls + cudaGalaxyPerf.mixSeedSigFailCalls + cudaPlanetPerf.veinBatchFailCalls + cudaPlanetPerf.coreBatchFallbackCalls}");
            }
        }

        private static string ExtractKind(string diff)
        {
            if (string.IsNullOrEmpty(diff)) return null;
            // 约定：DescribeFirstDifference* 的第一段会包含类似 "galaxy.xxx" / "star.xxx" / "planet.xxx" / "veinSpot.xxx"
            int sp = diff.IndexOf(' ');
            if (sp <= 0) return diff;
            return diff.Substring(0, sp);
        }

        private static ulong SignaturePipeline(
            GalaxyData g,
            bool useFp32Veins,
            int[] cudaCountsFlat = null,
            int cudaStride = 0,
            int cudaPlanetStartIdx = -1,
            bool allowCudaPlanetCounts = true)
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

                bool useCudaPlanetCounts;
                if (cudaCountsFlat != null && cudaStride > 0 && cudaPlanetStartIdx >= 0)
                {
                    useCudaPlanetCounts = true;
                }
                else if (allowCudaPlanetCounts)
                {
                    useCudaPlanetCounts = TryGetCudaPlanetCounts(g, useFp32Veins, out cudaCountsFlat, out cudaStride);
                }
                else
                {
                    useCudaPlanetCounts = false;
                }
                int cudaPlanetIdx = cudaPlanetStartIdx >= 0 ? cudaPlanetStartIdx : 0;

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
                        if (useCudaPlanetCounts)
                        {
                            int off = cudaPlanetIdx * cudaStride;
                            cudaPlanetIdx++;
                            int max = Math.Min(cudaStride, 32);
                            for (int vid = 1; vid < max; vid++)
                                Mix(cudaCountsFlat[off + vid]);
                        }
                        else
                        {
                            var counts = useFp32Veins
                                ? global::DspFindSeed.PlanetModelingManager.RefreshPlanetData_F32(p)
                                : global::DspFindSeed.PlanetModelingManager.RefreshPlanetData(p);
                            if (counts == null) { Mix(-3); continue; }
                            int max = Math.Min(counts.Length, 32);
                            for (int vid = 1; vid < max; vid++)
                                Mix(counts[vid]);
                        }
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

        private static ulong SignatureVeinsOnly(
            GalaxyData g,
            bool useFp32Veins,
            int[] cudaCountsFlat = null,
            int cudaStride = 0,
            int cudaPlanetStartIdx = -1,
            bool allowCudaPlanetCounts = true)
        {
            unchecked
            {
                const ulong FNV_OFFSET = 14695981039346656037UL;
                const ulong FNV_PRIME = 1099511628211UL;
                ulong h = FNV_OFFSET;
                void Mix(int v) { h ^= (uint)v; h *= FNV_PRIME; }

                if (g == null || g.stars == null) return 0;
                bool useCudaPlanetCounts;
                if (cudaCountsFlat != null && cudaStride > 0 && cudaPlanetStartIdx >= 0)
                {
                    useCudaPlanetCounts = true;
                }
                else if (allowCudaPlanetCounts)
                {
                    useCudaPlanetCounts = TryGetCudaPlanetCounts(g, useFp32Veins, out cudaCountsFlat, out cudaStride);
                }
                else
                {
                    useCudaPlanetCounts = false;
                }
                int cudaPlanetIdx = cudaPlanetStartIdx >= 0 ? cudaPlanetStartIdx : 0;
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
                        if (useCudaPlanetCounts)
                        {
                            int off = cudaPlanetIdx * cudaStride;
                            cudaPlanetIdx++;
                            int max = Math.Min(cudaStride, 32);
                            for (int vid = 1; vid < max; vid++)
                                Mix(cudaCountsFlat[off + vid]);
                        }
                        else
                        {
                            var counts = useFp32Veins
                                ? global::DspFindSeed.PlanetModelingManager.RefreshPlanetData_F32(p)
                                : global::DspFindSeed.PlanetModelingManager.RefreshPlanetData(p);
                            if (counts == null) { Mix(-3); continue; }
                            int max = Math.Min(counts.Length, 32);
                            for (int vid = 1; vid < max; vid++)
                                Mix(counts[vid]);
                        }
                    }
                }
                return h;
            }
        }

        private static bool TryGetCudaPlanetCounts(GalaxyData g, bool useFp32Veins, out int[] countsFlat, out int stride)
        {
            countsFlat = null;
            stride = 0;
            if (g == null || g.stars == null)
                return false;

            var planets = new List<PlanetData>();
            for (int si = 0; si < g.stars.Length; ++si)
            {
                var star = g.stars[si];
                if (star?.planets == null)
                    continue;
                for (int pi = 0; pi < star.planets.Length; ++pi)
                {
                    var p = star.planets[pi];
                    if (p == null || p.type == EPlanetType.Gas)
                        continue;
                    planets.Add(p);
                }
            }

            if (planets.Count == 0)
                return false;

            return CudaPlanetNative.TryRefreshPlanetVeinSpotsBatch(planets, useFp32Veins, out countsFlat, out stride);
        }

        private static bool TryGetCudaPlanetCountsBatch(
            GalaxyData[] galaxies,
            int galaxyCount,
            bool useFp32Veins,
            out int[] countsFlat,
            out int stride,
            out int[] galaxyPlanetStartIdx)
        {
            countsFlat = null;
            stride = 0;
            galaxyPlanetStartIdx = null;
            if (galaxies == null || galaxyCount <= 0)
                return false;

            var planets = new List<PlanetData>();
            int maxGi = Math.Min(galaxyCount, galaxies.Length);
            galaxyPlanetStartIdx = new int[maxGi];
            for (int gi = 0; gi < maxGi; ++gi)
            {
                galaxyPlanetStartIdx[gi] = planets.Count;
                var g = galaxies[gi];
                if (g == null || g.stars == null)
                    continue;
                for (int si = 0; si < g.stars.Length; ++si)
                {
                    var star = g.stars[si];
                    if (star?.planets == null)
                        continue;
                    for (int pi = 0; pi < star.planets.Length; ++pi)
                    {
                        var p = star.planets[pi];
                        if (p == null || p.type == EPlanetType.Gas)
                            continue;
                        planets.Add(p);
                    }
                }
            }

            if (planets.Count == 0)
                return false;

            return CudaPlanetNative.TryRefreshPlanetVeinSpotsBatch(planets, useFp32Veins, out countsFlat, out stride);
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

        private static void DumpSeedCompareMix(int seed, int starCount, bool useFp32Veins, bool mixCollisionFp64)
        {
            var algoVersion = global::DspFindSeed.UniverseGen.algoVersion;
            var gd = new global::DspFindSeed.GameDesc();
            gd.SetForNewGame(algoVersion, seed, starCount, 1, 1f);

            var g64 = global::DspFindSeed.UniverseGen.CreateGalaxy(gd);
            var gMix = UniverseGenPipelineMix.CreateGalaxy(gd, collisionFp64: mixCollisionFp64);
            if (g64 == null || g64.stars == null)
                throw new InvalidOperationException("FP64 CreateGalaxy 返回空。");
            if (gMix == null || gMix.stars == null)
                throw new InvalidOperationException("Mix CreateGalaxy 返回空。");

            ulong hg64 = SignatureGalaxyOnly(g64);
            ulong hgMix = SignatureGalaxyOnly(gMix);
            ulong hp64 = Signature(g64);
            ulong hpMix = Signature(gMix);
            ulong hv64 = SignatureVeinsOnly(g64, useFp32Veins: false);
            ulong hvMix = SignatureVeinsOnly(gMix, useFp32Veins: useFp32Veins);
            ulong hall64 = SignaturePipeline(g64, useFp32Veins: false);
            ulong hallMix = SignaturePipeline(gMix, useFp32Veins: useFp32Veins);

            string mode = useFp32Veins ? "mix-veins-f32" : "mix";
            Console.WriteLine($"dump-compare mode={mode} seed={seed} stars={starCount} algo={algoVersion}");
            Console.WriteLine($"mixCollisionFp64={mixCollisionFp64}");
            Console.WriteLine($"galaxySig64=0x{hg64:X16} galaxySigMix=0x{hgMix:X16}");
            Console.WriteLine($"planetSig64=0x{hp64:X16} planetSigMix=0x{hpMix:X16}");
            Console.WriteLine($"veinSig64=0x{hv64:X16} veinSigMix=0x{hvMix:X16}");
            Console.WriteLine($"pipeSig64=0x{hall64:X16} pipeSigMix=0x{hallMix:X16}");
            Console.WriteLine();

            int starLen64 = g64.stars != null ? g64.stars.Length : 0;
            int starLenMix = gMix.stars != null ? gMix.stars.Length : 0;
            Console.WriteLine($"galaxy.starCount FP64={g64.starCount} MIX={gMix.starCount}");
            Console.WriteLine($"galaxy.birthStarId FP64={g64.birthStarId} MIX={gMix.birthStarId}");
            Console.WriteLine($"galaxy.birthPlanetId FP64={g64.birthPlanetId} MIX={gMix.birthPlanetId}");
            Console.WriteLine($"stars.Length FP64={starLen64} MIX={starLenMix}");
            Console.WriteLine();

            int sc = Math.Max(starLen64, starLenMix);
            for (int si = 0; si < sc; si++)
            {
                var s64 = si < starLen64 ? g64.stars[si] : null;
                var sMix = si < starLenMix ? gMix.stars[si] : null;

                Console.WriteLine($"[Star {si}]");
                Console.WriteLine($"  FP64: {FormatStarLine(s64)}");
                Console.WriteLine($"  MIX : {FormatStarLine(sMix)}");

                if (s64 == null || sMix == null)
                {
                    Console.WriteLine("  DIFF: star null 状态不同");
                    Console.WriteLine();
                    continue;
                }

                bool starFieldDiff = false;
                if (s64.id != sMix.id) { Console.WriteLine($"  DIFF: id FP64={s64.id} MIX={sMix.id}"); starFieldDiff = true; }
                if (s64.type != sMix.type) { Console.WriteLine($"  DIFF: type FP64={s64.type} MIX={sMix.type}"); starFieldDiff = true; }
                if (s64.spectr != sMix.spectr) { Console.WriteLine($"  DIFF: spectr FP64={s64.spectr} MIX={sMix.spectr}"); starFieldDiff = true; }
                if (s64.planetCount != sMix.planetCount) { Console.WriteLine($"  DIFF: planetCount FP64={s64.planetCount} MIX={sMix.planetCount}"); starFieldDiff = true; }

                int x64 = (int)Math.Round(s64.uPosition.x * 0.001);
                int y64 = (int)Math.Round(s64.uPosition.y * 0.001);
                int z64 = (int)Math.Round(s64.uPosition.z * 0.001);
                int xMix = (int)Math.Round(sMix.uPosition.x * 0.001);
                int yMix = (int)Math.Round(sMix.uPosition.y * 0.001);
                int zMix = (int)Math.Round(sMix.uPosition.z * 0.001);
                if (x64 != xMix || y64 != yMix || z64 != zMix)
                {
                    Console.WriteLine($"  DIFF: uPosition(×1e-3量化) FP64=({x64},{y64},{z64}) MIX=({xMix},{yMix},{zMix})");
                    starFieldDiff = true;
                }

                if (!starFieldDiff)
                    Console.WriteLine("  DIFF: (none)");

                var p64 = s64.planets;
                var pMix = sMix.planets;
                int pLen64 = p64 != null ? p64.Length : 0;
                int pLenMix = pMix != null ? pMix.Length : 0;
                int pc = Math.Max(pLen64, pLenMix);
                for (int pi = 0; pi < pc; pi++)
                {
                    var a = pi < pLen64 ? p64[pi] : null;
                    var b = pi < pLenMix ? pMix[pi] : null;

                    Console.WriteLine($"  [Planet {pi}]");
                    Console.WriteLine($"    FP64: {FormatPlanetLine(a)}");
                    Console.WriteLine($"    MIX : {FormatPlanetLine(b)}");

                    if (a == null || b == null)
                    {
                        Console.WriteLine("    DIFF: planet null 状态不同");
                        continue;
                    }

                    bool planetFieldDiff = false;
                    if (a.type != b.type) { Console.WriteLine($"    DIFF: type FP64={a.type} MIX={b.type}"); planetFieldDiff = true; }
                    if (a.theme != b.theme) { Console.WriteLine($"    DIFF: theme FP64={a.theme} MIX={b.theme}"); planetFieldDiff = true; }
                    if (a.waterItemId != b.waterItemId) { Console.WriteLine($"    DIFF: waterItemId FP64={a.waterItemId} MIX={b.waterItemId}"); planetFieldDiff = true; }
                    if (a.orbitIndex != b.orbitIndex) { Console.WriteLine($"    DIFF: orbitIndex FP64={a.orbitIndex} MIX={b.orbitIndex}"); planetFieldDiff = true; }
                    if (a.orbitAround != b.orbitAround) { Console.WriteLine($"    DIFF: orbitAround FP64={a.orbitAround} MIX={b.orbitAround}"); planetFieldDiff = true; }
                    if (a.singularity != b.singularity) { Console.WriteLine($"    DIFF: singularity FP64={a.singularity} MIX={b.singularity}"); planetFieldDiff = true; }
                    if (!planetFieldDiff)
                        Console.WriteLine("    DIFF: (none)");

                    if (a.type == EPlanetType.Gas || b.type == EPlanetType.Gas)
                        continue;

                    var c64 = global::DspFindSeed.PlanetModelingManager.RefreshPlanetData(a);
                    var cMix = useFp32Veins
                        ? global::DspFindSeed.PlanetModelingManager.RefreshPlanetData_F32(b)
                        : global::DspFindSeed.PlanetModelingManager.RefreshPlanetData(b);
                    DumpVeinCountsCompare(c64, cMix);
                }

                Console.WriteLine();
            }
        }

        private static string FormatStarLine(StarData star)
        {
            if (star == null)
                return "(null)";
            int x = (int)Math.Round(star.uPosition.x * 0.001);
            int y = (int)Math.Round(star.uPosition.y * 0.001);
            int z = (int)Math.Round(star.uPosition.z * 0.001);
            return $"id={star.id} name={star.displayName ?? star.name} type={star.type} spectr={star.spectr} planets={star.planetCount} uPosQ=({x},{y},{z})";
        }

        private static string FormatPlanetLine(PlanetData p)
        {
            if (p == null)
                return "(null)";
            var theme = global::DspFindSeed.LDB.themes.Select(p.theme);
            string themeName = theme != null ? theme.DisplayName : $"themeId={p.theme}";
            return $"id={p.id} name={p.displayName ?? p.name} type={p.type} theme={themeName}({p.theme}) orbitIndex={p.orbitIndex} orbitAround={p.orbitAround} singularity={p.singularity} waterItemId={p.waterItemId} tempBias={p.temperatureBias.ToString("R", CultureInfo.InvariantCulture)} habBias={p.habitableBias.ToString("R", CultureInfo.InvariantCulture)}";
        }

        private static void DumpVeinCountsCompare(int[] c64, int[] cMix)
        {
            int len64 = c64 != null ? c64.Length : 0;
            int lenMix = cMix != null ? cMix.Length : 0;
            int len = Math.Max(len64, lenMix);
            if (len <= 1)
            {
                Console.WriteLine("      veinSpot: (none)");
                return;
            }

            bool any = false;
            for (int vid = 1; vid < len; vid++)
            {
                int a = vid < len64 ? c64[vid] : 0;
                int b = vid < lenMix ? cMix[vid] : 0;
                if (a == 0 && b == 0)
                    continue;

                var vp = global::DspFindSeed.LDB.veins.Select(vid);
                string name = vp != null ? vp.Name : $"veinId={vid}";
                string marker = a == b ? "" : "  <DIFF>";
                Console.WriteLine($"      veinSpot {name} (id={vid}) FP64={a} MIX={b}{marker}");
                any = true;
            }

            if (!any)
                Console.WriteLine("      veinSpot: (none)");
        }

        private static void DebugVeinBranch(int seed, int starCount, int planetId)
        {
            if (planetId <= 0)
            {
                Console.WriteLine("请提供 --planet-id <行星ID>。");
                return;
            }

            var gd = new global::DspFindSeed.GameDesc();
            gd.SetForNewGame(global::DspFindSeed.UniverseGen.algoVersion, seed, starCount, 1, 1f);
            var galaxy = global::DspFindSeed.UniverseGen.CreateGalaxy(gd);
            if (galaxy?.stars == null)
            {
                Console.WriteLine("星系生成失败。");
                return;
            }

            PlanetData target = null;
            StarData star = null;
            for (int si = 0; si < galaxy.stars.Length && target == null; ++si)
            {
                var s = galaxy.stars[si];
                if (s?.planets == null)
                    continue;
                for (int pi = 0; pi < s.planets.Length; ++pi)
                {
                    var planetCandidate = s.planets[pi];
                    if (planetCandidate != null && planetCandidate.id == planetId)
                    {
                        target = planetCandidate;
                        star = s;
                        break;
                    }
                }
            }

            if (target == null || star == null)
            {
                Console.WriteLine($"未找到 planetId={planetId}。");
                return;
            }
            if (target.type == EPlanetType.Gas)
            {
                Console.WriteLine($"planetId={planetId} 是气态行星，矿脉统计路径不适用。");
                return;
            }

            var theme = global::DspFindSeed.LDB.themes.Select(target.theme);
            if (theme == null)
            {
                Console.WriteLine($"themeId={target.theme} 不存在。");
                return;
            }

            var strictCounts = global::DspFindSeed.PlanetModelingManager.RefreshPlanetData(target);
            var f32Counts = global::DspFindSeed.PlanetModelingManager.RefreshPlanetData_F32(target);

            Console.WriteLine($"debug-vein-branch seed={seed} stars={starCount} planetId={planetId}");
            Console.WriteLine($"starId={star.id} starIndex={star.index} starType={star.type} spectr={star.spectr}");
            Console.WriteLine($"planet={target.displayName ?? target.name} type={target.type} theme={target.theme}({theme.DisplayName})");

            bool anyCountDiff = false;
            int maxLen = Math.Max(strictCounts?.Length ?? 0, f32Counts?.Length ?? 0);
            for (int vid = 1; vid < maxLen; ++vid)
            {
                int a = strictCounts != null && vid < strictCounts.Length ? strictCounts[vid] : 0;
                int b = f32Counts != null && vid < f32Counts.Length ? f32Counts[vid] : 0;
                if (a == b)
                    continue;
                var vp = global::DspFindSeed.LDB.veins.Select(vid);
                string name = vp != null ? vp.Name : $"veinId={vid}";
                Console.WriteLine($"veinDiff {name}(id={vid}) strict={a} f32={b}");
                anyCountDiff = true;
            }
            if (!anyCountDiff)
            {
                Console.WriteLine("该行星严格版与F32版矿脉统计一致，无分支翻转。");
                return;
            }

            float p = CalcVeinP(star);
            Console.WriteLine($"p={p.ToString("R", CultureInfo.InvariantCulture)}");
            if (theme.RareVeins != null && theme.RareSettings != null)
            {
                for (int idx = 0; idx < theme.RareVeins.Length; ++idx)
                {
                    int veinId = theme.RareVeins[idx];
                    float appearBase = star.index == 0 ? theme.RareSettings[idx * 4] : theme.RareSettings[idx * 4 + 1];
                    float chainProb = theme.RareSettings[idx * 4 + 2];
                    float appearProb = 1f - global::UnityEngine.Mathf.Pow(1f - appearBase, p);
                    Console.WriteLine(
                        $"rare[{idx}] veinId={veinId} appearBase={appearBase.ToString("R", CultureInfo.InvariantCulture)} " +
                        $"appearProb={appearProb.ToString("R", CultureInfo.InvariantCulture)} chainProb={chainProb.ToString("R", CultureInfo.InvariantCulture)}");
                }
            }

            string firstDivergence = FindFirstVeinProbBranchDivergence(target, theme, star, p);
            Console.WriteLine(firstDivergence ?? "未定位到概率比较分支翻转（请检查是否由更上游星系/行星差异引起）。");
        }

        private static float CalcVeinP(StarData star)
        {
            float p = 1f;
            if (star == null)
                return p;

            switch (star.type)
            {
                case EStarType.MainSeqStar:
                    switch (star.spectr)
                    {
                        case ESpectrType.M: p = 2.5f; break;
                        case ESpectrType.K: p = 1f; break;
                        case ESpectrType.G: p = 0.7f; break;
                        case ESpectrType.F: p = 0.6f; break;
                        case ESpectrType.A: p = 1f; break;
                        case ESpectrType.B: p = 0.4f; break;
                        case ESpectrType.O: p = 1.6f; break;
                    }
                    break;
                case EStarType.GiantStar:
                    p = 2.5f;
                    break;
                case EStarType.WhiteDwarf:
                    p = 3.5f;
                    break;
                case EStarType.NeutronStar:
                    p = 4.5f;
                    break;
                case EStarType.BlackHole:
                    p = 5f;
                    break;
            }

            return p;
        }

        private static string FindFirstVeinProbBranchDivergence(PlanetData planet, global::DspFindSeed.ThemeProto theme, StarData star, float p)
        {
            DotNet35Random rng64 = new DotNet35Random(planet.seed);
            DotNet35Random rng32 = new DotNet35Random(planet.seed);

            for (int i = 0; i < 5; ++i)
            {
                int a = rng64.Next();
                int b = rng32.Next();
                if (a != b)
                    return $"RNG 预热阶段已不一致 i={i} strict={a} f32={b}";
            }
            int s1 = rng64.Next();
            int s2 = rng32.Next();
            if (s1 != s2)
                return $"RNG 分叉前 seed 已不一致 strict={s1} f32={s2}";
            _ = new DotNet35Random(s1);
            _ = new DotNet35Random(s2);

            string CompareOnce(string stage, int rareIdx, int chainStep, double threshold64, float threshold32, out bool hit)
            {
                double rv64 = rng64.NextDouble();
                double rv32 = rng32.NextDouble();
                float rv32f = (float)rv32;

                bool hit64 = rv64 < threshold64;
                bool hit32 = rv32f < threshold32;
                hit = hit64;

                if (hit64 == hit32)
                    return null;

                return
                    $"firstDivergence stage={stage} rareIdx={rareIdx} chainStep={chainStep} " +
                    $"rv={rv64.ToString("R", CultureInfo.InvariantCulture)} rvF32={rv32f.ToString("R", CultureInfo.InvariantCulture)} " +
                    $"threshold64={threshold64.ToString("R", CultureInfo.InvariantCulture)} " +
                    $"threshold32={threshold32.ToString("R", CultureInfo.InvariantCulture)} " +
                    $"hit64={hit64} hit32={hit32}";
            }

            if (star.type == EStarType.WhiteDwarf)
            {
                for (int i = 1; i < 12; ++i)
                {
                    var diff = CompareOnce("bonus.whiteDwarf.vein9", -1, i, 0.449999988079071, 0.449999988079071f, out bool hit);
                    if (diff != null) return diff;
                    if (!hit) break;
                }
                for (int i = 1; i < 12; ++i)
                {
                    var diff = CompareOnce("bonus.whiteDwarf.vein10", -1, i, 0.449999988079071, 0.449999988079071f, out bool hit);
                    if (diff != null) return diff;
                    if (!hit) break;
                }
                for (int i = 1; i < 12; ++i)
                {
                    var diff = CompareOnce("bonus.whiteDwarf.vein12", -1, i, 0.5, 0.5f, out bool hit);
                    if (diff != null) return diff;
                    if (!hit) break;
                }
            }
            else if (star.type == EStarType.NeutronStar || star.type == EStarType.BlackHole)
            {
                for (int i = 1; i < 12; ++i)
                {
                    var diff = CompareOnce("bonus.compact.vein14", -1, i, 0.649999976158142, 0.649999976158142f, out bool hit);
                    if (diff != null) return diff;
                    if (!hit) break;
                }
            }

            if (theme.RareVeins == null || theme.RareSettings == null)
                return null;

            for (int idx = 0; idx < theme.RareVeins.Length; ++idx)
            {
                float appearBase = star.index == 0 ? theme.RareSettings[idx * 4] : theme.RareSettings[idx * 4 + 1];
                float chainProb = theme.RareSettings[idx * 4 + 2];
                float appearProb = 1f - global::UnityEngine.Mathf.Pow(1f - appearBase, p);

                var appearDiff = CompareOnce("rare.appear", idx, 0, (double)appearProb, appearProb, out bool appearHit);
                if (appearDiff != null) return appearDiff;
                if (!appearHit) continue;

                for (int i = 1; i < 12; ++i)
                {
                    var chainDiff = CompareOnce("rare.chain", idx, i, (double)chainProb, chainProb, out bool chainHit);
                    if (chainDiff != null) return chainDiff;
                    if (!chainHit) break;
                }
            }

            return null;
        }

        private struct CpuFp64SigRow
        {
            public int Seed;
            public ulong GalaxySig;
            public ulong PlanetSig;
            public ulong VeinSig;
            public ulong PipelineSig;
        }

        private struct NativeSeedSigChunkResult
        {
            public int seedBase;
            public int chunkSize;
            public bool useNativeSeedSigs;
            public long submitTicks;
            public ulong[] galaxySigs;
            public ulong[] planetSigs;
            public ulong[] veinSigs;
            public ulong[] pipelineSigs;
        }

        private static string ResolveCpuFp64CacheFile(string cliPath, int startSeed, int starCount, int count)
        {
            if (!string.IsNullOrWhiteSpace(cliPath))
            {
                var full = Path.GetFullPath(cliPath);
                var dir = Path.GetDirectoryName(full);
                if (!string.IsNullOrEmpty(dir))
                    Directory.CreateDirectory(dir);
                return full;
            }

            string dirDefault = Path.Combine("logs", "cpu_fp64_cache");
            Directory.CreateDirectory(dirDefault);
            string name = $"mix_cpu64_s{startSeed}_n{count}_stars{starCount}.bin";
            return Path.Combine(dirDefault, name);
        }

        private static bool TrySaveCpuFp64SigCache(string path, int startSeed, int starCount, int count, CpuFp64SigRow[] rows)
        {
            if (string.IsNullOrEmpty(path) || rows == null || rows.Length != count)
                return false;

            try
            {
                using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read))
                using (var bw = new BinaryWriter(fs))
                {
                    bw.Write(0x44535043); // DSPC
                    bw.Write(2); // version
                    int algoVersion = global::DspFindSeed.UniverseGen.algoVersion;
                    bw.Write(algoVersion);
                    bw.Write(startSeed);
                    bw.Write(starCount);
                    bw.Write(count);
                    bw.Write(BuildCpuCacheFingerprint(algoVersion, startSeed, starCount, count));
                    for (int i = 0; i < rows.Length; ++i)
                    {
                        bw.Write(rows[i].Seed);
                        bw.Write(rows[i].GalaxySig);
                        bw.Write(rows[i].PlanetSig);
                        bw.Write(rows[i].VeinSig);
                        bw.Write(rows[i].PipelineSig);
                    }
                }
                return true;
            }
            catch
            {
                return false;
            }
        }

        private static bool TryLoadCpuFp64SigCache(string path, int startSeed, int starCount, int count, CpuFp64SigRow[] rows)
        {
            if (string.IsNullOrEmpty(path) || rows == null || rows.Length != count || !File.Exists(path))
                return false;

            try
            {
                using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
                using (var br = new BinaryReader(fs))
                {
                    int magic = br.ReadInt32();
                    int version = br.ReadInt32();
                    int algoVersion = br.ReadInt32();
                    int fileStartSeed = br.ReadInt32();
                    int fileStarCount = br.ReadInt32();
                    int fileCount = br.ReadInt32();
                    if (magic != 0x44535043 || (version != 1 && version != 2))
                        return false;
                    if (algoVersion != global::DspFindSeed.UniverseGen.algoVersion)
                        return false;
                    if (fileStartSeed != startSeed || fileStarCount != starCount || fileCount != count)
                        return false;
                    if (version >= 2)
                    {
                        ulong fp = br.ReadUInt64();
                        if (fp != BuildCpuCacheFingerprint(algoVersion, fileStartSeed, fileStarCount, fileCount))
                            return false;
                    }

                    for (int i = 0; i < count; ++i)
                    {
                        rows[i].Seed = br.ReadInt32();
                        rows[i].GalaxySig = br.ReadUInt64();
                        rows[i].PlanetSig = br.ReadUInt64();
                        rows[i].VeinSig = br.ReadUInt64();
                        rows[i].PipelineSig = br.ReadUInt64();
                    }
                    return true;
                }
            }
            catch
            {
                return false;
            }
        }

        private static ulong BuildCpuCacheFingerprint(int algoVersion, int startSeed, int starCount, int count)
        {
            unchecked
            {
                const ulong FnvOffset = 14695981039346656037UL;
                const ulong FnvPrime = 1099511628211UL;
                ulong h = FnvOffset;
                void Mix(int v)
                {
                    h ^= (uint)v;
                    h *= FnvPrime;
                }

                Mix(0x43504348); // CPCH
                Mix(algoVersion);
                Mix(startSeed);
                Mix(starCount);
                Mix(count);
                return h;
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

        private static string GetStringArg(string[] args, string key, string defaultValue)
        {
            for (int i = 0; i < args.Length; i++)
            {
                if (!string.Equals(args[i], key, StringComparison.OrdinalIgnoreCase))
                    continue;
                if (i + 1 >= args.Length)
                    return defaultValue;
                return args[i + 1];
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

        private static int GetIntEnv(string key, int defaultValue)
        {
            if (string.IsNullOrEmpty(key))
                return defaultValue;
            string raw = Environment.GetEnvironmentVariable(key);
            if (string.IsNullOrEmpty(raw))
                return defaultValue;
            if (!int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v))
                return defaultValue;
            return v;
        }
    }
}
