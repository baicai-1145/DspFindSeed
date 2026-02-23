using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;

namespace SeedCli
{
    internal static class CudaGalaxyNative
    {
        private const string LibName = "dsp_cuda_galaxy";
        private const int Ok = 0;
        private const int MaxVeinSpotLen = 16;
        private const int MaxRareVeins = 8;
        private const int RareSettingsGroup = 4;
        private const int MaxOutVeinLen = 32;

        private static bool _enabledByCli;
        private static bool _nativeBroken;
        private static string _nativeBrokenReason;
        private static bool _printedFallback;
        private static bool _printedForceColl64;
        private static bool _mixSigEntryMissing;
        private static bool _mixSeedSigEntryMissing;
        private static readonly object _mixThemeLock = new object();
        private static global::DspFindSeed.ThemeProto[] _mixThemeProtoArrayRef;
        private static MixThemeFlatData _mixThemeCache;
        [ThreadStatic] private static NativeVec3d[] _singlePoseScratch;
        [ThreadStatic] private static int[] _mixSeedSigSeeds;
        [ThreadStatic] private static int[] _mixSigSeedStarOffsets;
        [ThreadStatic] private static int[] _mixSigSeedPlanetOffsets;
        [ThreadStatic] private static int[] _mixSigGalaxyStarCounts;
        [ThreadStatic] private static int[] _mixSigBirthStarIds;
        [ThreadStatic] private static int[] _mixSigBirthPlanetIds;
        [ThreadStatic] private static int[] _mixSigStarIds;
        [ThreadStatic] private static int[] _mixSigStarTypes;
        [ThreadStatic] private static int[] _mixSigStarSpectrs;
        [ThreadStatic] private static int[] _mixSigStarPlanetCounts;
        [ThreadStatic] private static int[] _mixSigStarPlanetLoopCounts;
        [ThreadStatic] private static int[] _mixSigStarPosX;
        [ThreadStatic] private static int[] _mixSigStarPosY;
        [ThreadStatic] private static int[] _mixSigStarPosZ;
        [ThreadStatic] private static int[] _mixSigPlanetIds;
        [ThreadStatic] private static int[] _mixSigPlanetTypes;
        [ThreadStatic] private static int[] _mixSigPlanetThemes;
        [ThreadStatic] private static int[] _mixSigPlanetWaterItemIds;
        [ThreadStatic] private static int[] _mixSigPlanetOrbitIndexes;
        [ThreadStatic] private static int[] _mixSigPlanetOrbitArounds;
        [ThreadStatic] private static int[] _mixSigPlanetIsNull;
        [ThreadStatic] private static int[] _mixSigPlanetIsGas;
        [ThreadStatic] private static int[] _mixSigPlanetVeinOffsets;
        [ThreadStatic] private static ulong[] _mixSigOutGalaxy;
        [ThreadStatic] private static ulong[] _mixSigOutPlanet;
        [ThreadStatic] private static ulong[] _mixSigOutVein;
        [ThreadStatic] private static ulong[] _mixSigOutPipeline;
        private static long _perfSingleCalls;
        private static long _perfSingleTicks;
        private static long _perfBatchCalls;
        private static long _perfBatchTicks;
        private static long _perfBatchSeeds;
        private static long _perfFailCalls;
        private static long _perfMixSeedSigCalls;
        private static long _perfMixSeedSigTicks;
        private static long _perfMixSeedSigSeeds;
        private static long _perfMixSeedSigFailCalls;

        internal struct PerfStats
        {
            public long singleCalls;
            public double singleMs;
            public long batchCalls;
            public double batchMs;
            public long batchSeeds;
            public long failCalls;
            public long mixSeedSigCalls;
            public double mixSeedSigMs;
            public long mixSeedSigSeeds;
            public long mixSeedSigFailCalls;
        }

        private sealed class MixThemeFlatData
        {
            public int count;
            public int[] ids;
            public int[] planetTypes;
            public float[] temperatures;
            public int[] distributes;
            public int[] waterItemIds;
            public int[] veinOffsets;
            public int[] veinValues;
            public int[] rareOffsets;
            public int[] rareValues;
            public int[] rareSettingsOffsets;
            public float[] rareSettingsValues;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct NativeVec3d
        {
            public double x;
            public double y;
            public double z;
        }

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "dsp_cuda_generate_temp_poses_params_fp64")]
        private static extern int NativeGenerateTempPosesParamsFp64(
            int seed,
            int maxCount,
            double minDist,
            double minStepLen,
            double maxStepLen,
            double flatten,
            int collisionFp64,
            int deviceId,
            [Out] NativeVec3d[] outPoses,
            int outCapacity,
            out int outCount);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "dsp_cuda_generate_temp_poses_params_fp64_batch")]
        private static extern int NativeGenerateTempPosesParamsFp64Batch(
            [In] int[] seeds,
            int seedCount,
            int maxCount,
            double minDist,
            double minStepLen,
            double maxStepLen,
            double flatten,
            int collisionFp64,
            int deviceId,
            [Out] NativeVec3d[] outPoses,
            int outStride,
            [Out] int[] outCounts);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "dsp_cuda_debug_rng_nextdouble")]
        private static extern int NativeDebugRngNextDouble(
            int seed,
            int count,
            int deviceId,
            [Out] double[] outValues);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "dsp_cuda_debug_rng_state_after_ctor")]
        private static extern int NativeDebugRngStateAfterCtor(
            int seed,
            int deviceId,
            [Out] int[] outSeedArray56,
            out int outInext,
            out int outInextp);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "dsp_cuda_mix_chunk_reduce_signatures")]
        private static extern int NativeMixChunkReduceSignatures(
            int seedCount,
            [In] int[] seedStarOffsets,
            [In] int[] seedPlanetOffsets,
            [In] int[] galaxyStarCounts,
            [In] int[] birthStarIds,
            [In] int[] birthPlanetIds,
            [In] int[] starIds,
            [In] int[] starTypes,
            [In] int[] starSpectrs,
            [In] int[] starPlanetCounts,
            [In] int[] starPlanetLoopCounts,
            [In] int[] starPosX,
            [In] int[] starPosY,
            [In] int[] starPosZ,
            [In] int[] planetIds,
            [In] int[] planetTypes,
            [In] int[] planetThemes,
            [In] int[] planetWaterItemIds,
            [In] int[] planetOrbitIndexes,
            [In] int[] planetOrbitArounds,
            [In] int[] planetIsNull,
            [In] int[] planetIsGas,
            [In] int[] planetVeinOffsets,
            [In] int[] veinCountsFlat,
            int veinStride,
            [Out] ulong[] outGalaxySigs,
            [Out] ulong[] outPlanetSigs,
            [Out] ulong[] outVeinSigs,
            [Out] ulong[] outPipelineSigs);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "dsp_cuda_mix_signatures_from_seeds_f32")]
        private static extern int NativeMixSignaturesFromSeedsF32(
            [In] int[] galaxySeeds,
            int seedCount,
            int starCount,
            int collisionFp64,
            int useFp32ProbCompare,
            int veinLen,
            int deviceId,
            int starTypeMainSeq,
            int starTypeGiant,
            int starTypeWhiteDwarf,
            int starTypeNeutronStar,
            int starTypeBlackHole,
            int spectrM,
            int spectrK,
            int spectrG,
            int spectrF,
            int spectrA,
            int spectrB,
            int spectrO,
            int spectrX,
            int planetTypeGas,
            int planetTypeOcean,
            int planetTypeVocano,
            int planetTypeDesert,
            int planetTypeIce,
            int themeDistributeDefault,
            int themeDistributeBirth,
            int themeDistributeInterstellar,
            [In] int[] themeIds,
            [In] int[] themePlanetTypes,
            [In] float[] themeTemperatures,
            [In] int[] themeDistributes,
            [In] int[] themeWaterItemIds,
            [In] int[] themeVeinSpotOffsets,
            [In] int[] themeVeinSpotValues,
            [In] int[] themeRareVeinOffsets,
            [In] int[] themeRareVeinValues,
            [In] int[] themeRareSettingsOffsets,
            [In] float[] themeRareSettingsValues,
            int themeCount,
            [Out] ulong[] outGalaxySigs,
            [Out] ulong[] outPlanetSigs,
            [Out] ulong[] outVeinSigs,
            [Out] ulong[] outPipelineSigs);

        public static void EnableByCli(bool enabled)
        {
            _enabledByCli = enabled;
        }

        public static bool IsEnabled()
        {
            if (_enabledByCli)
                return true;

            var env = Environment.GetEnvironmentVariable("DSP_USE_CUDA_GALAXY");
            return string.Equals(env, "1", StringComparison.Ordinal);
        }

        public static bool IsDebugPoseDiffEnabled()
        {
            var env = Environment.GetEnvironmentVariable("DSP_DEBUG_CUDA_POSES");
            return string.Equals(env, "1", StringComparison.Ordinal);
        }

        public static bool IsDebugRngEnabled()
        {
            var env = Environment.GetEnvironmentVariable("DSP_DEBUG_CUDA_RNG");
            return string.Equals(env, "1", StringComparison.Ordinal);
        }

        public static bool IsDebugRngStateEnabled()
        {
            var env = Environment.GetEnvironmentVariable("DSP_DEBUG_CUDA_RNG_STATE");
            return string.Equals(env, "1", StringComparison.Ordinal);
        }

        public static void ResetPerfStats()
        {
            Interlocked.Exchange(ref _perfSingleCalls, 0);
            Interlocked.Exchange(ref _perfSingleTicks, 0);
            Interlocked.Exchange(ref _perfBatchCalls, 0);
            Interlocked.Exchange(ref _perfBatchTicks, 0);
            Interlocked.Exchange(ref _perfBatchSeeds, 0);
            Interlocked.Exchange(ref _perfFailCalls, 0);
            Interlocked.Exchange(ref _perfMixSeedSigCalls, 0);
            Interlocked.Exchange(ref _perfMixSeedSigTicks, 0);
            Interlocked.Exchange(ref _perfMixSeedSigSeeds, 0);
            Interlocked.Exchange(ref _perfMixSeedSigFailCalls, 0);
        }

        public static PerfStats GetPerfStats()
        {
            double toMs = 1000.0 / Stopwatch.Frequency;
            return new PerfStats
            {
                singleCalls = Interlocked.Read(ref _perfSingleCalls),
                singleMs = Interlocked.Read(ref _perfSingleTicks) * toMs,
                batchCalls = Interlocked.Read(ref _perfBatchCalls),
                batchMs = Interlocked.Read(ref _perfBatchTicks) * toMs,
                batchSeeds = Interlocked.Read(ref _perfBatchSeeds),
                failCalls = Interlocked.Read(ref _perfFailCalls),
                mixSeedSigCalls = Interlocked.Read(ref _perfMixSeedSigCalls),
                mixSeedSigMs = Interlocked.Read(ref _perfMixSeedSigTicks) * toMs,
                mixSeedSigSeeds = Interlocked.Read(ref _perfMixSeedSigSeeds),
                mixSeedSigFailCalls = Interlocked.Read(ref _perfMixSeedSigFailCalls)
            };
        }

        public static bool TryGenerateRandomPosesParamsFp64(
            int seed,
            int maxCount,
            double minDist,
            double minStepLen,
            double maxStepLen,
            double flatten,
            bool collisionFp64,
            List<VectorLF3> tmpPoses,
            out int generatedCount)
        {
            generatedCount = 0;

            if (!IsEnabled())
                return false;
            if (_nativeBroken)
                return false;
            if (tmpPoses == null)
                return false;
            if (maxCount <= 0)
                return false;

            long t0 = Stopwatch.GetTimestamp();
            Interlocked.Increment(ref _perfSingleCalls);
            var nativePoses = EnsureSinglePoseScratch(maxCount);
            int deviceId = GetDeviceIdFromEnv();
            bool effectiveCollisionFp64 = collisionFp64 || ForceCollisionFp64FromEnv();
            if (effectiveCollisionFp64 && !collisionFp64 && !_printedForceColl64)
            {
                _printedForceColl64 = true;
                Console.WriteLine("[cuda-galaxy] force collisionFp64 by env DSP_CUDA_FORCE_COLL64=1");
            }

            int rc;
            int outCount;
            try
            {
                rc = NativeGenerateTempPosesParamsFp64(
                    seed,
                    maxCount,
                    minDist,
                    minStepLen,
                    maxStepLen,
                    flatten,
                    effectiveCollisionFp64 ? 1 : 0,
                    deviceId,
                    nativePoses,
                    nativePoses.Length,
                    out outCount);
            }
            catch (Exception ex) when (ex is DllNotFoundException || ex is EntryPointNotFoundException || ex is BadImageFormatException)
            {
                Interlocked.Add(ref _perfSingleTicks, Stopwatch.GetTimestamp() - t0);
                Interlocked.Increment(ref _perfFailCalls);
                MarkNativeBroken(ex.GetType().Name + ": " + ex.Message);
                return false;
            }

            if (rc != Ok)
            {
                Interlocked.Add(ref _perfSingleTicks, Stopwatch.GetTimestamp() - t0);
                Interlocked.Increment(ref _perfFailCalls);
                MarkNativeBroken("native return code=" + rc);
                return false;
            }

            if (outCount < 0 || outCount > maxCount)
            {
                Interlocked.Add(ref _perfSingleTicks, Stopwatch.GetTimestamp() - t0);
                Interlocked.Increment(ref _perfFailCalls);
                MarkNativeBroken("invalid outCount=" + outCount);
                return false;
            }

            tmpPoses.Clear();
            for (int i = 0; i < outCount; ++i)
            {
                var p = nativePoses[i];
                tmpPoses.Add(new VectorLF3(p.x, p.y, p.z));
            }

            generatedCount = outCount;
            Interlocked.Add(ref _perfSingleTicks, Stopwatch.GetTimestamp() - t0);
            return true;
        }

        public static bool TryGenerateRandomPosesParamsFp64Batch(
            int[] seeds,
            int maxCount,
            double minDist,
            double minStepLen,
            double maxStepLen,
            double flatten,
            bool collisionFp64,
            out NativeVec3d[] poses,
            out int[] counts,
            out int outStride)
        {
            poses = null;
            counts = null;
            outStride = 0;

            if (!IsEnabled())
                return false;
            if (seeds == null || seeds.Length == 0)
                return false;
            if (_nativeBroken)
                return false;
            if (maxCount <= 0)
                return false;

            outStride = maxCount;
            poses = new NativeVec3d[seeds.Length * outStride];
            counts = new int[seeds.Length];

            bool ok = TryGenerateRandomPosesParamsFp64BatchInto(
                seeds,
                seeds.Length,
                maxCount,
                minDist,
                minStepLen,
                maxStepLen,
                flatten,
                collisionFp64,
                poses,
                outStride,
                counts);
            if (!ok)
            {
                poses = null;
                counts = null;
                outStride = 0;
                return false;
            }
            return true;
        }

        public static bool TryGenerateRandomPosesParamsFp64BatchInto(
            int[] seeds,
            int seedCount,
            int maxCount,
            double minDist,
            double minStepLen,
            double maxStepLen,
            double flatten,
            bool collisionFp64,
            NativeVec3d[] poses,
            int outStride,
            int[] counts)
        {
            if (!IsEnabled())
                return false;
            if (seeds == null || poses == null || counts == null)
                return false;
            if (seedCount <= 0 || seedCount > seeds.Length)
                return false;
            if (maxCount <= 0 || outStride < maxCount)
                return false;
            if (counts.Length < seedCount)
                return false;
            if (poses.Length < seedCount * outStride)
                return false;
            if (_nativeBroken)
                return false;

            long t0 = Stopwatch.GetTimestamp();
            Interlocked.Increment(ref _perfBatchCalls);
            Interlocked.Add(ref _perfBatchSeeds, seedCount);
            int deviceId = GetDeviceIdFromEnv();
            bool effectiveCollisionFp64 = collisionFp64 || ForceCollisionFp64FromEnv();
            if (effectiveCollisionFp64 && !collisionFp64 && !_printedForceColl64)
            {
                _printedForceColl64 = true;
                Console.WriteLine("[cuda-galaxy] force collisionFp64 by env DSP_CUDA_FORCE_COLL64=1");
            }

            int rc;
            try
            {
                rc = NativeGenerateTempPosesParamsFp64Batch(
                    seeds,
                    seedCount,
                    maxCount,
                    minDist,
                    minStepLen,
                    maxStepLen,
                    flatten,
                    effectiveCollisionFp64 ? 1 : 0,
                    deviceId,
                    poses,
                    outStride,
                    counts);
            }
            catch (Exception ex) when (ex is DllNotFoundException || ex is EntryPointNotFoundException || ex is BadImageFormatException)
            {
                Interlocked.Add(ref _perfBatchTicks, Stopwatch.GetTimestamp() - t0);
                Interlocked.Increment(ref _perfFailCalls);
                MarkNativeBroken(ex.GetType().Name + ": " + ex.Message);
                return false;
            }

            if (rc != Ok)
            {
                Interlocked.Add(ref _perfBatchTicks, Stopwatch.GetTimestamp() - t0);
                Interlocked.Increment(ref _perfFailCalls);
                MarkNativeBroken("batch native return code=" + rc);
                return false;
            }

            Interlocked.Add(ref _perfBatchTicks, Stopwatch.GetTimestamp() - t0);
            return true;
        }

        public static bool TryReduceMixChunkSignatures(
            GalaxyData[] galaxies,
            int galaxyCount,
            int[] veinCountsFlat,
            int veinStride,
            int[] galaxyPlanetStartIdx,
            out ulong[] outGalaxySigs,
            out ulong[] outPlanetSigs,
            out ulong[] outVeinSigs,
            out ulong[] outPipelineSigs)
        {
            outGalaxySigs = null;
            outPlanetSigs = null;
            outVeinSigs = null;
            outPipelineSigs = null;

            if (!IsEnabled() || _nativeBroken || _mixSigEntryMissing)
                return false;
            if (galaxies == null || galaxyCount <= 0 || galaxyCount > galaxies.Length)
                return false;
            if (veinCountsFlat == null || veinStride <= 1 || galaxyPlanetStartIdx == null || galaxyPlanetStartIdx.Length < galaxyCount)
                return false;

            int totalStars = 0;
            int totalPlanets = 0;
            for (int gi = 0; gi < galaxyCount; ++gi)
            {
                var g = galaxies[gi];
                if (g == null || g.stars == null)
                    return false;

                totalStars += g.stars.Length;
                for (int si = 0; si < g.stars.Length; ++si)
                {
                    var s = g.stars[si];
                    if (s == null || s.planets == null)
                        continue;
                    totalPlanets += s.planets.Length;
                }
            }

            EnsureIntBuffer(ref _mixSigSeedStarOffsets, galaxyCount + 1);
            EnsureIntBuffer(ref _mixSigSeedPlanetOffsets, galaxyCount + 1);
            EnsureIntBuffer(ref _mixSigGalaxyStarCounts, galaxyCount);
            EnsureIntBuffer(ref _mixSigBirthStarIds, galaxyCount);
            EnsureIntBuffer(ref _mixSigBirthPlanetIds, galaxyCount);

            EnsureIntBuffer(ref _mixSigStarIds, totalStars);
            EnsureIntBuffer(ref _mixSigStarTypes, totalStars);
            EnsureIntBuffer(ref _mixSigStarSpectrs, totalStars);
            EnsureIntBuffer(ref _mixSigStarPlanetCounts, totalStars);
            EnsureIntBuffer(ref _mixSigStarPlanetLoopCounts, totalStars);
            EnsureIntBuffer(ref _mixSigStarPosX, totalStars);
            EnsureIntBuffer(ref _mixSigStarPosY, totalStars);
            EnsureIntBuffer(ref _mixSigStarPosZ, totalStars);

            EnsureIntBuffer(ref _mixSigPlanetIds, totalPlanets);
            EnsureIntBuffer(ref _mixSigPlanetTypes, totalPlanets);
            EnsureIntBuffer(ref _mixSigPlanetThemes, totalPlanets);
            EnsureIntBuffer(ref _mixSigPlanetWaterItemIds, totalPlanets);
            EnsureIntBuffer(ref _mixSigPlanetOrbitIndexes, totalPlanets);
            EnsureIntBuffer(ref _mixSigPlanetOrbitArounds, totalPlanets);
            EnsureIntBuffer(ref _mixSigPlanetIsNull, totalPlanets);
            EnsureIntBuffer(ref _mixSigPlanetIsGas, totalPlanets);
            EnsureIntBuffer(ref _mixSigPlanetVeinOffsets, totalPlanets);

            EnsureUlongBuffer(ref _mixSigOutGalaxy, galaxyCount);
            EnsureUlongBuffer(ref _mixSigOutPlanet, galaxyCount);
            EnsureUlongBuffer(ref _mixSigOutVein, galaxyCount);
            EnsureUlongBuffer(ref _mixSigOutPipeline, galaxyCount);

            int starCursor = 0;
            int planetCursor = 0;
            for (int gi = 0; gi < galaxyCount; ++gi)
            {
                var g = galaxies[gi];
                _mixSigSeedStarOffsets[gi] = starCursor;
                _mixSigSeedPlanetOffsets[gi] = planetCursor;
                _mixSigGalaxyStarCounts[gi] = g.starCount;
                _mixSigBirthStarIds[gi] = g.birthStarId;
                _mixSigBirthPlanetIds[gi] = g.birthPlanetId;

                int solidOffset = galaxyPlanetStartIdx[gi];
                for (int si = 0; si < g.stars.Length; ++si)
                {
                    var s = g.stars[si];
                    if (s == null)
                    {
                        _mixSigStarIds[starCursor] = -1;
                        _mixSigStarTypes[starCursor] = 0;
                        _mixSigStarSpectrs[starCursor] = 0;
                        _mixSigStarPlanetCounts[starCursor] = 0;
                        _mixSigStarPlanetLoopCounts[starCursor] = 0;
                        _mixSigStarPosX[starCursor] = 0;
                        _mixSigStarPosY[starCursor] = 0;
                        _mixSigStarPosZ[starCursor] = 0;
                        starCursor++;
                        continue;
                    }

                    _mixSigStarIds[starCursor] = s.id;
                    _mixSigStarTypes[starCursor] = (int)s.type;
                    _mixSigStarSpectrs[starCursor] = (int)s.spectr;
                    _mixSigStarPlanetCounts[starCursor] = s.planetCount;
                    _mixSigStarPlanetLoopCounts[starCursor] = s.planets != null ? s.planets.Length : 0;
                    _mixSigStarPosX[starCursor] = (int)Math.Round(s.uPosition.x * 0.001);
                    _mixSigStarPosY[starCursor] = (int)Math.Round(s.uPosition.y * 0.001);
                    _mixSigStarPosZ[starCursor] = (int)Math.Round(s.uPosition.z * 0.001);
                    starCursor++;

                    var planets = s.planets;
                    if (planets == null)
                        continue;
                    for (int pi = 0; pi < planets.Length; ++pi)
                    {
                        var p = planets[pi];
                        if (p == null)
                        {
                            _mixSigPlanetIds[planetCursor] = 0;
                            _mixSigPlanetTypes[planetCursor] = 0;
                            _mixSigPlanetThemes[planetCursor] = 0;
                            _mixSigPlanetWaterItemIds[planetCursor] = 0;
                            _mixSigPlanetOrbitIndexes[planetCursor] = 0;
                            _mixSigPlanetOrbitArounds[planetCursor] = 0;
                            _mixSigPlanetIsNull[planetCursor] = 1;
                            _mixSigPlanetIsGas[planetCursor] = 0;
                            _mixSigPlanetVeinOffsets[planetCursor] = -1;
                            planetCursor++;
                            continue;
                        }

                        bool isGas = p.type == EPlanetType.Gas;
                        _mixSigPlanetIds[planetCursor] = p.id;
                        _mixSigPlanetTypes[planetCursor] = (int)p.type;
                        _mixSigPlanetThemes[planetCursor] = p.theme;
                        _mixSigPlanetWaterItemIds[planetCursor] = p.waterItemId;
                        _mixSigPlanetOrbitIndexes[planetCursor] = p.orbitIndex;
                        _mixSigPlanetOrbitArounds[planetCursor] = p.orbitAround;
                        _mixSigPlanetIsNull[planetCursor] = 0;
                        _mixSigPlanetIsGas[planetCursor] = isGas ? 1 : 0;
                        if (isGas)
                        {
                            _mixSigPlanetVeinOffsets[planetCursor] = -1;
                        }
                        else
                        {
                            _mixSigPlanetVeinOffsets[planetCursor] = solidOffset;
                            solidOffset++;
                        }
                        planetCursor++;
                    }
                }
            }
            _mixSigSeedStarOffsets[galaxyCount] = starCursor;
            _mixSigSeedPlanetOffsets[galaxyCount] = planetCursor;

            int rc;
            try
            {
                rc = NativeMixChunkReduceSignatures(
                    galaxyCount,
                    _mixSigSeedStarOffsets,
                    _mixSigSeedPlanetOffsets,
                    _mixSigGalaxyStarCounts,
                    _mixSigBirthStarIds,
                    _mixSigBirthPlanetIds,
                    _mixSigStarIds,
                    _mixSigStarTypes,
                    _mixSigStarSpectrs,
                    _mixSigStarPlanetCounts,
                    _mixSigStarPlanetLoopCounts,
                    _mixSigStarPosX,
                    _mixSigStarPosY,
                    _mixSigStarPosZ,
                    _mixSigPlanetIds,
                    _mixSigPlanetTypes,
                    _mixSigPlanetThemes,
                    _mixSigPlanetWaterItemIds,
                    _mixSigPlanetOrbitIndexes,
                    _mixSigPlanetOrbitArounds,
                    _mixSigPlanetIsNull,
                    _mixSigPlanetIsGas,
                    _mixSigPlanetVeinOffsets,
                    veinCountsFlat,
                    veinStride,
                    _mixSigOutGalaxy,
                    _mixSigOutPlanet,
                    _mixSigOutVein,
                    _mixSigOutPipeline);
            }
            catch (EntryPointNotFoundException)
            {
                _mixSigEntryMissing = true;
                return false;
            }
            catch (Exception ex) when (ex is DllNotFoundException || ex is BadImageFormatException)
            {
                MarkNativeBroken(ex.GetType().Name + ": " + ex.Message);
                return false;
            }

            if (rc != Ok)
            {
                MarkNativeBroken("mix-signature native return code=" + rc);
                return false;
            }

            outGalaxySigs = _mixSigOutGalaxy;
            outPlanetSigs = _mixSigOutPlanet;
            outVeinSigs = _mixSigOutVein;
            outPipelineSigs = _mixSigOutPipeline;
            return true;
        }

        public static bool TryEvalMixSignaturesFromSeedRange(
            int seedStart,
            int seedCount,
            int starCount,
            bool collisionFp64,
            bool useFp32ProbCompare,
            out ulong[] outGalaxySigs,
            out ulong[] outPlanetSigs,
            out ulong[] outVeinSigs,
            out ulong[] outPipelineSigs)
        {
            outGalaxySigs = null;
            outPlanetSigs = null;
            outVeinSigs = null;
            outPipelineSigs = null;

            if (!IsEnabled() || _nativeBroken || _mixSeedSigEntryMissing)
                return false;
            if (seedCount <= 0 || starCount <= 0)
                return false;

            if (!TryGetMixThemeFlatData(out var themes))
                return false;

            int veinLen = global::DspFindSeed.PlanetModelingManager.veinProtos != null
                ? global::DspFindSeed.PlanetModelingManager.veinProtos.Length
                : 0;
            veinLen = Math.Min(veinLen, MaxOutVeinLen);
            if (veinLen <= 1)
                return false;

            EnsureIntBuffer(ref _mixSeedSigSeeds, seedCount);
            for (int i = 0; i < seedCount; ++i)
                _mixSeedSigSeeds[i] = seedStart + i;

            EnsureUlongBuffer(ref _mixSigOutGalaxy, seedCount);
            EnsureUlongBuffer(ref _mixSigOutPlanet, seedCount);
            EnsureUlongBuffer(ref _mixSigOutVein, seedCount);
            EnsureUlongBuffer(ref _mixSigOutPipeline, seedCount);

            int deviceId = GetDeviceIdFromEnv();
            long t0 = Stopwatch.GetTimestamp();
            Interlocked.Increment(ref _perfMixSeedSigCalls);
            Interlocked.Add(ref _perfMixSeedSigSeeds, seedCount);

            int rc;
            try
            {
                rc = NativeMixSignaturesFromSeedsF32(
                    _mixSeedSigSeeds,
                    seedCount,
                    starCount,
                    collisionFp64 ? 1 : 0,
                    useFp32ProbCompare ? 1 : 0,
                    veinLen,
                    deviceId,
                    (int)EStarType.MainSeqStar,
                    (int)EStarType.GiantStar,
                    (int)EStarType.WhiteDwarf,
                    (int)EStarType.NeutronStar,
                    (int)EStarType.BlackHole,
                    (int)ESpectrType.M,
                    (int)ESpectrType.K,
                    (int)ESpectrType.G,
                    (int)ESpectrType.F,
                    (int)ESpectrType.A,
                    (int)ESpectrType.B,
                    (int)ESpectrType.O,
                    (int)ESpectrType.X,
                    (int)EPlanetType.Gas,
                    (int)EPlanetType.Ocean,
                    (int)EPlanetType.Vocano,
                    (int)EPlanetType.Desert,
                    (int)EPlanetType.Ice,
                    (int)EThemeDistribute.Default,
                    (int)EThemeDistribute.Birth,
                    (int)EThemeDistribute.Interstellar,
                    themes.ids,
                    themes.planetTypes,
                    themes.temperatures,
                    themes.distributes,
                    themes.waterItemIds,
                    themes.veinOffsets,
                    themes.veinValues,
                    themes.rareOffsets,
                    themes.rareValues,
                    themes.rareSettingsOffsets,
                    themes.rareSettingsValues,
                    themes.count,
                    _mixSigOutGalaxy,
                    _mixSigOutPlanet,
                    _mixSigOutVein,
                    _mixSigOutPipeline);
            }
            catch (EntryPointNotFoundException)
            {
                Interlocked.Add(ref _perfMixSeedSigTicks, Stopwatch.GetTimestamp() - t0);
                Interlocked.Increment(ref _perfMixSeedSigFailCalls);
                _mixSeedSigEntryMissing = true;
                return false;
            }
            catch (Exception ex) when (ex is DllNotFoundException || ex is BadImageFormatException)
            {
                Interlocked.Add(ref _perfMixSeedSigTicks, Stopwatch.GetTimestamp() - t0);
                Interlocked.Increment(ref _perfMixSeedSigFailCalls);
                MarkNativeBroken(ex.GetType().Name + ": " + ex.Message);
                return false;
            }

            Interlocked.Add(ref _perfMixSeedSigTicks, Stopwatch.GetTimestamp() - t0);
            if (rc != Ok)
            {
                Interlocked.Increment(ref _perfMixSeedSigFailCalls);
                MarkNativeBroken("mix-seed-signature native return code=" + rc);
                return false;
            }

            outGalaxySigs = _mixSigOutGalaxy;
            outPlanetSigs = _mixSigOutPlanet;
            outVeinSigs = _mixSigOutVein;
            outPipelineSigs = _mixSigOutPipeline;
            return true;
        }

        private static bool TryGetMixThemeFlatData(out MixThemeFlatData cache)
        {
            cache = null;
            var themeSet = global::DspFindSeed.LDB.themes;
            if (themeSet == null || themeSet.dataArray == null || themeSet.dataArray.Length == 0)
                return false;
            var themeArray = themeSet.dataArray;

            lock (_mixThemeLock)
            {
                if (_mixThemeCache != null &&
                    ReferenceEquals(_mixThemeProtoArrayRef, themeArray) &&
                    _mixThemeCache.count == themeArray.Length)
                {
                    cache = _mixThemeCache;
                    return true;
                }

                int count = themeArray.Length;
                var ids = new int[count];
                var planetTypes = new int[count];
                var temperatures = new float[count];
                var distributes = new int[count];
                var waterItemIds = new int[count];
                var veinOffsets = new int[count + 1];
                var rareOffsets = new int[count + 1];
                var rareSettingsOffsets = new int[count + 1];
                var veinValues = new List<int>(count * 8);
                var rareValues = new List<int>(count * 4);
                var rareSettingsValues = new List<float>(count * 12);

                for (int i = 0; i < count; ++i)
                {
                    var theme = themeArray[i];
                    if (theme == null)
                        return false;

                    var veinSpot = theme.VeinSpot ?? Array.Empty<int>();
                    var rareVeins = theme.RareVeins ?? Array.Empty<int>();
                    var rareSettings = theme.RareSettings ?? Array.Empty<float>();
                    if (veinSpot.Length > MaxVeinSpotLen)
                        return false;
                    if (rareVeins.Length > MaxRareVeins)
                        return false;
                    if (rareSettings.Length < rareVeins.Length * RareSettingsGroup)
                        return false;

                    ids[i] = theme.ID;
                    planetTypes[i] = (int)theme.PlanetType;
                    temperatures[i] = theme.Temperature;
                    distributes[i] = (int)theme.Distribute;
                    waterItemIds[i] = theme.WaterItemId;

                    veinOffsets[i] = veinValues.Count;
                    for (int j = 0; j < veinSpot.Length; ++j)
                        veinValues.Add(veinSpot[j]);

                    rareOffsets[i] = rareValues.Count;
                    for (int j = 0; j < rareVeins.Length; ++j)
                        rareValues.Add(rareVeins[j]);

                    rareSettingsOffsets[i] = rareSettingsValues.Count;
                    int rareSettingsCount = rareVeins.Length * RareSettingsGroup;
                    for (int j = 0; j < rareSettingsCount; ++j)
                        rareSettingsValues.Add(rareSettings[j]);
                }

                veinOffsets[count] = veinValues.Count;
                rareOffsets[count] = rareValues.Count;
                rareSettingsOffsets[count] = rareSettingsValues.Count;

                _mixThemeCache = new MixThemeFlatData
                {
                    count = count,
                    ids = ids,
                    planetTypes = planetTypes,
                    temperatures = temperatures,
                    distributes = distributes,
                    waterItemIds = waterItemIds,
                    veinOffsets = veinOffsets,
                    veinValues = veinValues.ToArray(),
                    rareOffsets = rareOffsets,
                    rareValues = rareValues.ToArray(),
                    rareSettingsOffsets = rareSettingsOffsets,
                    rareSettingsValues = rareSettingsValues.ToArray()
                };
                _mixThemeProtoArrayRef = themeArray;
                cache = _mixThemeCache;
                return true;
            }
        }

        private static NativeVec3d[] EnsureSinglePoseScratch(int needed)
        {
            if (_singlePoseScratch == null || _singlePoseScratch.Length < needed)
                _singlePoseScratch = new NativeVec3d[needed];
            return _singlePoseScratch;
        }

        private static void EnsureIntBuffer(ref int[] arr, int needed)
        {
            if (arr == null || arr.Length < needed)
                arr = new int[needed];
        }

        private static void EnsureUlongBuffer(ref ulong[] arr, int needed)
        {
            if (arr == null || arr.Length < needed)
                arr = new ulong[needed];
        }

        private static int GetDeviceIdFromEnv()
        {
            var env = Environment.GetEnvironmentVariable("DSP_CUDA_DEVICE");
            if (string.IsNullOrEmpty(env))
                return -1;
            int parsed;
            if (!int.TryParse(env, out parsed))
                return -1;
            return parsed;
        }

        private static bool ForceCollisionFp64FromEnv()
        {
            var env = Environment.GetEnvironmentVariable("DSP_CUDA_FORCE_COLL64");
            return string.Equals(env, "1", StringComparison.Ordinal);
        }

        public static bool TryGetGpuRngSequence(int seed, int count, out double[] values)
        {
            values = null;
            if (count <= 0)
                return false;
            if (_nativeBroken)
                return false;

            int deviceId = GetDeviceIdFromEnv();
            values = new double[count];
            int rc;
            try
            {
                rc = NativeDebugRngNextDouble(seed, count, deviceId, values);
            }
            catch (Exception ex) when (ex is DllNotFoundException || ex is EntryPointNotFoundException || ex is BadImageFormatException)
            {
                MarkNativeBroken(ex.GetType().Name + ": " + ex.Message);
                values = null;
                return false;
            }

            if (rc != Ok)
            {
                MarkNativeBroken("debug rng native return code=" + rc);
                values = null;
                return false;
            }
            return true;
        }

        public static bool TryGetGpuRngStateAfterCtor(int seed, out int[] seedArray56, out int inext, out int inextp)
        {
            seedArray56 = null;
            inext = 0;
            inextp = 0;
            if (_nativeBroken)
                return false;

            int deviceId = GetDeviceIdFromEnv();
            var arr = new int[56];
            int rc;
            try
            {
                rc = NativeDebugRngStateAfterCtor(seed, deviceId, arr, out inext, out inextp);
            }
            catch (Exception ex) when (ex is DllNotFoundException || ex is EntryPointNotFoundException || ex is BadImageFormatException)
            {
                MarkNativeBroken(ex.GetType().Name + ": " + ex.Message);
                return false;
            }

            if (rc != Ok)
            {
                MarkNativeBroken("debug rng state native return code=" + rc);
                return false;
            }

            seedArray56 = arr;
            return true;
        }

        private static void MarkNativeBroken(string reason)
        {
            _nativeBroken = true;
            _nativeBrokenReason = reason;
            if (_printedFallback)
                return;
            _printedFallback = true;
            Console.WriteLine("[cuda-galaxy] fallback to CPU: " + _nativeBrokenReason);
        }
    }
}
