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

        private static bool _enabledByCli;
        private static bool _nativeBroken;
        private static string _nativeBrokenReason;
        private static bool _printedFallback;
        private static bool _printedForceColl64;
        private static bool _mixSigEntryMissing;
        [ThreadStatic] private static NativeVec3d[] _singlePoseScratch;
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

        internal struct PerfStats
        {
            public long singleCalls;
            public double singleMs;
            public long batchCalls;
            public double batchMs;
            public long batchSeeds;
            public long failCalls;
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
                failCalls = Interlocked.Read(ref _perfFailCalls)
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
