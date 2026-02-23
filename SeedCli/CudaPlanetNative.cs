using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;

namespace SeedCli
{
    internal static class CudaPlanetNative
    {
        private const string LibName = "dsp_cuda_galaxy";
        private const int Ok = 0;
        private const int MaxVeinSpotLen = 16;
        private const int MaxRareVeins = 8;
        private const int RareSettingsGroup = 4;
        private const int MaxOutVeinLen = 32;
        private const int SingularityLaySide = 1;
        private const int SingularityTidal1 = 2;
        private const int SingularityTidal2 = 4;
        private const int SingularityTidal4 = 8;
        private const int SingularityClockwise = 16;

        private static bool _enabledByCli;
        private static bool _coreEnabledByCli;
        private static bool _nativeBroken;
        private static string _nativeBrokenReason;
        private static bool _printedFallback;
        private static bool _coreBatchEntryMissing;
        private static bool _coreBatchFallbackPrinted;
        private static bool _mixChunkPlanetEntryMissing;
        private static bool _mixChunkVeinEntryMissing;
        private static readonly ConcurrentQueue<PlanetCoreBatchRequest> _coreBatchQueue = new ConcurrentQueue<PlanetCoreBatchRequest>();
        private static readonly AutoResetEvent _coreBatchWake = new AutoResetEvent(false);
        private static int _coreBatchWorkerStarted;
        private static long _perfCoreReqCount;
        private static long _perfCoreReqWaitTicks;
        private static long _perfCoreBatchCalls;
        private static long _perfCoreBatchItems;
        private static long _perfCoreBatchTicks;
        private static long _perfCoreBatchFallbackCalls;
        private static long _perfCoreSingleCalls;
        private static long _perfCoreSingleTicks;
        private static long _perfVeinBatchCalls;
        private static long _perfVeinBatchPlanets;
        private static long _perfVeinBatchTicks;
        private static long _perfVeinBatchFailCalls;

        [ThreadStatic] private static int[] _planetSeeds;
        [ThreadStatic] private static float[] _pValues;
        [ThreadStatic] private static int[] _bonusCases;
        [ThreadStatic] private static int[] _isBirthStars;
        [ThreadStatic] private static int[] _veinSpotLens;
        [ThreadStatic] private static int[] _rareVeinLens;
        [ThreadStatic] private static int[] _veinSpotValues;
        [ThreadStatic] private static int[] _rareVeinValues;
        [ThreadStatic] private static float[] _rareSettingsValues;
        [ThreadStatic] private static int[] _outCounts;
        [ThreadStatic] private static int[] _coreInfoSeeds;
        [ThreadStatic] private static int[] _coreOrbitArounds;
        [ThreadStatic] private static int[] _coreOrbitIndexes;
        [ThreadStatic] private static int[] _coreGasGiants;
        [ThreadStatic] private static int[] _coreStarIndexes;
        [ThreadStatic] private static int[] _coreGalaxyStarCounts;
        [ThreadStatic] private static int[] _coreGalaxyHabitableCounts;
        [ThreadStatic] private static int[] _coreBoostInclinationNs;
        [ThreadStatic] private static int[] _coreCompactTypeCases;
        [ThreadStatic] private static float[] _coreStarOrbitScalers;
        [ThreadStatic] private static double[] _coreStarMasses;
        [ThreadStatic] private static float[] _coreStarHabitableRadii;
        [ThreadStatic] private static float[] _coreStarLightBalanceRadii;
        [ThreadStatic] private static float[] _coreAroundRealRadii;
        [ThreadStatic] private static float[] _coreAroundOrbitRadii;
        [ThreadStatic] private static double[] _coreAroundOrbitalPeriods;

        [StructLayout(LayoutKind.Sequential)]
        internal struct PlanetCoreF32Out
        {
            public float orbit_radius;
            public float orbit_inclination;
            public float orbit_longitude;
            public double orbital_period;
            public float orbit_phase;
            public float obliquity;
            public double rotation_period;
            public float rotation_phase;
            public float sun_distance;
            public float scale;
            public float habitable_bias;
            public float temperature_bias;
            public float radius;
            public float luminosity;
            public double rand1;
            public double rand2;
            public double rand3;
            public double rand4;
            public double num13;
            public double num14;
            public int theme_seed;
            public int type_case;
            public int singularity_flags;
            public int habitable_count_delta;
            public int precision;
            public int segment;
        }

        internal struct PlanetCoreBatchInput
        {
            public int infoSeed;
            public int orbitAround;
            public int orbitIndex;
            public int gasGiant;
            public int starIndex;
            public int galaxyStarCount;
            public int galaxyHabitableCount;
            public int boostInclinationNs;
            public int compactTypeCase;
            public float starOrbitScaler;
            public double starMass;
            public float starHabitableRadius;
            public float starLightBalanceRadius;
            public float orbitAroundPlanetRealRadius;
            public float orbitAroundPlanetOrbitRadius;
            public double orbitAroundPlanetOrbitalPeriod;
        }

        internal struct PlanetCoreBatchArrays
        {
            public int[] infoSeeds;
            public int[] orbitArounds;
            public int[] orbitIndexes;
            public int[] gasGiants;
            public int[] starIndexes;
            public int[] galaxyStarCounts;
            public int[] galaxyHabitableCounts;
            public int[] boostInclinationNs;
            public int[] compactTypeCases;
            public float[] starOrbitScalers;
            public double[] starMasses;
            public float[] starHabitableRadii;
            public float[] starLightBalanceRadii;
            public float[] aroundRealRadii;
            public float[] aroundOrbitRadii;
            public double[] aroundOrbitalPeriods;
        }

        private sealed class PlanetCoreBatchRequest
        {
            public int infoSeed;
            public int orbitAround;
            public int orbitIndex;
            public int gasGiant;
            public int starIndex;
            public int galaxyStarCount;
            public int galaxyHabitableCount;
            public int boostInclinationNs;
            public int compactTypeCase;
            public float starOrbitScaler;
            public double starMass;
            public float starHabitableRadius;
            public float starLightBalanceRadius;
            public float orbitAroundPlanetRealRadius;
            public float orbitAroundPlanetOrbitRadius;
            public double orbitAroundPlanetOrbitalPeriod;
            public PlanetCoreF32Out core;
            public bool success;
            public ManualResetEventSlim done = new ManualResetEventSlim(false);
        }

        internal struct PerfStats
        {
            public long coreReqCount;
            public double coreReqWaitMs;
            public long coreBatchCalls;
            public long coreBatchItems;
            public double coreBatchMs;
            public long coreBatchFallbackCalls;
            public long coreSingleCalls;
            public double coreSingleMs;
            public long veinBatchCalls;
            public long veinBatchPlanets;
            public double veinBatchMs;
            public long veinBatchFailCalls;
        }

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "dsp_cuda_refresh_planet_vein_spots_batch")]
        private static extern int NativeRefreshPlanetVeinSpotsBatch(
            [In] int[] planetSeeds,
            [In] float[] pValues,
            [In] int[] bonusCases,
            [In] int[] isBirthStars,
            [In] int[] veinSpotLens,
            [In] int[] rareVeinLens,
            [In] int[] veinSpotValues,
            int veinSpotStride,
            [In] int[] rareVeinValues,
            int rareVeinStride,
            [In] float[] rareSettingsValues,
            int rareSettingsStride,
            int planetCount,
            int outVeinLen,
            int useFp32ProbCompare,
            int deviceId,
            [Out] int[] outCounts);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "dsp_cuda_mix_chunk_eval_veins_f32")]
        private static extern int NativeMixChunkEvalVeinsF32(
            [In] int[] planetSeeds,
            [In] float[] pValues,
            [In] int[] bonusCases,
            [In] int[] isBirthStars,
            [In] int[] veinSpotLens,
            [In] int[] rareVeinLens,
            [In] int[] veinSpotValues,
            int veinSpotStride,
            [In] int[] rareVeinValues,
            int rareVeinStride,
            [In] float[] rareSettingsValues,
            int rareSettingsStride,
            int planetCount,
            int outVeinLen,
            int useFp32ProbCompare,
            int deviceId,
            [Out] int[] outCounts);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "dsp_cuda_planet_eval_core_f32")]
        private static extern int NativePlanetEvalCoreF32(
            int infoSeed,
            int orbitAround,
            int orbitIndex,
            int gasGiant,
            int starIndex,
            int galaxyStarCount,
            int galaxyHabitableCount,
            int boostInclinationNs,
            int compactTypeCase,
            float starOrbitScaler,
            double starMass,
            float starHabitableRadius,
            float starLightBalanceRadius,
            float orbitAroundPlanetRealRadius,
            float orbitAroundPlanetOrbitRadius,
            double orbitAroundPlanetOrbitalPeriod,
            int deviceId,
            out PlanetCoreF32Out outResult);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "dsp_cuda_planet_eval_core_f32_batch")]
        private static extern int NativePlanetEvalCoreF32Batch(
            [In] int[] infoSeeds,
            [In] int[] orbitArounds,
            [In] int[] orbitIndexes,
            [In] int[] gasGiants,
            [In] int[] starIndexes,
            [In] int[] galaxyStarCounts,
            [In] int[] galaxyHabitableCounts,
            [In] int[] boostInclinationNs,
            [In] int[] compactTypeCases,
            [In] float[] starOrbitScalers,
            [In] double[] starMasses,
            [In] float[] starHabitableRadii,
            [In] float[] starLightBalanceRadii,
            [In] float[] orbitAroundPlanetRealRadii,
            [In] float[] orbitAroundPlanetOrbitRadii,
            [In] double[] orbitAroundPlanetOrbitalPeriods,
            int batchCount,
            int deviceId,
            [Out] PlanetCoreF32Out[] outResults);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "dsp_cuda_mix_chunk_eval_planets_f32")]
        private static extern int NativeMixChunkEvalPlanetsF32(
            [In] int[] infoSeeds,
            [In] int[] orbitArounds,
            [In] int[] orbitIndexes,
            [In] int[] gasGiants,
            [In] int[] starIndexes,
            [In] int[] galaxyStarCounts,
            [In] int[] galaxyHabitableCounts,
            [In] int[] boostInclinationNs,
            [In] int[] compactTypeCases,
            [In] float[] starOrbitScalers,
            [In] double[] starMasses,
            [In] float[] starHabitableRadii,
            [In] float[] starLightBalanceRadii,
            [In] float[] orbitAroundPlanetRealRadii,
            [In] float[] orbitAroundPlanetOrbitRadii,
            [In] double[] orbitAroundPlanetOrbitalPeriods,
            int batchCount,
            int deviceId,
            [Out] PlanetCoreF32Out[] outResults);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "dsp_cuda_planet_eval_gas_details_f32")]
        private static extern int NativePlanetEvalGasDetailsF32(
            int themeSeed,
            float gasCoef,
            float resourceCoef,
            [In] float[] inGasSpeeds,
            [In] float[] inGasHeatValues,
            int gasLen,
            int deviceId,
            [Out] float[] outGasSpeeds,
            out double outTotalHeat);

        public static void EnableByCli(bool enabled)
        {
            _enabledByCli = enabled;
        }

        public static void EnableCoreByCli(bool enabled)
        {
            _coreEnabledByCli = enabled;
        }

        public static bool IsEnabled()
        {
            if (_enabledByCli)
                return true;
            var env = Environment.GetEnvironmentVariable("DSP_USE_CUDA_PLANET");
            return string.Equals(env, "1", StringComparison.Ordinal);
        }

        public static bool IsCoreEnabled()
        {
            if (_coreEnabledByCli)
                return true;
            var env = Environment.GetEnvironmentVariable("DSP_USE_CUDA_PLANET_CORE");
            return string.Equals(env, "1", StringComparison.Ordinal);
        }

        public static void ResetPerfStats()
        {
            Interlocked.Exchange(ref _perfCoreReqCount, 0);
            Interlocked.Exchange(ref _perfCoreReqWaitTicks, 0);
            Interlocked.Exchange(ref _perfCoreBatchCalls, 0);
            Interlocked.Exchange(ref _perfCoreBatchItems, 0);
            Interlocked.Exchange(ref _perfCoreBatchTicks, 0);
            Interlocked.Exchange(ref _perfCoreBatchFallbackCalls, 0);
            Interlocked.Exchange(ref _perfCoreSingleCalls, 0);
            Interlocked.Exchange(ref _perfCoreSingleTicks, 0);
            Interlocked.Exchange(ref _perfVeinBatchCalls, 0);
            Interlocked.Exchange(ref _perfVeinBatchPlanets, 0);
            Interlocked.Exchange(ref _perfVeinBatchTicks, 0);
            Interlocked.Exchange(ref _perfVeinBatchFailCalls, 0);
        }

        public static PerfStats GetPerfStats()
        {
            double toMs = 1000.0 / Stopwatch.Frequency;
            return new PerfStats
            {
                coreReqCount = Interlocked.Read(ref _perfCoreReqCount),
                coreReqWaitMs = Interlocked.Read(ref _perfCoreReqWaitTicks) * toMs,
                coreBatchCalls = Interlocked.Read(ref _perfCoreBatchCalls),
                coreBatchItems = Interlocked.Read(ref _perfCoreBatchItems),
                coreBatchMs = Interlocked.Read(ref _perfCoreBatchTicks) * toMs,
                coreBatchFallbackCalls = Interlocked.Read(ref _perfCoreBatchFallbackCalls),
                coreSingleCalls = Interlocked.Read(ref _perfCoreSingleCalls),
                coreSingleMs = Interlocked.Read(ref _perfCoreSingleTicks) * toMs,
                veinBatchCalls = Interlocked.Read(ref _perfVeinBatchCalls),
                veinBatchPlanets = Interlocked.Read(ref _perfVeinBatchPlanets),
                veinBatchMs = Interlocked.Read(ref _perfVeinBatchTicks) * toMs,
                veinBatchFailCalls = Interlocked.Read(ref _perfVeinBatchFailCalls)
            };
        }

        public static bool TryRefreshPlanetVeinSpotsBatch(
            IList<PlanetData> planets,
            bool useFp32ProbCompare,
            out int[] countsFlat,
            out int outVeinLen)
        {
            countsFlat = null;
            outVeinLen = 0;

            if (!IsEnabled())
                return false;
            if (_nativeBroken)
                return false;
            if (planets == null || planets.Count == 0)
                return false;

            int veinLen = global::DspFindSeed.PlanetModelingManager.veinProtos != null
                ? global::DspFindSeed.PlanetModelingManager.veinProtos.Length
                : 0;
            veinLen = Math.Min(veinLen, MaxOutVeinLen);
            if (veinLen <= 1)
                return false;

            int planetCount = planets.Count;
            long t0 = Stopwatch.GetTimestamp();
            Interlocked.Increment(ref _perfVeinBatchCalls);
            Interlocked.Add(ref _perfVeinBatchPlanets, planetCount);
            EnsureBuffers(planetCount, veinLen);

            for (int i = 0; i < planetCount; ++i)
            {
                var planet = planets[i];
                if (planet == null || planet.type == EPlanetType.Gas || planet.star == null)
                    return false;

                var theme = global::DspFindSeed.LDB.themes.Select(planet.theme);
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

                _planetSeeds[i] = planet.seed;
                _pValues[i] = CalcPAndBonusCase(planet.star.type, planet.star.spectr, out _bonusCases[i]);
                _isBirthStars[i] = planet.star.index == 0 ? 1 : 0;
                _veinSpotLens[i] = veinSpot.Length;
                _rareVeinLens[i] = rareVeins.Length;

                int vbase = i * MaxVeinSpotLen;
                for (int j = 0; j < MaxVeinSpotLen; ++j)
                    _veinSpotValues[vbase + j] = j < veinSpot.Length ? veinSpot[j] : 0;

                int rbase = i * MaxRareVeins;
                for (int j = 0; j < MaxRareVeins; ++j)
                    _rareVeinValues[rbase + j] = j < rareVeins.Length ? rareVeins[j] : 0;

                int sbase = i * MaxRareVeins * RareSettingsGroup;
                for (int j = 0; j < MaxRareVeins * RareSettingsGroup; ++j)
                    _rareSettingsValues[sbase + j] = 0f;
                int fill = rareVeins.Length * RareSettingsGroup;
                for (int j = 0; j < fill; ++j)
                    _rareSettingsValues[sbase + j] = rareSettings[j];
            }

            int deviceId = GetDeviceIdFromEnv();
            int rc;
            try
            {
                if (_mixChunkVeinEntryMissing)
                {
                    rc = NativeRefreshPlanetVeinSpotsBatch(
                        _planetSeeds,
                        _pValues,
                        _bonusCases,
                        _isBirthStars,
                        _veinSpotLens,
                        _rareVeinLens,
                        _veinSpotValues,
                        MaxVeinSpotLen,
                        _rareVeinValues,
                        MaxRareVeins,
                        _rareSettingsValues,
                        MaxRareVeins * RareSettingsGroup,
                        planetCount,
                        veinLen,
                        useFp32ProbCompare ? 1 : 0,
                        deviceId,
                        _outCounts);
                }
                else
                {
                    rc = NativeMixChunkEvalVeinsF32(
                        _planetSeeds,
                        _pValues,
                        _bonusCases,
                        _isBirthStars,
                        _veinSpotLens,
                        _rareVeinLens,
                        _veinSpotValues,
                        MaxVeinSpotLen,
                        _rareVeinValues,
                        MaxRareVeins,
                        _rareSettingsValues,
                        MaxRareVeins * RareSettingsGroup,
                        planetCount,
                        veinLen,
                        useFp32ProbCompare ? 1 : 0,
                        deviceId,
                        _outCounts);
                }
            }
            catch (EntryPointNotFoundException)
            {
                _mixChunkVeinEntryMissing = true;
                rc = NativeRefreshPlanetVeinSpotsBatch(
                    _planetSeeds,
                    _pValues,
                    _bonusCases,
                    _isBirthStars,
                    _veinSpotLens,
                    _rareVeinLens,
                    _veinSpotValues,
                    MaxVeinSpotLen,
                    _rareVeinValues,
                    MaxRareVeins,
                    _rareSettingsValues,
                    MaxRareVeins * RareSettingsGroup,
                    planetCount,
                    veinLen,
                    useFp32ProbCompare ? 1 : 0,
                    deviceId,
                    _outCounts);
            }
            catch (Exception ex) when (ex is DllNotFoundException || ex is BadImageFormatException)
            {
                Interlocked.Add(ref _perfVeinBatchTicks, Stopwatch.GetTimestamp() - t0);
                Interlocked.Increment(ref _perfVeinBatchFailCalls);
                MarkNativeBroken(ex.GetType().Name + ": " + ex.Message);
                return false;
            }

            if (rc != Ok)
            {
                Interlocked.Add(ref _perfVeinBatchTicks, Stopwatch.GetTimestamp() - t0);
                Interlocked.Increment(ref _perfVeinBatchFailCalls);
                MarkNativeBroken("planet native return code=" + rc);
                return false;
            }

            countsFlat = _outCounts;
            outVeinLen = veinLen;
            Interlocked.Add(ref _perfVeinBatchTicks, Stopwatch.GetTimestamp() - t0);
            return true;
        }

        internal static PlanetCoreBatchArrays AcquireCoreBatchArrays(int itemCount)
        {
            if (itemCount < 1)
                itemCount = 1;
            EnsureCoreBatchBuffers(itemCount);
            return new PlanetCoreBatchArrays
            {
                infoSeeds = _coreInfoSeeds,
                orbitArounds = _coreOrbitArounds,
                orbitIndexes = _coreOrbitIndexes,
                gasGiants = _coreGasGiants,
                starIndexes = _coreStarIndexes,
                galaxyStarCounts = _coreGalaxyStarCounts,
                galaxyHabitableCounts = _coreGalaxyHabitableCounts,
                boostInclinationNs = _coreBoostInclinationNs,
                compactTypeCases = _coreCompactTypeCases,
                starOrbitScalers = _coreStarOrbitScalers,
                starMasses = _coreStarMasses,
                starHabitableRadii = _coreStarHabitableRadii,
                starLightBalanceRadii = _coreStarLightBalanceRadii,
                aroundRealRadii = _coreAroundRealRadii,
                aroundOrbitRadii = _coreAroundOrbitRadii,
                aroundOrbitalPeriods = _coreAroundOrbitalPeriods
            };
        }

        internal static bool TryEvalPlanetCoreF32BatchRaw(
            PlanetCoreBatchArrays arrays,
            int batchCount,
            PlanetCoreF32Out[] outResults)
        {
            if (!IsCoreEnabled())
                return false;
            if (_nativeBroken)
                return false;
            if (batchCount < 1 || outResults == null || outResults.Length < batchCount)
                return false;

            long t0 = Stopwatch.GetTimestamp();
            Interlocked.Increment(ref _perfCoreBatchCalls);
            Interlocked.Add(ref _perfCoreBatchItems, batchCount);

            int deviceId = GetDeviceIdFromEnv();
            int rc;
            try
            {
                if (_mixChunkPlanetEntryMissing)
                {
                    rc = NativePlanetEvalCoreF32Batch(
                        arrays.infoSeeds,
                        arrays.orbitArounds,
                        arrays.orbitIndexes,
                        arrays.gasGiants,
                        arrays.starIndexes,
                        arrays.galaxyStarCounts,
                        arrays.galaxyHabitableCounts,
                        arrays.boostInclinationNs,
                        arrays.compactTypeCases,
                        arrays.starOrbitScalers,
                        arrays.starMasses,
                        arrays.starHabitableRadii,
                        arrays.starLightBalanceRadii,
                        arrays.aroundRealRadii,
                        arrays.aroundOrbitRadii,
                        arrays.aroundOrbitalPeriods,
                        batchCount,
                        deviceId,
                        outResults);
                }
                else
                {
                    rc = NativeMixChunkEvalPlanetsF32(
                        arrays.infoSeeds,
                        arrays.orbitArounds,
                        arrays.orbitIndexes,
                        arrays.gasGiants,
                        arrays.starIndexes,
                        arrays.galaxyStarCounts,
                        arrays.galaxyHabitableCounts,
                        arrays.boostInclinationNs,
                        arrays.compactTypeCases,
                        arrays.starOrbitScalers,
                        arrays.starMasses,
                        arrays.starHabitableRadii,
                        arrays.starLightBalanceRadii,
                        arrays.aroundRealRadii,
                        arrays.aroundOrbitRadii,
                        arrays.aroundOrbitalPeriods,
                        batchCount,
                        deviceId,
                        outResults);
                }
            }
            catch (EntryPointNotFoundException)
            {
                _mixChunkPlanetEntryMissing = true;
                try
                {
                    rc = NativePlanetEvalCoreF32Batch(
                        arrays.infoSeeds,
                        arrays.orbitArounds,
                        arrays.orbitIndexes,
                        arrays.gasGiants,
                        arrays.starIndexes,
                        arrays.galaxyStarCounts,
                        arrays.galaxyHabitableCounts,
                        arrays.boostInclinationNs,
                        arrays.compactTypeCases,
                        arrays.starOrbitScalers,
                        arrays.starMasses,
                        arrays.starHabitableRadii,
                        arrays.starLightBalanceRadii,
                        arrays.aroundRealRadii,
                        arrays.aroundOrbitRadii,
                        arrays.aroundOrbitalPeriods,
                        batchCount,
                        deviceId,
                        outResults);
                }
                catch (EntryPointNotFoundException)
                {
                    Interlocked.Add(ref _perfCoreBatchTicks, Stopwatch.GetTimestamp() - t0);
                    _coreBatchEntryMissing = true;
                    return false;
                }
            }
            catch (Exception ex) when (ex is DllNotFoundException || ex is BadImageFormatException)
            {
                Interlocked.Add(ref _perfCoreBatchTicks, Stopwatch.GetTimestamp() - t0);
                MarkNativeBroken(ex.GetType().Name + ": " + ex.Message);
                return false;
            }

            if (rc != Ok)
            {
                Interlocked.Add(ref _perfCoreBatchTicks, Stopwatch.GetTimestamp() - t0);
                MarkNativeBroken("planet-core-batch native return code=" + rc);
                return false;
            }

            Interlocked.Add(ref _perfCoreBatchTicks, Stopwatch.GetTimestamp() - t0);
            return true;
        }

        public static bool TryEvalPlanetCoreF32Batch(
            IList<PlanetCoreBatchInput> inputs,
            PlanetCoreF32Out[] outResults)
        {
            if (inputs == null || inputs.Count == 0 || outResults == null || outResults.Length < inputs.Count)
                return false;

            int n = inputs.Count;
            var arrays = AcquireCoreBatchArrays(n);
            for (int i = 0; i < n; ++i)
            {
                var r = inputs[i];
                arrays.infoSeeds[i] = r.infoSeed;
                arrays.orbitArounds[i] = r.orbitAround;
                arrays.orbitIndexes[i] = r.orbitIndex;
                arrays.gasGiants[i] = r.gasGiant;
                arrays.starIndexes[i] = r.starIndex;
                arrays.galaxyStarCounts[i] = r.galaxyStarCount;
                arrays.galaxyHabitableCounts[i] = r.galaxyHabitableCount;
                arrays.boostInclinationNs[i] = r.boostInclinationNs;
                arrays.compactTypeCases[i] = r.compactTypeCase;
                arrays.starOrbitScalers[i] = r.starOrbitScaler;
                arrays.starMasses[i] = r.starMass;
                arrays.starHabitableRadii[i] = r.starHabitableRadius;
                arrays.starLightBalanceRadii[i] = r.starLightBalanceRadius;
                arrays.aroundRealRadii[i] = r.orbitAroundPlanetRealRadius;
                arrays.aroundOrbitRadii[i] = r.orbitAroundPlanetOrbitRadius;
                arrays.aroundOrbitalPeriods[i] = r.orbitAroundPlanetOrbitalPeriod;
            }

            return TryEvalPlanetCoreF32BatchRaw(arrays, n, outResults);
        }

        public static bool TryEvalPlanetCoreF32(
            int infoSeed,
            int orbitAround,
            int orbitIndex,
            bool gasGiant,
            StarData star,
            PlanetData orbitAroundPlanet,
            int galaxyHabitableCount,
            out PlanetCoreF32Out core)
        {
            core = default;
            if (!IsCoreEnabled())
                return false;
            if (_nativeBroken)
                return false;
            if (star == null || star.galaxy == null)
                return false;

            int boostInclinationNs = star.type >= EStarType.NeutronStar ? 1 : 0;
            int compactTypeCase = 0;
            if (star.type == EStarType.WhiteDwarf) compactTypeCase = 1;
            else if (star.type == EStarType.NeutronStar) compactTypeCase = 2;
            else if (star.type == EStarType.BlackHole) compactTypeCase = 3;

            float aroundRealRadius = orbitAroundPlanet != null ? orbitAroundPlanet.realRadius : 0f;
            float aroundOrbitRadius = orbitAroundPlanet != null ? orbitAroundPlanet.orbitRadius : 0f;
            double aroundOrbitalPeriod = orbitAroundPlanet != null ? orbitAroundPlanet.orbitalPeriod : 0.0;

            var req = new PlanetCoreBatchRequest
            {
                infoSeed = infoSeed,
                orbitAround = orbitAround,
                orbitIndex = orbitIndex,
                gasGiant = gasGiant ? 1 : 0,
                starIndex = star.index,
                galaxyStarCount = star.galaxy.starCount,
                galaxyHabitableCount = galaxyHabitableCount,
                boostInclinationNs = boostInclinationNs,
                compactTypeCase = compactTypeCase,
                starOrbitScaler = star.orbitScaler,
                starMass = star.mass,
                starHabitableRadius = star.habitableRadius,
                starLightBalanceRadius = star.lightBalanceRadius,
                orbitAroundPlanetRealRadius = aroundRealRadius,
                orbitAroundPlanetOrbitRadius = aroundOrbitRadius,
                orbitAroundPlanetOrbitalPeriod = aroundOrbitalPeriod
            };

            if (TryEvalPlanetCoreF32Batched(req, out core))
                return true;

            return TryEvalPlanetCoreF32Single(req, out core);
        }

        private static bool TryEvalPlanetCoreF32Batched(PlanetCoreBatchRequest req, out PlanetCoreF32Out core)
        {
            core = default;
            if (_coreBatchEntryMissing || _nativeBroken)
                return false;

            long t0 = Stopwatch.GetTimestamp();
            Interlocked.Increment(ref _perfCoreReqCount);
            EnsureCoreBatchWorker();
            _coreBatchQueue.Enqueue(req);
            _coreBatchWake.Set();

            // 保守等待，避免 worker 异常导致调用方永久阻塞。
            if (!req.done.Wait(30000))
            {
                Interlocked.Add(ref _perfCoreReqWaitTicks, Stopwatch.GetTimestamp() - t0);
                return false;
            }
            if (!req.success)
            {
                Interlocked.Add(ref _perfCoreReqWaitTicks, Stopwatch.GetTimestamp() - t0);
                return false;
            }
            core = req.core;
            Interlocked.Add(ref _perfCoreReqWaitTicks, Stopwatch.GetTimestamp() - t0);
            return true;
        }

        private static void EnsureCoreBatchWorker()
        {
            if (Interlocked.CompareExchange(ref _coreBatchWorkerStarted, 1, 0) != 0)
                return;

            var t = new Thread(CoreBatchWorkerLoop)
            {
                IsBackground = true,
                Name = "CudaPlanetCoreBatchWorker"
            };
            t.Start();
        }

        private static void CoreBatchWorkerLoop()
        {
            var batch = new List<PlanetCoreBatchRequest>(GetCoreBatchMaxSize());
            int collectUs = GetCoreBatchCollectUs();
            while (true)
            {
                batch.Clear();
                if (!_coreBatchQueue.TryDequeue(out var first))
                {
                    _coreBatchWake.WaitOne();
                    if (!_coreBatchQueue.TryDequeue(out first))
                        continue;
                }

                batch.Add(first);
                if (collectUs > 0)
                {
                    long waitTicks = (long)(collectUs * (Stopwatch.Frequency / 1_000_000.0));
                    long deadline = Stopwatch.GetTimestamp() + waitTicks;
                    while (batch.Count < GetCoreBatchMaxSize())
                    {
                        if (_coreBatchQueue.TryDequeue(out var req))
                        {
                            batch.Add(req);
                            continue;
                        }
                        if (Stopwatch.GetTimestamp() >= deadline)
                            break;
                        Thread.SpinWait(64);
                    }
                }

                while (batch.Count < GetCoreBatchMaxSize() && _coreBatchQueue.TryDequeue(out var req2))
                {
                    batch.Add(req2);
                }

                bool ok = TryEvalPlanetCoreF32BatchInternal(batch);
                if (!ok)
                {
                    Interlocked.Increment(ref _perfCoreBatchFallbackCalls);
                    for (int i = 0; i < batch.Count; ++i)
                    {
                        var r = batch[i];
                        r.success = TryEvalPlanetCoreF32Single(r, out r.core);
                    }
                }

                for (int i = 0; i < batch.Count; ++i)
                    batch[i].done.Set();
            }
        }

        private static bool TryEvalPlanetCoreF32BatchInternal(List<PlanetCoreBatchRequest> batch)
        {
            if (batch == null || batch.Count == 0)
                return false;
            if (_coreBatchEntryMissing || _nativeBroken)
                return false;

            int n = batch.Count;
            long t0 = Stopwatch.GetTimestamp();
            Interlocked.Increment(ref _perfCoreBatchCalls);
            Interlocked.Add(ref _perfCoreBatchItems, n);
            var infoSeeds = new int[n];
            var orbitArounds = new int[n];
            var orbitIndexes = new int[n];
            var gasGiants = new int[n];
            var starIndexes = new int[n];
            var galaxyStarCounts = new int[n];
            var galaxyHabitableCounts = new int[n];
            var boostInclinationNs = new int[n];
            var compactTypeCases = new int[n];
            var starOrbitScalers = new float[n];
            var starMasses = new double[n];
            var starHabitableRadii = new float[n];
            var starLightBalanceRadii = new float[n];
            var aroundRealRadii = new float[n];
            var aroundOrbitRadii = new float[n];
            var aroundOrbitalPeriods = new double[n];
            var outResults = new PlanetCoreF32Out[n];

            for (int i = 0; i < n; ++i)
            {
                var r = batch[i];
                infoSeeds[i] = r.infoSeed;
                orbitArounds[i] = r.orbitAround;
                orbitIndexes[i] = r.orbitIndex;
                gasGiants[i] = r.gasGiant;
                starIndexes[i] = r.starIndex;
                galaxyStarCounts[i] = r.galaxyStarCount;
                galaxyHabitableCounts[i] = r.galaxyHabitableCount;
                boostInclinationNs[i] = r.boostInclinationNs;
                compactTypeCases[i] = r.compactTypeCase;
                starOrbitScalers[i] = r.starOrbitScaler;
                starMasses[i] = r.starMass;
                starHabitableRadii[i] = r.starHabitableRadius;
                starLightBalanceRadii[i] = r.starLightBalanceRadius;
                aroundRealRadii[i] = r.orbitAroundPlanetRealRadius;
                aroundOrbitRadii[i] = r.orbitAroundPlanetOrbitRadius;
                aroundOrbitalPeriods[i] = r.orbitAroundPlanetOrbitalPeriod;
            }

            int deviceId = GetDeviceIdFromEnv();
            int rc;
            try
            {
                if (_mixChunkPlanetEntryMissing)
                {
                    rc = NativePlanetEvalCoreF32Batch(
                        infoSeeds,
                        orbitArounds,
                        orbitIndexes,
                        gasGiants,
                        starIndexes,
                        galaxyStarCounts,
                        galaxyHabitableCounts,
                        boostInclinationNs,
                        compactTypeCases,
                        starOrbitScalers,
                        starMasses,
                        starHabitableRadii,
                        starLightBalanceRadii,
                        aroundRealRadii,
                        aroundOrbitRadii,
                        aroundOrbitalPeriods,
                        n,
                        deviceId,
                        outResults);
                }
                else
                {
                    rc = NativeMixChunkEvalPlanetsF32(
                        infoSeeds,
                        orbitArounds,
                        orbitIndexes,
                        gasGiants,
                        starIndexes,
                        galaxyStarCounts,
                        galaxyHabitableCounts,
                        boostInclinationNs,
                        compactTypeCases,
                        starOrbitScalers,
                        starMasses,
                        starHabitableRadii,
                        starLightBalanceRadii,
                        aroundRealRadii,
                        aroundOrbitRadii,
                        aroundOrbitalPeriods,
                        n,
                        deviceId,
                        outResults);
                }
            }
            catch (EntryPointNotFoundException)
            {
                _mixChunkPlanetEntryMissing = true;
                try
                {
                    rc = NativePlanetEvalCoreF32Batch(
                        infoSeeds,
                        orbitArounds,
                        orbitIndexes,
                        gasGiants,
                        starIndexes,
                        galaxyStarCounts,
                        galaxyHabitableCounts,
                        boostInclinationNs,
                        compactTypeCases,
                        starOrbitScalers,
                        starMasses,
                        starHabitableRadii,
                        starLightBalanceRadii,
                        aroundRealRadii,
                        aroundOrbitRadii,
                        aroundOrbitalPeriods,
                        n,
                        deviceId,
                        outResults);
                }
                catch (EntryPointNotFoundException)
                {
                    Interlocked.Add(ref _perfCoreBatchTicks, Stopwatch.GetTimestamp() - t0);
                    _coreBatchEntryMissing = true;
                    if (!_coreBatchFallbackPrinted)
                    {
                        _coreBatchFallbackPrinted = true;
                        Console.WriteLine("[cuda-planet] planet-core batch entry not found, fallback to single.");
                    }
                    return false;
                }
            }
            catch (Exception ex) when (ex is DllNotFoundException || ex is BadImageFormatException)
            {
                Interlocked.Add(ref _perfCoreBatchTicks, Stopwatch.GetTimestamp() - t0);
                MarkNativeBroken(ex.GetType().Name + ": " + ex.Message);
                return false;
            }

            if (rc != Ok)
            {
                Interlocked.Add(ref _perfCoreBatchTicks, Stopwatch.GetTimestamp() - t0);
                MarkNativeBroken("planet-core-batch native return code=" + rc);
                return false;
            }

            for (int i = 0; i < n; ++i)
            {
                batch[i].core = outResults[i];
                batch[i].success = true;
            }
            Interlocked.Add(ref _perfCoreBatchTicks, Stopwatch.GetTimestamp() - t0);
            return true;
        }

        private static bool TryEvalPlanetCoreF32Single(PlanetCoreBatchRequest req, out PlanetCoreF32Out core)
        {
            core = default;
            if (_nativeBroken)
                return false;

            long t0 = Stopwatch.GetTimestamp();
            Interlocked.Increment(ref _perfCoreSingleCalls);
            int deviceId = GetDeviceIdFromEnv();
            int rc;
            try
            {
                rc = NativePlanetEvalCoreF32(
                    req.infoSeed,
                    req.orbitAround,
                    req.orbitIndex,
                    req.gasGiant,
                    req.starIndex,
                    req.galaxyStarCount,
                    req.galaxyHabitableCount,
                    req.boostInclinationNs,
                    req.compactTypeCase,
                    req.starOrbitScaler,
                    req.starMass,
                    req.starHabitableRadius,
                    req.starLightBalanceRadius,
                    req.orbitAroundPlanetRealRadius,
                    req.orbitAroundPlanetOrbitRadius,
                    req.orbitAroundPlanetOrbitalPeriod,
                    deviceId,
                    out core);
            }
            catch (Exception ex) when (ex is DllNotFoundException || ex is EntryPointNotFoundException || ex is BadImageFormatException)
            {
                Interlocked.Add(ref _perfCoreSingleTicks, Stopwatch.GetTimestamp() - t0);
                MarkNativeBroken(ex.GetType().Name + ": " + ex.Message);
                return false;
            }

            if (rc != Ok)
            {
                Interlocked.Add(ref _perfCoreSingleTicks, Stopwatch.GetTimestamp() - t0);
                MarkNativeBroken("planet-core native return code=" + rc);
                return false;
            }
            Interlocked.Add(ref _perfCoreSingleTicks, Stopwatch.GetTimestamp() - t0);
            return true;
        }

        private static int GetCoreBatchMaxSize()
        {
            const int defaultSize = 2048;
            var env = Environment.GetEnvironmentVariable("DSP_CUDA_PLANET_CORE_BATCH_MAX");
            if (string.IsNullOrEmpty(env))
                return defaultSize;
            if (!int.TryParse(env, out var parsed))
                return defaultSize;
            if (parsed < 8)
                return 8;
            if (parsed > 16384)
                return 16384;
            return parsed;
        }

        private static int GetCoreBatchCollectUs()
        {
            const int defaultUs = 300;
            var env = Environment.GetEnvironmentVariable("DSP_CUDA_PLANET_CORE_BATCH_WAIT_US");
            if (string.IsNullOrEmpty(env))
                return defaultUs;
            if (!int.TryParse(env, out var parsed))
                return defaultUs;
            if (parsed < 0)
                return 0;
            if (parsed > 5000)
                return 5000;
            return parsed;
        }

        public static bool TryEvalPlanetGasDetailsF32(
            int themeSeed,
            float gasCoef,
            float resourceCoef,
            float[] inGasSpeeds,
            float[] inGasHeatValues,
            float[] outGasSpeeds,
            out double totalHeat)
        {
            totalHeat = 0.0;
            if (!IsCoreEnabled())
                return false;
            if (_nativeBroken)
                return false;
            if (inGasSpeeds == null || inGasHeatValues == null || outGasSpeeds == null)
                return false;
            if (inGasSpeeds.Length == 0)
                return false;
            if (inGasHeatValues.Length < inGasSpeeds.Length || outGasSpeeds.Length < inGasSpeeds.Length)
                return false;

            int deviceId = GetDeviceIdFromEnv();
            int rc;
            try
            {
                rc = NativePlanetEvalGasDetailsF32(
                    themeSeed,
                    gasCoef,
                    resourceCoef,
                    inGasSpeeds,
                    inGasHeatValues,
                    inGasSpeeds.Length,
                    deviceId,
                    outGasSpeeds,
                    out totalHeat);
            }
            catch (Exception ex) when (ex is DllNotFoundException || ex is EntryPointNotFoundException || ex is BadImageFormatException)
            {
                MarkNativeBroken(ex.GetType().Name + ": " + ex.Message);
                return false;
            }

            if (rc != Ok)
            {
                MarkNativeBroken("planet-gas native return code=" + rc);
                return false;
            }
            return true;
        }

        public static int MapTypeCaseToPlanetType(int typeCase)
        {
            switch (typeCase)
            {
                case 0: return (int)EPlanetType.Gas;
                case 1: return (int)EPlanetType.Ocean;
                case 2: return (int)EPlanetType.Vocano;
                case 4: return (int)EPlanetType.Ice;
                default: return (int)EPlanetType.Desert;
            }
        }

        public static EPlanetSingularity BuildSingularityFromFlags(int flags)
        {
            EPlanetSingularity s = 0;
            if ((flags & SingularityLaySide) != 0) s |= EPlanetSingularity.LaySide;
            if ((flags & SingularityTidal1) != 0) s |= EPlanetSingularity.TidalLocked;
            if ((flags & SingularityTidal2) != 0) s |= EPlanetSingularity.TidalLocked2;
            if ((flags & SingularityTidal4) != 0) s |= EPlanetSingularity.TidalLocked4;
            if ((flags & SingularityClockwise) != 0) s |= EPlanetSingularity.ClockwiseRotate;
            return s;
        }

        private static float CalcPAndBonusCase(EStarType starType, ESpectrType spectr, out int bonusCase)
        {
            bonusCase = 0;
            float p = 1f;
            switch (starType)
            {
                case EStarType.MainSeqStar:
                    switch (spectr)
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
                    bonusCase = 1;
                    break;
                case EStarType.NeutronStar:
                    p = 4.5f;
                    bonusCase = 2;
                    break;
                case EStarType.BlackHole:
                    p = 5f;
                    bonusCase = 2;
                    break;
            }
            return p;
        }

        private static void EnsureBuffers(int planetCount, int outVeinLen)
        {
            EnsureIntBuffer(ref _planetSeeds, planetCount);
            EnsureFloatBuffer(ref _pValues, planetCount);
            EnsureIntBuffer(ref _bonusCases, planetCount);
            EnsureIntBuffer(ref _isBirthStars, planetCount);
            EnsureIntBuffer(ref _veinSpotLens, planetCount);
            EnsureIntBuffer(ref _rareVeinLens, planetCount);
            EnsureIntBuffer(ref _veinSpotValues, planetCount * MaxVeinSpotLen);
            EnsureIntBuffer(ref _rareVeinValues, planetCount * MaxRareVeins);
            EnsureFloatBuffer(ref _rareSettingsValues, planetCount * MaxRareVeins * RareSettingsGroup);
            EnsureIntBuffer(ref _outCounts, planetCount * outVeinLen);
        }

        private static void EnsureCoreBatchBuffers(int itemCount)
        {
            EnsureIntBuffer(ref _coreInfoSeeds, itemCount);
            EnsureIntBuffer(ref _coreOrbitArounds, itemCount);
            EnsureIntBuffer(ref _coreOrbitIndexes, itemCount);
            EnsureIntBuffer(ref _coreGasGiants, itemCount);
            EnsureIntBuffer(ref _coreStarIndexes, itemCount);
            EnsureIntBuffer(ref _coreGalaxyStarCounts, itemCount);
            EnsureIntBuffer(ref _coreGalaxyHabitableCounts, itemCount);
            EnsureIntBuffer(ref _coreBoostInclinationNs, itemCount);
            EnsureIntBuffer(ref _coreCompactTypeCases, itemCount);
            EnsureFloatBuffer(ref _coreStarOrbitScalers, itemCount);
            EnsureDoubleBuffer(ref _coreStarMasses, itemCount);
            EnsureFloatBuffer(ref _coreStarHabitableRadii, itemCount);
            EnsureFloatBuffer(ref _coreStarLightBalanceRadii, itemCount);
            EnsureFloatBuffer(ref _coreAroundRealRadii, itemCount);
            EnsureFloatBuffer(ref _coreAroundOrbitRadii, itemCount);
            EnsureDoubleBuffer(ref _coreAroundOrbitalPeriods, itemCount);
        }

        private static void EnsureIntBuffer(ref int[] arr, int needed)
        {
            if (arr == null || arr.Length < needed)
                arr = new int[needed];
        }

        private static void EnsureFloatBuffer(ref float[] arr, int needed)
        {
            if (arr == null || arr.Length < needed)
                arr = new float[needed];
        }

        private static void EnsureDoubleBuffer(ref double[] arr, int needed)
        {
            if (arr == null || arr.Length < needed)
                arr = new double[needed];
        }

        private static int GetDeviceIdFromEnv()
        {
            var env = Environment.GetEnvironmentVariable("DSP_CUDA_DEVICE");
            if (string.IsNullOrEmpty(env))
                return -1;
            if (!int.TryParse(env, out var parsed))
                return -1;
            return parsed;
        }

        private static void MarkNativeBroken(string reason)
        {
            _nativeBroken = true;
            _nativeBrokenReason = reason;
            if (_printedFallback)
                return;
            _printedFallback = true;
            Console.WriteLine("[cuda-planet] fallback to CPU: " + _nativeBrokenReason);
        }
    }
}
