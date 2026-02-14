using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

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
            public float rand1;
            public float rand2;
            public float rand3;
            public float rand4;
            public int theme_seed;
            public int type_case;
            public int singularity_flags;
            public int habitable_count_delta;
            public int precision;
            public int segment;
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
            catch (Exception ex) when (ex is DllNotFoundException || ex is EntryPointNotFoundException || ex is BadImageFormatException)
            {
                MarkNativeBroken(ex.GetType().Name + ": " + ex.Message);
                return false;
            }

            if (rc != Ok)
            {
                MarkNativeBroken("planet native return code=" + rc);
                return false;
            }

            countsFlat = _outCounts;
            outVeinLen = veinLen;
            return true;
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

            int deviceId = GetDeviceIdFromEnv();
            int rc;
            try
            {
                rc = NativePlanetEvalCoreF32(
                    infoSeed,
                    orbitAround,
                    orbitIndex,
                    gasGiant ? 1 : 0,
                    star.index,
                    star.galaxy.starCount,
                    galaxyHabitableCount,
                    boostInclinationNs,
                    compactTypeCase,
                    star.orbitScaler,
                    star.mass,
                    star.habitableRadius,
                    star.lightBalanceRadius,
                    aroundRealRadius,
                    aroundOrbitRadius,
                    aroundOrbitalPeriod,
                    deviceId,
                    out core);
            }
            catch (Exception ex) when (ex is DllNotFoundException || ex is EntryPointNotFoundException || ex is BadImageFormatException)
            {
                MarkNativeBroken(ex.GetType().Name + ": " + ex.Message);
                return false;
            }

            if (rc != Ok)
            {
                MarkNativeBroken("planet-core native return code=" + rc);
                return false;
            }

            return true;
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
