using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

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
        [ThreadStatic] private static NativeVec3d[] _singlePoseScratch;

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
                MarkNativeBroken(ex.GetType().Name + ": " + ex.Message);
                return false;
            }

            if (rc != Ok)
            {
                MarkNativeBroken("native return code=" + rc);
                return false;
            }

            if (outCount < 0 || outCount > maxCount)
            {
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
                MarkNativeBroken(ex.GetType().Name + ": " + ex.Message);
                return false;
            }

            if (rc != Ok)
            {
                MarkNativeBroken("batch native return code=" + rc);
                return false;
            }

            return true;
        }

        private static NativeVec3d[] EnsureSinglePoseScratch(int needed)
        {
            if (_singlePoseScratch == null || _singlePoseScratch.Length < needed)
                _singlePoseScratch = new NativeVec3d[needed];
            return _singlePoseScratch;
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
