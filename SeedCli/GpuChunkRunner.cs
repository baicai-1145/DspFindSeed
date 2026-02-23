using System.Diagnostics;
using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;

namespace SeedCli
{
    internal sealed class GpuChunkRunner
    {
        internal struct ChunkRunResult
        {
            public bool Prefetched;
            public long PrefetchTicks;
            public long CreateGalaxyTicks;
        }

        private readonly int _starCount;
        private readonly bool _mixCollisionFp64;
        private readonly int _mixThreads;

        public GpuChunkRunner(int starCount, bool mixCollisionFp64, int mixThreads)
        {
            _starCount = starCount;
            _mixCollisionFp64 = mixCollisionFp64;
            _mixThreads = mixThreads < 1 ? 1 : mixThreads;
        }

        public bool PrefetchChunk(int seedBase, int chunkSize, bool timingDebug, GpuChunkBuffers buffers, out long prefetchTicks)
        {
            prefetchTicks = 0;
            if (chunkSize <= 0)
                return false;

            PrepareChunkBuffers(seedBase, chunkSize, buffers);
            long tp0 = timingDebug ? Stopwatch.GetTimestamp() : 0;
            var seedSlice = new ArraySegment<int>(buffers.SeedBuffer, 0, chunkSize);
            bool prefetched = UniverseGenF32.PrecomputeTempPosesParamsFp64Batch(seedSlice, _starCount, collisionFp64: _mixCollisionFp64);
            if (timingDebug)
                prefetchTicks = Stopwatch.GetTimestamp() - tp0;
            return prefetched;
        }

        public ChunkRunResult RunChunk(int seedBase, int chunkSize, bool enableBatchPrefetch, bool timingDebug, GpuChunkBuffers buffers)
        {
            MixRuntimeFlags.ChunkWideCoreBatch = false;
            PrepareChunkBuffers(seedBase, chunkSize, buffers);

            var result = new ChunkRunResult();
            if (enableBatchPrefetch)
            {
                result.Prefetched = PrefetchChunk(seedBase, chunkSize, timingDebug, buffers, out var prefetchTicks);
                result.PrefetchTicks = prefetchTicks;
            }

            long tg0 = timingDebug ? Stopwatch.GetTimestamp() : 0;
            bool useChunkWideCoreBatch = _mixThreads == 1
                && chunkSize > 1
                && CudaPlanetNative.IsCoreEnabled()
                && MixRuntimeFlags.SignatureOnlyFastPath;
            bool useChunkWideCoreBatchParallel = _mixThreads > 1
                && chunkSize > 1
                && CudaPlanetNative.IsCoreEnabled()
                && MixRuntimeFlags.SignatureOnlyFastPath;
            if (_mixThreads > 1)
            {
                var options = new ParallelOptions { MaxDegreeOfParallelism = _mixThreads };
                if (useChunkWideCoreBatchParallel)
                {
                    int groupSize = MixRuntimeFlags.ChunkWideCoreBatchSeedGroup;
                    if (groupSize < 1)
                        groupSize = 1;
                    MixRuntimeFlags.ChunkWideCoreBatch = true;
                    try
                    {
                        int rangeSize = (chunkSize + _mixThreads - 1) / _mixThreads;
                        if (rangeSize < 1)
                            rangeSize = 1;
                        Parallel.ForEach(
                            Partitioner.Create(0, chunkSize, rangeSize),
                            options,
                            range =>
                            {
                                StarGenPlanetsF32.BeginGalaxyBatch();
                                int groupStart = range.Item1;
                                int groupCount = 0;
                                try
                                {
                                    for (int i = range.Item1; i < range.Item2; ++i)
                                    {
                                        int seed = buffers.SeedBuffer[i];
                                        var gd = new global::DspFindSeed.GameDesc();
                                        gd.SetForNewGame(global::DspFindSeed.UniverseGen.algoVersion, seed, _starCount, 1, 1f);
                                        buffers.GalaxyBuffer[i] = UniverseGenPipelineMix.CreateGalaxy(gd, collisionFp64: _mixCollisionFp64);
                                        groupCount++;
                                        if (groupCount >= groupSize)
                                        {
                                            StarGenPlanetsF32.FlushGalaxyBatch();
                                            for (int k = groupStart; k <= i; ++k)
                                                UniverseGenPipelineMix.RefreshBirthPlanet(buffers.GalaxyBuffer[k]);
                                            if (i + 1 < range.Item2)
                                                StarGenPlanetsF32.BeginGalaxyBatch();
                                            groupStart = i + 1;
                                            groupCount = 0;
                                        }
                                    }
                                }
                                finally
                                {
                                    StarGenPlanetsF32.FlushGalaxyBatch();
                                    for (int i = groupStart; i < range.Item2; ++i)
                                        UniverseGenPipelineMix.RefreshBirthPlanet(buffers.GalaxyBuffer[i]);
                                }
                            });
                    }
                    finally
                    {
                        MixRuntimeFlags.ChunkWideCoreBatch = false;
                    }
                }
                else
                {
                    Parallel.For(0, chunkSize, options, i =>
                    {
                        int seed = buffers.SeedBuffer[i];
                        var gd = new global::DspFindSeed.GameDesc();
                        gd.SetForNewGame(global::DspFindSeed.UniverseGen.algoVersion, seed, _starCount, 1, 1f);
                        buffers.GalaxyBuffer[i] = UniverseGenPipelineMix.CreateGalaxy(gd, collisionFp64: _mixCollisionFp64);
                    });
                }
            }
            else
            {
                if (useChunkWideCoreBatch)
                {
                    MixRuntimeFlags.ChunkWideCoreBatch = true;
                    int groupSize = MixRuntimeFlags.ChunkWideCoreBatchSeedGroup;
                    if (groupSize < 1)
                        groupSize = 1;
                    StarGenPlanetsF32.BeginGalaxyBatch();
                    int groupStart = 0;
                    int groupCount = 0;
                    try
                    {
                        for (int i = 0; i < chunkSize; ++i)
                        {
                            int seed = buffers.SeedBuffer[i];
                            var gd = new global::DspFindSeed.GameDesc();
                            gd.SetForNewGame(global::DspFindSeed.UniverseGen.algoVersion, seed, _starCount, 1, 1f);
                            buffers.GalaxyBuffer[i] = UniverseGenPipelineMix.CreateGalaxy(gd, collisionFp64: _mixCollisionFp64);
                            groupCount++;
                            if (groupCount >= groupSize)
                            {
                                StarGenPlanetsF32.FlushGalaxyBatch();
                                for (int k = groupStart; k <= i; ++k)
                                    UniverseGenPipelineMix.RefreshBirthPlanet(buffers.GalaxyBuffer[k]);
                                if (i + 1 < chunkSize)
                                    StarGenPlanetsF32.BeginGalaxyBatch();
                                groupStart = i + 1;
                                groupCount = 0;
                            }
                        }
                    }
                    finally
                    {
                        StarGenPlanetsF32.FlushGalaxyBatch();
                        MixRuntimeFlags.ChunkWideCoreBatch = false;
                    }
                    for (int i = groupStart; i < chunkSize; ++i)
                    {
                        UniverseGenPipelineMix.RefreshBirthPlanet(buffers.GalaxyBuffer[i]);
                    }
                }
                else
                {
                    for (int i = 0; i < chunkSize; ++i)
                    {
                        int seed = buffers.SeedBuffer[i];
                        var gd = new global::DspFindSeed.GameDesc();
                        gd.SetForNewGame(global::DspFindSeed.UniverseGen.algoVersion, seed, _starCount, 1, 1f);
                        buffers.GalaxyBuffer[i] = UniverseGenPipelineMix.CreateGalaxy(gd, collisionFp64: _mixCollisionFp64);
                    }
                }
            }

            if (timingDebug)
                result.CreateGalaxyTicks = Stopwatch.GetTimestamp() - tg0;
            return result;
        }

        private static void PrepareChunkBuffers(int seedBase, int chunkSize, GpuChunkBuffers buffers)
        {
            buffers.EnsureCapacity(chunkSize);
            for (int i = 0; i < chunkSize; ++i)
            {
                buffers.SeedBuffer[i] = seedBase + i;
                buffers.GalaxyBuffer[i] = null;
            }
        }

        public static void ClearPrefetch()
        {
            UniverseGenF32.ClearPrefetchedTempPosesParamsFp64();
        }
    }
}
