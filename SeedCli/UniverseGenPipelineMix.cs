using UnityEngine;

namespace SeedCli
{
    /// <summary>
    /// 端到端“混合精度（已验证星系 0 mismatch）管线”：
    /// - 星系点位：UniverseGenF32.CreateGalaxy_PtFp64_RandFp64_ParamsFp64（其余尽量 FP32）
    /// - 行星生成：StarGenPlanetsF32 + PlanetGenF32（更多 FP32 化，但此前对比为 0 mismatch）
    /// </summary>
    internal static class UniverseGenPipelineMix
    {
        public static GalaxyData CreateGalaxy(global::DspFindSeed.GameDesc gameDesc)
        {
            return CreateGalaxy(gameDesc, collisionFp64: false);
        }

        public static GalaxyData CreateGalaxy(global::DspFindSeed.GameDesc gameDesc, bool collisionFp64)
        {
            bool signatureFastPath = MixRuntimeFlags.SignatureOnlyFastPath;
            var g = collisionFp64
                ? (signatureFastPath
                    ? UniverseGenF32.CreateGalaxy_PtFp64_RandFp64_ParamsFp64_CollFp64_StarsOnly(gameDesc)
                    : UniverseGenF32.CreateGalaxy_PtFp64_RandFp64_ParamsFp64_CollFp64(gameDesc))
                : (signatureFastPath
                    ? UniverseGenF32.CreateGalaxy_PtFp64_RandFp64_ParamsFp64_StarsOnly(gameDesc)
                    : UniverseGenF32.CreateGalaxy_PtFp64_RandFp64_ParamsFp64(gameDesc));
            if (g == null || g.stars == null)
                return g;

            // 非 fast-path 下 UniverseGenF32 可能已生成过行星；统一重置后按本管线覆盖生成。
            g.habitableCount = 0;

            var astrosData = g.astrosData;
            var stars = g.stars;
            bool useGalaxyCoreBatch = CudaPlanetNative.IsCoreEnabled() && g.starCount > 1;
            bool chunkWideCoreBatch = useGalaxyCoreBatch && MixRuntimeFlags.ChunkWideCoreBatch;
            bool manageGalaxyCoreBatchScope = useGalaxyCoreBatch && !chunkWideCoreBatch;
            if (manageGalaxyCoreBatchScope)
                StarGenPlanetsF32.BeginGalaxyBatch();
            try
            {
                bool writeAstroPose = !MixRuntimeFlags.SignatureOnlyFastPath;
                for (int i = 0; i < g.starCount; ++i)
                {
                    StarGenPlanetsF32.CreateStarPlanetsF32(g, stars[i], gameDesc);
                    if (writeAstroPose)
                    {
                        astrosData[stars[i].id * 100].uPos = astrosData[stars[i].id * 100].uPosNext = stars[i].uPosition;
                        astrosData[stars[i].id * 100].uRot = astrosData[stars[i].id * 100].uRotNext = Quaternion.identity;
                        astrosData[stars[i].id * 100].uRadius = stars[i].physicsRadius;
                    }
                }
            }
            finally
            {
                if (manageGalaxyCoreBatchScope)
                    StarGenPlanetsF32.FlushGalaxyBatch();
            }

            if (!chunkWideCoreBatch)
                RefreshBirthPlanet(g);

            return g;
        }

        public static void RefreshBirthPlanet(GalaxyData g)
        {
            if (g == null || g.stars == null)
                return;

            g.birthPlanetId = 0;
            if (g.starCount <= 0)
                return;

            StarData birthStar = g.stars[0];
            for (int i = 0; i < birthStar.planetCount; ++i)
            {
                PlanetData planet = birthStar.planets[i];
                var themeProto = global::DspFindSeed.LDB.themes.Select(planet.theme);
                if (themeProto != null && themeProto.Distribute == EThemeDistribute.Birth)
                {
                    g.birthPlanetId = planet.id;
                    g.birthStarId = birthStar.id;
                    break;
                }
            }
        }
    }
}
