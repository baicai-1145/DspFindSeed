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
            var g = UniverseGenF32.CreateGalaxy_PtFp64_RandFp64_ParamsFp64(gameDesc);
            if (g == null || g.stars == null)
                return g;

            // UniverseGenF32 变体内部已生成过一遍行星，必须重置 habitableCount 再覆盖生成
            g.habitableCount = 0;

            var astrosData = g.astrosData;
            var stars = g.stars;
            for (int i = 0; i < g.starCount; ++i)
            {
                StarGenPlanetsF32.CreateStarPlanetsF32(g, stars[i], gameDesc);
                astrosData[stars[i].id * 100].uPos = astrosData[stars[i].id * 100].uPosNext = stars[i].uPosition;
                astrosData[stars[i].id * 100].uRot = astrosData[stars[i].id * 100].uRotNext = Quaternion.identity;
                astrosData[stars[i].id * 100].uRadius = stars[i].physicsRadius;
            }

            g.birthPlanetId = 0;
            if (g.starCount > 0)
            {
                StarData birthStar = stars[0];
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

            return g;
        }
    }
}

