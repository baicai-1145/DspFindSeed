using UnityEngine;

namespace SeedCli
{
    /// <summary>
    /// 端到端“全 FP32 管线”实验（用于正确率/差异率统计，不追求与原版一致）：
    /// - 星系点位：UniverseGenF32（pt/collision/rand 全 FP32）
    /// - 行星生成：StarGenPlanetsF32 + PlanetGenF32（更多 FP32 化）
    /// - 矿脉：在对比签名里用 RefreshPlanetData_F32（概率比较也强制 FP32）
    /// </summary>
    internal static class UniverseGenPipelineF32
    {
        public static GalaxyData CreateGalaxy(global::DspFindSeed.GameDesc gameDesc)
        {
            // 先用 UniverseGenF32 生成“星系+恒星点位”（它内部也会生成行星，但我们会覆盖）
            var g = UniverseGenF32.CreateGalaxy(gameDesc);
            if (g == null || g.stars == null)
                return g;

            // 重要：行星生成会读写 galaxy.habitableCount。UniverseGenF32 已经生成过一遍行星，
            // 若不重置会导致第二遍（FP32 行星）系统性偏差。
            g.habitableCount = 0;

            // 覆盖行星生成：改用 FP32 版本
            var astrosData = g.astrosData;
            var stars = g.stars;
            for (int i = 0; i < g.starCount; ++i)
            {
                StarGenPlanetsF32.CreateStarPlanetsF32(g, stars[i], gameDesc);
                astrosData[stars[i].id * 100].uPos = astrosData[stars[i].id * 100].uPosNext = stars[i].uPosition;
                astrosData[stars[i].id * 100].uRot = astrosData[stars[i].id * 100].uRotNext = Quaternion.identity;
                astrosData[stars[i].id * 100].uRadius = stars[i].physicsRadius;
            }

            // 重新计算出生星/出生行星（因为行星主题可能变化）
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

