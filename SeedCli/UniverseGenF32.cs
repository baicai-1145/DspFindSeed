using System;
using System.Collections.Generic;
using UnityEngine;

namespace SeedCli
{
    /// <summary>
    /// 实验用途：把星系点位生成链路（RandomPoses/CheckCollision）改成 FP32，
    /// 观察与原版 FP64 的差异（并不追求与游戏一致）。
    ///
    /// 注意：VectorLF3/GalaxyData/StarData 等类型来自 Assembly-CSharp。
    /// </summary>
    internal static class UniverseGenF32
    {
        private static readonly List<VectorLF3> _tmpPoses = new List<VectorLF3>();
        private static readonly List<VectorLF3> _tmpDrunk = new List<VectorLF3>();

        public static GalaxyData CreateGalaxy(DspFindSeed.GameDesc gameDesc)
        {
            int galaxyAlgo = gameDesc.galaxyAlgo;
            int galaxySeed = gameDesc.galaxySeed;
            int starCount  = gameDesc.starCount;

            if (galaxyAlgo < 20200101 || galaxyAlgo > 20591231)
                throw new Exception("Wrong version of unigen algorithm!");

            DspFindSeed.PlanetGen.gasCoef = gameDesc.isRareResource ? 0.8f : 1f;

            var rng = new DotNet35Random(galaxySeed);
            int tempPoses = GenerateTempPosesF32(rng.Next(), starCount, 4, 2.0f, 2.3f, 3.5f, 0.18f, ptFp64: false, collisionFp64: false, randFp64: false);

            GalaxyData galaxy = new GalaxyData();
            galaxy.seed = galaxySeed;
            galaxy.starCount = tempPoses;
            galaxy.stars = new StarData[tempPoses];
            if (tempPoses <= 0)
                return galaxy;

            float num1 = (float)rng.NextDouble();
            float num2 = (float)rng.NextDouble();
            float num3 = (float)rng.NextDouble();
            float num4 = (float)rng.NextDouble();

            int num5  = Mathf.CeilToInt((float)(0.01 * tempPoses + num1 * 0.300000011920929));
            int num6  = Mathf.CeilToInt((float)(0.01 * tempPoses + num2 * 0.300000011920929));
            int num7  = Mathf.CeilToInt((float)(0.0160000007599592 * tempPoses + num3 * 0.400000005960464));
            int num8  = Mathf.CeilToInt((float)(0.0130000002682209 * tempPoses + num4 * 1.39999997615814));
            int num9  = tempPoses - num5;
            int num10 = num9 - num6;
            int num11 = num10 - num7;
            int num12 = (num11 - 1) / num8;
            int num13 = num12 / 2;

            for (int index = 0; index < tempPoses; ++index)
            {
                int seed = rng.Next();
                if (index == 0)
                {
                    galaxy.stars[index] = DspFindSeed.StarGen.CreateBirthStar(galaxy, seed);
                }
                else
                {
                    ESpectrType needSpectr = ESpectrType.X;
                    if (index == 3) needSpectr = ESpectrType.M;
                    else if (index == num11 - 1) needSpectr = ESpectrType.O;

                    EStarType needtype = EStarType.MainSeqStar;
                    if (index % num12 == num13) needtype = EStarType.GiantStar;
                    if (index >= num9) needtype = EStarType.BlackHole;
                    else if (index >= num10) needtype = EStarType.NeutronStar;
                    else if (index >= num11) needtype = EStarType.WhiteDwarf;

                    galaxy.stars[index] = DspFindSeed.StarGen.CreateStar(galaxy, _tmpPoses[index], index + 1, seed, needtype, needSpectr);
                }
            }

            // 下面这段与原版保持一致：创建行星与 astro 数据
            AstroData[] astrosData = galaxy.astrosData;
            StarData[] stars = galaxy.stars;

            for (int i = 0; i < galaxy.astrosData.Length; ++i)
            {
                astrosData[i].uRot.w = 1f;
                astrosData[i].uRotNext.w = 1f;
            }

            for (int i = 0; i < tempPoses; ++i)
            {
                DspFindSeed.StarGen.CreateStarPlanets(galaxy, stars[i], gameDesc);
                astrosData[stars[i].id * 100].uPos = astrosData[stars[i].id * 100].uPosNext = stars[i].uPosition;
                astrosData[stars[i].id * 100].uRot = astrosData[stars[i].id * 100].uRotNext = Quaternion.identity;
                astrosData[stars[i].id * 100].uRadius = stars[i].physicsRadius;
            }

            galaxy.birthPlanetId = 0;
            if (tempPoses > 0)
            {
                StarData birthStar = stars[0];
                for (int i = 0; i < birthStar.planetCount; ++i)
                {
                    PlanetData planet = birthStar.planets[i];
                    var themeProto = DspFindSeed.LDB.themes.Select(planet.theme);
                    if (themeProto != null && themeProto.Distribute == EThemeDistribute.Birth)
                    {
                        galaxy.birthPlanetId = planet.id;
                        galaxy.birthStarId = birthStar.id;
                        break;
                    }
                }
            }

            DspFindSeed.UniverseGen.CreateGalaxyStarGraph(galaxy);
            DspFindSeed.PlanetGen.gasCoef = 1f;
            return galaxy;
        }

        /// <summary>
        /// 变体：候选点 pt 生成使用 FP64（sqrt/缩放/坐标），碰撞检测仍为 FP32。
        /// </summary>
        public static GalaxyData CreateGalaxy_PtFp64(DspFindSeed.GameDesc gameDesc)
        {
            int galaxyAlgo = gameDesc.galaxyAlgo;
            int galaxySeed = gameDesc.galaxySeed;
            int starCount = gameDesc.starCount;

            if (galaxyAlgo < 20200101 || galaxyAlgo > 20591231)
                throw new Exception("Wrong version of unigen algorithm!");

            DspFindSeed.PlanetGen.gasCoef = gameDesc.isRareResource ? 0.8f : 1f;

            var rng = new DotNet35Random(galaxySeed);
            int tempPoses = GenerateTempPosesF32(rng.Next(), starCount, 4, 2.0f, 2.3f, 3.5f, 0.18f, ptFp64: true, collisionFp64: false, randFp64: false);

            GalaxyData galaxy = new GalaxyData();
            galaxy.seed = galaxySeed;
            galaxy.starCount = tempPoses;
            galaxy.stars = new StarData[tempPoses];
            if (tempPoses <= 0)
                return galaxy;

            float num1 = (float)rng.NextDouble();
            float num2 = (float)rng.NextDouble();
            float num3 = (float)rng.NextDouble();
            float num4 = (float)rng.NextDouble();

            int num5 = Mathf.CeilToInt((float)(0.01 * tempPoses + num1 * 0.300000011920929));
            int num6 = Mathf.CeilToInt((float)(0.01 * tempPoses + num2 * 0.300000011920929));
            int num7 = Mathf.CeilToInt((float)(0.0160000007599592 * tempPoses + num3 * 0.400000005960464));
            int num8 = Mathf.CeilToInt((float)(0.0130000002682209 * tempPoses + num4 * 1.39999997615814));
            int num9 = tempPoses - num5;
            int num10 = num9 - num6;
            int num11 = num10 - num7;
            int num12 = (num11 - 1) / num8;
            int num13 = num12 / 2;

            for (int index = 0; index < tempPoses; ++index)
            {
                int seed = rng.Next();
                if (index == 0)
                {
                    galaxy.stars[index] = DspFindSeed.StarGen.CreateBirthStar(galaxy, seed);
                }
                else
                {
                    ESpectrType needSpectr = ESpectrType.X;
                    if (index == 3) needSpectr = ESpectrType.M;
                    else if (index == num11 - 1) needSpectr = ESpectrType.O;

                    EStarType needtype = EStarType.MainSeqStar;
                    if (index % num12 == num13) needtype = EStarType.GiantStar;
                    if (index >= num9) needtype = EStarType.BlackHole;
                    else if (index >= num10) needtype = EStarType.NeutronStar;
                    else if (index >= num11) needtype = EStarType.WhiteDwarf;

                    galaxy.stars[index] = DspFindSeed.StarGen.CreateStar(galaxy, _tmpPoses[index], index + 1, seed, needtype, needSpectr);
                }
            }

            AstroData[] astrosData = galaxy.astrosData;
            StarData[] stars = galaxy.stars;
            for (int i = 0; i < galaxy.astrosData.Length; ++i)
            {
                astrosData[i].uRot.w = 1f;
                astrosData[i].uRotNext.w = 1f;
            }

            for (int i = 0; i < tempPoses; ++i)
            {
                DspFindSeed.StarGen.CreateStarPlanets(galaxy, stars[i], gameDesc);
                astrosData[stars[i].id * 100].uPos = astrosData[stars[i].id * 100].uPosNext = stars[i].uPosition;
                astrosData[stars[i].id * 100].uRot = astrosData[stars[i].id * 100].uRotNext = Quaternion.identity;
                astrosData[stars[i].id * 100].uRadius = stars[i].physicsRadius;
            }

            galaxy.birthPlanetId = 0;
            if (tempPoses > 0)
            {
                StarData birthStar = stars[0];
                for (int i = 0; i < birthStar.planetCount; ++i)
                {
                    PlanetData planet = birthStar.planets[i];
                    var themeProto = DspFindSeed.LDB.themes.Select(planet.theme);
                    if (themeProto != null && themeProto.Distribute == EThemeDistribute.Birth)
                    {
                        galaxy.birthPlanetId = planet.id;
                        galaxy.birthStarId = birthStar.id;
                        break;
                    }
                }
            }

            DspFindSeed.UniverseGen.CreateGalaxyStarGraph(galaxy);
            DspFindSeed.PlanetGen.gasCoef = 1f;
            return galaxy;
        }

        /// <summary>
        /// 变体：候选点 pt 生成 FP64，碰撞检测也改回 FP64（更接近原版控制流）。
        /// </summary>
        public static GalaxyData CreateGalaxy_PtFp64_CollFp64(DspFindSeed.GameDesc gameDesc)
        {
            int galaxyAlgo = gameDesc.galaxyAlgo;
            int galaxySeed = gameDesc.galaxySeed;
            int starCount = gameDesc.starCount;

            if (galaxyAlgo < 20200101 || galaxyAlgo > 20591231)
                throw new Exception("Wrong version of unigen algorithm!");

            DspFindSeed.PlanetGen.gasCoef = gameDesc.isRareResource ? 0.8f : 1f;

            var rng = new DotNet35Random(galaxySeed);
            int tempPoses = GenerateTempPosesF32(rng.Next(), starCount, 4, 2.0f, 2.3f, 3.5f, 0.18f, ptFp64: true, collisionFp64: true, randFp64: false);

            GalaxyData galaxy = new GalaxyData();
            galaxy.seed = galaxySeed;
            galaxy.starCount = tempPoses;
            galaxy.stars = new StarData[tempPoses];
            if (tempPoses <= 0)
                return galaxy;

            float num1 = (float)rng.NextDouble();
            float num2 = (float)rng.NextDouble();
            float num3 = (float)rng.NextDouble();
            float num4 = (float)rng.NextDouble();

            int num5 = Mathf.CeilToInt((float)(0.01 * tempPoses + num1 * 0.300000011920929));
            int num6 = Mathf.CeilToInt((float)(0.01 * tempPoses + num2 * 0.300000011920929));
            int num7 = Mathf.CeilToInt((float)(0.0160000007599592 * tempPoses + num3 * 0.400000005960464));
            int num8 = Mathf.CeilToInt((float)(0.0130000002682209 * tempPoses + num4 * 1.39999997615814));
            int num9 = tempPoses - num5;
            int num10 = num9 - num6;
            int num11 = num10 - num7;
            int num12 = (num11 - 1) / num8;
            int num13 = num12 / 2;

            for (int index = 0; index < tempPoses; ++index)
            {
                int seed = rng.Next();
                if (index == 0)
                {
                    galaxy.stars[index] = DspFindSeed.StarGen.CreateBirthStar(galaxy, seed);
                }
                else
                {
                    ESpectrType needSpectr = ESpectrType.X;
                    if (index == 3) needSpectr = ESpectrType.M;
                    else if (index == num11 - 1) needSpectr = ESpectrType.O;

                    EStarType needtype = EStarType.MainSeqStar;
                    if (index % num12 == num13) needtype = EStarType.GiantStar;
                    if (index >= num9) needtype = EStarType.BlackHole;
                    else if (index >= num10) needtype = EStarType.NeutronStar;
                    else if (index >= num11) needtype = EStarType.WhiteDwarf;

                    galaxy.stars[index] = DspFindSeed.StarGen.CreateStar(galaxy, _tmpPoses[index], index + 1, seed, needtype, needSpectr);
                }
            }

            AstroData[] astrosData = galaxy.astrosData;
            StarData[] stars = galaxy.stars;
            for (int i = 0; i < galaxy.astrosData.Length; ++i)
            {
                astrosData[i].uRot.w = 1f;
                astrosData[i].uRotNext.w = 1f;
            }

            for (int i = 0; i < tempPoses; ++i)
            {
                DspFindSeed.StarGen.CreateStarPlanets(galaxy, stars[i], gameDesc);
                astrosData[stars[i].id * 100].uPos = astrosData[stars[i].id * 100].uPosNext = stars[i].uPosition;
                astrosData[stars[i].id * 100].uRot = astrosData[stars[i].id * 100].uRotNext = Quaternion.identity;
                astrosData[stars[i].id * 100].uRadius = stars[i].physicsRadius;
            }

            galaxy.birthPlanetId = 0;
            if (tempPoses > 0)
            {
                StarData birthStar = stars[0];
                for (int i = 0; i < birthStar.planetCount; ++i)
                {
                    PlanetData planet = birthStar.planets[i];
                    var themeProto = DspFindSeed.LDB.themes.Select(planet.theme);
                    if (themeProto != null && themeProto.Distribute == EThemeDistribute.Birth)
                    {
                        galaxy.birthPlanetId = planet.id;
                        galaxy.birthStarId = birthStar.id;
                        break;
                    }
                }
            }

            DspFindSeed.UniverseGen.CreateGalaxyStarGraph(galaxy);
            DspFindSeed.PlanetGen.gasCoef = 1f;
            return galaxy;
        }

        /// <summary>
        /// 变体：候选点 pt 生成 FP64，且随机输入也保持 FP64（不再把 NextDouble() 截断为 float）。
        /// 碰撞检测默认仍可选 FP32/FP64（此处用 FP32 更能隔离“随机截断”的影响）。
        /// </summary>
        public static GalaxyData CreateGalaxy_PtFp64_RandFp64(DspFindSeed.GameDesc gameDesc)
        {
            int galaxyAlgo = gameDesc.galaxyAlgo;
            int galaxySeed = gameDesc.galaxySeed;
            int starCount = gameDesc.starCount;

            if (galaxyAlgo < 20200101 || galaxyAlgo > 20591231)
                throw new Exception("Wrong version of unigen algorithm!");

            DspFindSeed.PlanetGen.gasCoef = gameDesc.isRareResource ? 0.8f : 1f;

            var rng = new DotNet35Random(galaxySeed);
            int tempPoses = GenerateTempPosesF32(rng.Next(), starCount, 4, 2.0f, 2.3f, 3.5f, 0.18f, ptFp64: true, collisionFp64: false, randFp64: true);

            GalaxyData galaxy = new GalaxyData();
            galaxy.seed = galaxySeed;
            galaxy.starCount = tempPoses;
            galaxy.stars = new StarData[tempPoses];
            if (tempPoses <= 0)
                return galaxy;

            float num1 = (float)rng.NextDouble();
            float num2 = (float)rng.NextDouble();
            float num3 = (float)rng.NextDouble();
            float num4 = (float)rng.NextDouble();

            int num5 = Mathf.CeilToInt((float)(0.01 * tempPoses + num1 * 0.300000011920929));
            int num6 = Mathf.CeilToInt((float)(0.01 * tempPoses + num2 * 0.300000011920929));
            int num7 = Mathf.CeilToInt((float)(0.0160000007599592 * tempPoses + num3 * 0.400000005960464));
            int num8 = Mathf.CeilToInt((float)(0.0130000002682209 * tempPoses + num4 * 1.39999997615814));
            int num9 = tempPoses - num5;
            int num10 = num9 - num6;
            int num11 = num10 - num7;
            int num12 = (num11 - 1) / num8;
            int num13 = num12 / 2;

            for (int index = 0; index < tempPoses; ++index)
            {
                int seed = rng.Next();
                if (index == 0)
                {
                    galaxy.stars[index] = DspFindSeed.StarGen.CreateBirthStar(galaxy, seed);
                }
                else
                {
                    ESpectrType needSpectr = ESpectrType.X;
                    if (index == 3) needSpectr = ESpectrType.M;
                    else if (index == num11 - 1) needSpectr = ESpectrType.O;

                    EStarType needtype = EStarType.MainSeqStar;
                    if (index % num12 == num13) needtype = EStarType.GiantStar;
                    if (index >= num9) needtype = EStarType.BlackHole;
                    else if (index >= num10) needtype = EStarType.NeutronStar;
                    else if (index >= num11) needtype = EStarType.WhiteDwarf;

                    galaxy.stars[index] = DspFindSeed.StarGen.CreateStar(galaxy, _tmpPoses[index], index + 1, seed, needtype, needSpectr);
                }
            }

            AstroData[] astrosData = galaxy.astrosData;
            StarData[] stars = galaxy.stars;
            for (int i = 0; i < galaxy.astrosData.Length; ++i)
            {
                astrosData[i].uRot.w = 1f;
                astrosData[i].uRotNext.w = 1f;
            }

            for (int i = 0; i < tempPoses; ++i)
            {
                DspFindSeed.StarGen.CreateStarPlanets(galaxy, stars[i], gameDesc);
                astrosData[stars[i].id * 100].uPos = astrosData[stars[i].id * 100].uPosNext = stars[i].uPosition;
                astrosData[stars[i].id * 100].uRot = astrosData[stars[i].id * 100].uRotNext = Quaternion.identity;
                astrosData[stars[i].id * 100].uRadius = stars[i].physicsRadius;
            }

            galaxy.birthPlanetId = 0;
            if (tempPoses > 0)
            {
                StarData birthStar = stars[0];
                for (int i = 0; i < birthStar.planetCount; ++i)
                {
                    PlanetData planet = birthStar.planets[i];
                    var themeProto = DspFindSeed.LDB.themes.Select(planet.theme);
                    if (themeProto != null && themeProto.Distribute == EThemeDistribute.Birth)
                    {
                        galaxy.birthPlanetId = planet.id;
                        galaxy.birthStarId = birthStar.id;
                        break;
                    }
                }
            }

            DspFindSeed.UniverseGen.CreateGalaxyStarGraph(galaxy);
            DspFindSeed.PlanetGen.gasCoef = 1f;
            return galaxy;
        }

        /// <summary>
        /// 变体：随机输入 FP64 + pt FP64 + 碰撞判定 FP64。
        /// </summary>
        public static GalaxyData CreateGalaxy_PtFp64_RandFp64_CollFp64(DspFindSeed.GameDesc gameDesc)
        {
            int galaxyAlgo = gameDesc.galaxyAlgo;
            int galaxySeed = gameDesc.galaxySeed;
            int starCount = gameDesc.starCount;

            if (galaxyAlgo < 20200101 || galaxyAlgo > 20591231)
                throw new Exception("Wrong version of unigen algorithm!");

            DspFindSeed.PlanetGen.gasCoef = gameDesc.isRareResource ? 0.8f : 1f;

            var rng = new DotNet35Random(galaxySeed);
            int tempPoses = GenerateTempPosesF32(rng.Next(), starCount, 4, 2.0f, 2.3f, 3.5f, 0.18f, ptFp64: true, collisionFp64: true, randFp64: true);

            GalaxyData galaxy = new GalaxyData();
            galaxy.seed = galaxySeed;
            galaxy.starCount = tempPoses;
            galaxy.stars = new StarData[tempPoses];
            if (tempPoses <= 0)
                return galaxy;

            float num1 = (float)rng.NextDouble();
            float num2 = (float)rng.NextDouble();
            float num3 = (float)rng.NextDouble();
            float num4 = (float)rng.NextDouble();

            int num5 = Mathf.CeilToInt((float)(0.01 * tempPoses + num1 * 0.300000011920929));
            int num6 = Mathf.CeilToInt((float)(0.01 * tempPoses + num2 * 0.300000011920929));
            int num7 = Mathf.CeilToInt((float)(0.0160000007599592 * tempPoses + num3 * 0.400000005960464));
            int num8 = Mathf.CeilToInt((float)(0.0130000002682209 * tempPoses + num4 * 1.39999997615814));
            int num9 = tempPoses - num5;
            int num10 = num9 - num6;
            int num11 = num10 - num7;
            int num12 = (num11 - 1) / num8;
            int num13 = num12 / 2;

            for (int index = 0; index < tempPoses; ++index)
            {
                int seed = rng.Next();
                if (index == 0)
                {
                    galaxy.stars[index] = DspFindSeed.StarGen.CreateBirthStar(galaxy, seed);
                }
                else
                {
                    ESpectrType needSpectr = ESpectrType.X;
                    if (index == 3) needSpectr = ESpectrType.M;
                    else if (index == num11 - 1) needSpectr = ESpectrType.O;

                    EStarType needtype = EStarType.MainSeqStar;
                    if (index % num12 == num13) needtype = EStarType.GiantStar;
                    if (index >= num9) needtype = EStarType.BlackHole;
                    else if (index >= num10) needtype = EStarType.NeutronStar;
                    else if (index >= num11) needtype = EStarType.WhiteDwarf;

                    galaxy.stars[index] = DspFindSeed.StarGen.CreateStar(galaxy, _tmpPoses[index], index + 1, seed, needtype, needSpectr);
                }
            }

            AstroData[] astrosData = galaxy.astrosData;
            StarData[] stars = galaxy.stars;
            for (int i = 0; i < galaxy.astrosData.Length; ++i)
            {
                astrosData[i].uRot.w = 1f;
                astrosData[i].uRotNext.w = 1f;
            }

            for (int i = 0; i < tempPoses; ++i)
            {
                DspFindSeed.StarGen.CreateStarPlanets(galaxy, stars[i], gameDesc);
                astrosData[stars[i].id * 100].uPos = astrosData[stars[i].id * 100].uPosNext = stars[i].uPosition;
                astrosData[stars[i].id * 100].uRot = astrosData[stars[i].id * 100].uRotNext = Quaternion.identity;
                astrosData[stars[i].id * 100].uRadius = stars[i].physicsRadius;
            }

            galaxy.birthPlanetId = 0;
            if (tempPoses > 0)
            {
                StarData birthStar = stars[0];
                for (int i = 0; i < birthStar.planetCount; ++i)
                {
                    PlanetData planet = birthStar.planets[i];
                    var themeProto = DspFindSeed.LDB.themes.Select(planet.theme);
                    if (themeProto != null && themeProto.Distribute == EThemeDistribute.Birth)
                    {
                        galaxy.birthPlanetId = planet.id;
                        galaxy.birthStarId = birthStar.id;
                        break;
                    }
                }
            }

            DspFindSeed.UniverseGen.CreateGalaxyStarGraph(galaxy);
            DspFindSeed.PlanetGen.gasCoef = 1f;
            return galaxy;
        }

        /// <summary>
        /// 变体：随机输入 FP64 + pt FP64 + 参数常量也用 FP64（minDist/minStepLen/maxStepLen/flatten 为 double）。
        /// 碰撞判定可选 FP32/FP64，这里提供两个入口方便对比。
        /// </summary>
        public static GalaxyData CreateGalaxy_PtFp64_RandFp64_ParamsFp64(DspFindSeed.GameDesc gameDesc)
        {
            return CreateGalaxy_PtFp64_RandFp64_ParamsFp64_Impl(gameDesc, collisionFp64: false);
        }

        public static GalaxyData CreateGalaxy_PtFp64_RandFp64_ParamsFp64_CollFp64(DspFindSeed.GameDesc gameDesc)
        {
            return CreateGalaxy_PtFp64_RandFp64_ParamsFp64_Impl(gameDesc, collisionFp64: true);
        }

        private static GalaxyData CreateGalaxy_PtFp64_RandFp64_ParamsFp64_Impl(DspFindSeed.GameDesc gameDesc, bool collisionFp64)
        {
            int galaxyAlgo = gameDesc.galaxyAlgo;
            int galaxySeed = gameDesc.galaxySeed;
            int starCount = gameDesc.starCount;

            if (galaxyAlgo < 20200101 || galaxyAlgo > 20591231)
                throw new Exception("Wrong version of unigen algorithm!");

            DspFindSeed.PlanetGen.gasCoef = gameDesc.isRareResource ? 0.8f : 1f;

            var rng = new DotNet35Random(galaxySeed);

            // 注意：这里把参数常量改为 double（更贴近原版 UniverseGen）
            int tempPoses = GenerateTempPoses_ParamsFp64(
                rng.Next(),
                starCount,
                iterCount: 4,
                minDist: 2.0,
                minStepLen: 2.3,
                maxStepLen: 3.5,
                flatten: 0.18,
                collisionFp64: collisionFp64);

            GalaxyData galaxy = new GalaxyData();
            galaxy.seed = galaxySeed;
            galaxy.starCount = tempPoses;
            galaxy.stars = new StarData[tempPoses];
            if (tempPoses <= 0)
                return galaxy;

            float num1 = (float)rng.NextDouble();
            float num2 = (float)rng.NextDouble();
            float num3 = (float)rng.NextDouble();
            float num4 = (float)rng.NextDouble();

            int num5 = Mathf.CeilToInt((float)(0.01 * tempPoses + num1 * 0.300000011920929));
            int num6 = Mathf.CeilToInt((float)(0.01 * tempPoses + num2 * 0.300000011920929));
            int num7 = Mathf.CeilToInt((float)(0.0160000007599592 * tempPoses + num3 * 0.400000005960464));
            int num8 = Mathf.CeilToInt((float)(0.0130000002682209 * tempPoses + num4 * 1.39999997615814));
            int num9 = tempPoses - num5;
            int num10 = num9 - num6;
            int num11 = num10 - num7;
            int num12 = (num11 - 1) / num8;
            int num13 = num12 / 2;

            for (int index = 0; index < tempPoses; ++index)
            {
                int seed = rng.Next();
                if (index == 0)
                {
                    galaxy.stars[index] = DspFindSeed.StarGen.CreateBirthStar(galaxy, seed);
                }
                else
                {
                    ESpectrType needSpectr = ESpectrType.X;
                    if (index == 3) needSpectr = ESpectrType.M;
                    else if (index == num11 - 1) needSpectr = ESpectrType.O;

                    EStarType needtype = EStarType.MainSeqStar;
                    if (index % num12 == num13) needtype = EStarType.GiantStar;
                    if (index >= num9) needtype = EStarType.BlackHole;
                    else if (index >= num10) needtype = EStarType.NeutronStar;
                    else if (index >= num11) needtype = EStarType.WhiteDwarf;

                    galaxy.stars[index] = DspFindSeed.StarGen.CreateStar(galaxy, _tmpPoses[index], index + 1, seed, needtype, needSpectr);
                }
            }

            AstroData[] astrosData = galaxy.astrosData;
            StarData[] stars = galaxy.stars;
            for (int i = 0; i < galaxy.astrosData.Length; ++i)
            {
                astrosData[i].uRot.w = 1f;
                astrosData[i].uRotNext.w = 1f;
            }

            for (int i = 0; i < tempPoses; ++i)
            {
                DspFindSeed.StarGen.CreateStarPlanets(galaxy, stars[i], gameDesc);
                astrosData[stars[i].id * 100].uPos = astrosData[stars[i].id * 100].uPosNext = stars[i].uPosition;
                astrosData[stars[i].id * 100].uRot = astrosData[stars[i].id * 100].uRotNext = Quaternion.identity;
                astrosData[stars[i].id * 100].uRadius = stars[i].physicsRadius;
            }

            galaxy.birthPlanetId = 0;
            if (tempPoses > 0)
            {
                StarData birthStar = stars[0];
                for (int i = 0; i < birthStar.planetCount; ++i)
                {
                    PlanetData planet = birthStar.planets[i];
                    var themeProto = DspFindSeed.LDB.themes.Select(planet.theme);
                    if (themeProto != null && themeProto.Distribute == EThemeDistribute.Birth)
                    {
                        galaxy.birthPlanetId = planet.id;
                        galaxy.birthStarId = birthStar.id;
                        break;
                    }
                }
            }

            DspFindSeed.UniverseGen.CreateGalaxyStarGraph(galaxy);
            DspFindSeed.PlanetGen.gasCoef = 1f;
            return galaxy;
        }

        private static int GenerateTempPosesF32(
            int seed,
            int targetCount,
            int iterCount,
            float minDist,
            float minStepLen,
            float maxStepLen,
            float flatten,
            bool ptFp64,
            bool collisionFp64,
            bool randFp64)
        {
            _tmpPoses.Clear();
            _tmpDrunk.Clear();
            if (iterCount < 1) iterCount = 1;
            else if (iterCount > 16) iterCount = 16;

            RandomPosesF32(seed, targetCount * iterCount, minDist, minStepLen, maxStepLen, flatten, ptFp64, collisionFp64, randFp64);
            for (int i = _tmpPoses.Count - 1; i >= 0; --i)
            {
                if (i % iterCount != 0)
                    _tmpPoses.RemoveAt(i);
                if (_tmpPoses.Count <= targetCount)
                    break;
            }
            return _tmpPoses.Count;
        }

        private static int GenerateTempPoses_ParamsFp64(
            int seed,
            int targetCount,
            int iterCount,
            double minDist,
            double minStepLen,
            double maxStepLen,
            double flatten,
            bool collisionFp64)
        {
            _tmpPoses.Clear();
            _tmpDrunk.Clear();
            if (iterCount < 1) iterCount = 1;
            else if (iterCount > 16) iterCount = 16;

            RandomPoses_ParamsFp64(seed, targetCount * iterCount, minDist, minStepLen, maxStepLen, flatten, collisionFp64);
            for (int i = _tmpPoses.Count - 1; i >= 0; --i)
            {
                if (i % iterCount != 0)
                    _tmpPoses.RemoveAt(i);
                if (_tmpPoses.Count <= targetCount)
                    break;
            }
            return _tmpPoses.Count;
        }

        private static void RandomPoses_ParamsFp64(
            int seed,
            int maxCount,
            double minDist,
            double minStepLen,
            double maxStepLen,
            double flatten,
            bool collisionFp64)
        {
            var rng = new DotNet35Random(seed);
            double num1 = rng.NextDouble();
            _tmpPoses.Add(VectorLF3.zero);

            int num2 = 6, num3 = 8;
            if (num2 < 1) num2 = 1;
            if (num3 < 1) num3 = 1;

            double num4 = (num3 - num2);
            int num5 = (int)(num1 * num4 + num2);

            for (int i = 0; i < num5; ++i)
            {
                int tries = 0;
                while (tries++ < 256)
                {
                    // 随机输入保持 FP64
                    double xd = rng.NextDouble() * 2.0 - 1.0;
                    double yd = (rng.NextDouble() * 2.0 - 1.0) * flatten;
                    double zd = rng.NextDouble() * 2.0 - 1.0;
                    double num10d = rng.NextDouble();

                    double dd = xd * xd + yd * yd + zd * zd;
                    if (dd <= 1.0 && dd >= 1e-8)
                    {
                        double invLen = 1.0 / Math.Sqrt(dd);
                        double scale = (num10d * (maxStepLen - minStepLen) + minDist) * invLen;
                        var pt = new VectorLF3(xd * scale, yd * scale, zd * scale);

                        bool hit = collisionFp64 ? CheckCollisionFp64(_tmpPoses, pt, (float)minDist) : CheckCollisionF32(_tmpPoses, pt, (float)minDist);
                        if (!hit)
                        {
                            _tmpDrunk.Add(pt);
                            _tmpPoses.Add(pt);
                            if (_tmpPoses.Count >= maxCount)
                                return;
                            break;
                        }
                    }
                }
            }

            int outer = 0;
            while (outer++ < 256)
            {
                for (int i = 0; i < _tmpDrunk.Count; ++i)
                {
                    if (rng.NextDouble() <= 0.7)
                    {
                        int tries = 0;
                        while (tries++ < 256)
                        {
                            double xd = rng.NextDouble() * 2.0 - 1.0;
                            double yd = (rng.NextDouble() * 2.0 - 1.0) * flatten;
                            double zd = rng.NextDouble() * 2.0 - 1.0;
                            double num18d = rng.NextDouble();

                            double dd = xd * xd + yd * yd + zd * zd;
                            if (dd <= 1.0 && dd >= 1e-8)
                            {
                                double invLen = 1.0 / Math.Sqrt(dd);
                                double scale = (num18d * (maxStepLen - minStepLen) + minDist) * invLen;
                                var basePt = _tmpDrunk[i];
                                var pt = new VectorLF3(basePt.x + xd * scale, basePt.y + yd * scale, basePt.z + zd * scale);

                                bool hit = collisionFp64 ? CheckCollisionFp64(_tmpPoses, pt, (float)minDist) : CheckCollisionF32(_tmpPoses, pt, (float)minDist);
                                if (!hit)
                                {
                                    _tmpDrunk[i] = pt;
                                    _tmpPoses.Add(pt);
                                    if (_tmpPoses.Count >= maxCount)
                                        return;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        private static void RandomPosesF32(
            int seed,
            int maxCount,
            float minDist,
            float minStepLen,
            float maxStepLen,
            float flatten,
            bool ptFp64,
            bool collisionFp64,
            bool randFp64)
        {
            var rng = new DotNet35Random(seed);
            float num1 = (float)rng.NextDouble();
            _tmpPoses.Add(VectorLF3.zero);

            int num2 = 6, num3 = 8;
            if (num2 < 1) num2 = 1;
            if (num3 < 1) num3 = 1;

            float num4 = (num3 - num2);
            int num5 = (int)(num1 * num4 + num2);

            for (int i = 0; i < num5; ++i)
            {
                int tries = 0;
                while (tries++ < 256)
                {
                    double xd = randFp64 ? (rng.NextDouble() * 2.0 - 1.0) : ((double)(float)rng.NextDouble() * 2.0 - 1.0);
                    double yd = randFp64 ? ((rng.NextDouble() * 2.0 - 1.0) * (double)flatten) : (((double)(float)rng.NextDouble() * 2.0 - 1.0) * (double)flatten);
                    double zd = randFp64 ? (rng.NextDouble() * 2.0 - 1.0) : ((double)(float)rng.NextDouble() * 2.0 - 1.0);
                    double num10d = randFp64 ? rng.NextDouble() : (double)(float)rng.NextDouble();

                    double dd = xd * xd + yd * yd + zd * zd;
                    if (dd <= 1.0 && dd >= 1e-8)
                    {
                        VectorLF3 pt;
                        if (!ptFp64)
                        {
                            float x = (float)xd;
                            float y = (float)yd;
                            float z = (float)zd;
                            float num10 = (float)num10d;
                            float d = (float)dd;
                            float invLen = 1f / (float)Math.Sqrt(d);
                            float num12 = (num10 * (maxStepLen - minStepLen) + minDist) * invLen;
                            pt = new VectorLF3(x * num12, y * num12, z * num12);
                        }
                        else
                        {
                            // pt 生成用 FP64（更接近原版），但仍然沿用 FP32 的随机输入与 d（只修复归一化/缩放/坐标乘法的精度）
                            double invLen = 1.0 / Math.Sqrt(dd);
                            double scale = (num10d * (double)(maxStepLen - minStepLen) + (double)minDist) * invLen;
                            pt = new VectorLF3(xd * scale, yd * scale, zd * scale);
                        }
                        if (!(collisionFp64 ? CheckCollisionFp64(_tmpPoses, pt, minDist) : CheckCollisionF32(_tmpPoses, pt, minDist)))
                        {
                            _tmpDrunk.Add(pt);
                            _tmpPoses.Add(pt);
                            if (_tmpPoses.Count >= maxCount)
                                return;
                            break;
                        }
                    }
                }
            }

            int outer = 0;
            while (outer++ < 256)
            {
                for (int i = 0; i < _tmpDrunk.Count; ++i)
                {
                    if (rng.NextDouble() <= 0.7)
                    {
                        int tries = 0;
                        while (tries++ < 256)
                        {
                            double xd = randFp64 ? (rng.NextDouble() * 2.0 - 1.0) : ((double)(float)rng.NextDouble() * 2.0 - 1.0);
                            double yd = randFp64 ? ((rng.NextDouble() * 2.0 - 1.0) * (double)flatten) : (((double)(float)rng.NextDouble() * 2.0 - 1.0) * (double)flatten);
                            double zd = randFp64 ? (rng.NextDouble() * 2.0 - 1.0) : ((double)(float)rng.NextDouble() * 2.0 - 1.0);
                            double num18d = randFp64 ? rng.NextDouble() : (double)(float)rng.NextDouble();

                            double dd = xd * xd + yd * yd + zd * zd;
                            if (dd <= 1.0 && dd >= 1e-8)
                            {
                                var basePt = _tmpDrunk[i];
                                VectorLF3 pt;
                                if (!ptFp64)
                                {
                                    float x = (float)xd;
                                    float y = (float)yd;
                                    float z = (float)zd;
                                    float num18 = (float)num18d;
                                    float d = (float)dd;
                                    float invLen = 1f / (float)Math.Sqrt(d);
                                    float num20 = (num18 * (maxStepLen - minStepLen) + minDist) * invLen;
                                    pt = new VectorLF3(basePt.x + x * num20, basePt.y + y * num20, basePt.z + z * num20);
                                }
                                else
                                {
                                    double invLen = 1.0 / Math.Sqrt(dd);
                                    double scale = (num18d * (double)(maxStepLen - minStepLen) + (double)minDist) * invLen;
                                    pt = new VectorLF3(basePt.x + xd * scale, basePt.y + yd * scale, basePt.z + zd * scale);
                                }
                                if (!(collisionFp64 ? CheckCollisionFp64(_tmpPoses, pt, minDist) : CheckCollisionF32(_tmpPoses, pt, minDist)))
                                {
                                    _tmpDrunk[i] = pt;
                                    _tmpPoses.Add(pt);
                                    if (_tmpPoses.Count >= maxCount)
                                        return;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        private static bool CheckCollisionF32(List<VectorLF3> pts, VectorLF3 pt, float minDist)
        {
            float min2 = minDist * minDist;
            for (int i = 0; i < pts.Count; i++)
            {
                var p = pts[i];
                float dx = (float)(pt.x - p.x);
                float dy = (float)(pt.y - p.y);
                float dz = (float)(pt.z - p.z);
                if (dx * dx + dy * dy + dz * dz < min2)
                    return true;
            }
            return false;
        }

        private static bool CheckCollisionFp64(List<VectorLF3> pts, VectorLF3 pt, float minDist)
        {
            double min2 = (double)minDist * (double)minDist;
            for (int i = 0; i < pts.Count; i++)
            {
                var p = pts[i];
                double dx = pt.x - p.x;
                double dy = pt.y - p.y;
                double dz = pt.z - p.z;
                if (dx * dx + dy * dy + dz * dz < min2)
                    return true;
            }
            return false;
        }
    }
}

