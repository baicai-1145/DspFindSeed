using System;
using System.Collections.Generic;
using UnityEngine;

namespace SeedCli
{
    /// <summary>
    /// 实验用途：保持“恒星点位/恒星属性”完全按原版 UniverseGen.CreateGalaxy 生成（FP64），
    /// 仅把“行星生成”替换为 StarGenPlanetsF32 + PlanetGenF32（更多 FP32 化）。
    /// </summary>
    internal static class UniverseGenPlanetsF32
    {
        private static List<VectorLF3> _tmpPoses;
        private static List<VectorLF3> _tmpDrunk;

        public static GalaxyData CreateGalaxy(global::DspFindSeed.GameDesc gameDesc)
        {
            int galaxyAlgo = gameDesc.galaxyAlgo;
            int galaxySeed = gameDesc.galaxySeed;
            int starCount = gameDesc.starCount;
            if (galaxyAlgo < 20200101 || galaxyAlgo > 20591231)
                throw new Exception("Wrong version of unigen algorithm!");

            global::DspFindSeed.PlanetGen.gasCoef = gameDesc.isRareResource ? 0.8f : 1f;

            DotNet35Random rng = new DotNet35Random(galaxySeed);
            int tempPoses = GenerateTempPoses(rng.Next(), starCount, 4, 2.0, 2.3, 3.5, 0.18);

            GalaxyData galaxy = new GalaxyData();
            galaxy.seed = galaxySeed;
            galaxy.starCount = tempPoses;
            galaxy.stars = new StarData[tempPoses];
            if (tempPoses <= 0) return galaxy;

            float num1 = (float)rng.NextDouble();
            float num2 = (float)rng.NextDouble();
            float num3 = (float)rng.NextDouble();
            float num4 = (float)rng.NextDouble();

            int num5 = Mathf.CeilToInt((float)(0.00999999977648258 * tempPoses + num1 * 0.300000011920929));
            int num6 = Mathf.CeilToInt((float)(0.00999999977648258 * tempPoses + num2 * 0.300000011920929));
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
                    galaxy.stars[index] = global::DspFindSeed.StarGen.CreateBirthStar(galaxy, seed);
                }
                else
                {
                    ESpectrType needSpectr = ESpectrType.X;
                    if (index == 3) needSpectr = ESpectrType.M;
                    else if (index == num11 - 1) needSpectr = ESpectrType.O;

                    EStarType needType = EStarType.MainSeqStar;
                    if (index % num12 == num13) needType = EStarType.GiantStar;
                    if (index >= num9) needType = EStarType.BlackHole;
                    else if (index >= num10) needType = EStarType.NeutronStar;
                    else if (index >= num11) needType = EStarType.WhiteDwarf;

                    galaxy.stars[index] = global::DspFindSeed.StarGen.CreateStar(galaxy, _tmpPoses[index], index + 1, seed, needType, needSpectr);
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
                StarGenPlanetsF32.CreateStarPlanetsF32(galaxy, stars[i], gameDesc);
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
                    var themeProto = global::DspFindSeed.LDB.themes.Select(planet.theme);
                    if (themeProto != null && themeProto.Distribute == EThemeDistribute.Birth)
                    {
                        galaxy.birthPlanetId = planet.id;
                        galaxy.birthStarId = birthStar.id;
                        break;
                    }
                }
            }

            global::DspFindSeed.UniverseGen.CreateGalaxyStarGraph(galaxy);
            global::DspFindSeed.PlanetGen.gasCoef = 1f;
            return galaxy;
        }

        private static int GenerateTempPoses(int seed, int targetCount, int iterCount, double minDist, double minStepLen, double maxStepLen, double flatten)
        {
            if (_tmpPoses == null)
            {
                _tmpPoses = new List<VectorLF3>();
                _tmpDrunk = new List<VectorLF3>();
            }
            else
            {
                _tmpPoses.Clear();
                _tmpDrunk.Clear();
            }

            if (iterCount < 1) iterCount = 1;
            else if (iterCount > 16) iterCount = 16;

            RandomPoses(seed, targetCount * iterCount, minDist, minStepLen, maxStepLen, flatten);

            for (int index = _tmpPoses.Count - 1; index >= 0; --index)
            {
                if (index % iterCount != 0) _tmpPoses.RemoveAt(index);
                if (_tmpPoses.Count <= targetCount) break;
            }
            return _tmpPoses.Count;
        }

        private static void RandomPoses(int seed, int maxCount, double minDist, double minStepLen, double maxStepLen, double flatten)
        {
            DotNet35Random rng = new DotNet35Random(seed);
            double num1 = rng.NextDouble();
            _tmpPoses.Add(VectorLF3.zero);

            int num2 = 6;
            int num3 = 8;
            if (num2 < 1) num2 = 1;
            if (num3 < 1) num3 = 1;

            double num4 = num3 - num2;
            int num5 = (int)(num1 * num4 + num2);

            for (int index = 0; index < num5; ++index)
            {
                int n = 0;
                while (n++ < 256)
                {
                    double x = rng.NextDouble() * 2.0 - 1.0;
                    double y = (rng.NextDouble() * 2.0 - 1.0) * flatten;
                    double z = rng.NextDouble() * 2.0 - 1.0;
                    double t = rng.NextDouble();
                    double d = x * x + y * y + z * z;
                    if (d <= 1.0 && d >= 1E-08)
                    {
                        double len = Math.Sqrt(d);
                        double k = (t * (maxStepLen - minStepLen) + minDist) / len;
                        VectorLF3 pt = new VectorLF3(x * k, y * k, z * k);
                        if (!CheckCollision(_tmpPoses, pt, minDist))
                        {
                            _tmpDrunk.Add(pt);
                            _tmpPoses.Add(pt);
                            if (_tmpPoses.Count >= maxCount) return;
                            break;
                        }
                    }
                }
            }

            int num13 = 0;
            while (num13++ < 256)
            {
                for (int index = 0; index < _tmpDrunk.Count; ++index)
                {
                    if (rng.NextDouble() <= 0.7)
                    {
                        int n = 0;
                        while (n++ < 256)
                        {
                            double x = rng.NextDouble() * 2.0 - 1.0;
                            double y = (rng.NextDouble() * 2.0 - 1.0) * flatten;
                            double z = rng.NextDouble() * 2.0 - 1.0;
                            double t = rng.NextDouble();
                            double d = x * x + y * y + z * z;
                            if (d <= 1.0 && d >= 1E-08)
                            {
                                double len = Math.Sqrt(d);
                                double k = (t * (maxStepLen - minStepLen) + minDist) / len;
                                VectorLF3 pt = new VectorLF3(
                                    _tmpDrunk[index].x + x * k,
                                    _tmpDrunk[index].y + y * k,
                                    _tmpDrunk[index].z + z * k
                                );
                                if (!CheckCollision(_tmpPoses, pt, minDist))
                                {
                                    _tmpDrunk[index] = pt;
                                    _tmpPoses.Add(pt);
                                    if (_tmpPoses.Count >= maxCount) return;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        private static bool CheckCollision(List<VectorLF3> pts, VectorLF3 pt, double minDist)
        {
            double r2 = minDist * minDist;
            foreach (var p in pts)
            {
                double dx = pt.x - p.x;
                double dy = pt.y - p.y;
                double dz = pt.z - p.z;
                if (dx * dx + dy * dy + dz * dz < r2) return true;
            }
            return false;
        }
    }
}

