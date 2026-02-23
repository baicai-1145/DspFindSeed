using System.Collections.Concurrent;

namespace SeedCli
{
    internal static class MixObjectPool
    {
        private static readonly ConcurrentBag<PlanetData> PlanetPool = new ConcurrentBag<PlanetData>();
        private static readonly ConcurrentDictionary<int, ConcurrentBag<PlanetData[]>> PlanetArrayPools =
            new ConcurrentDictionary<int, ConcurrentBag<PlanetData[]>>();
        private static readonly ConcurrentDictionary<int, ConcurrentBag<CudaPlanetNative.PlanetCoreF32Out[]>> CoreArrayPools =
            new ConcurrentDictionary<int, ConcurrentBag<CudaPlanetNative.PlanetCoreF32Out[]>>();

        public static PlanetData RentPlanet()
        {
            if (!PlanetPool.TryTake(out var p) || p == null)
                p = new PlanetData();
            return p;
        }

        public static void ReturnPlanet(PlanetData planet)
        {
            if (planet == null)
                return;

            // 仅清理关键引用，避免跨 chunk 持有整棵对象图。
            planet.galaxy = null;
            planet.star = null;
            planet.orbitAroundPlanet = null;
            planet.gasItems = null;
            planet.gasSpeeds = null;
            planet.gasHeatValues = null;
            PlanetPool.Add(planet);
        }

        public static PlanetData[] RentPlanetArray(int count)
        {
            if (count < 0) count = 0;
            if (count == 0)
                return System.Array.Empty<PlanetData>();

            var bag = PlanetArrayPools.GetOrAdd(count, _ => new ConcurrentBag<PlanetData[]>());
            if (bag.TryTake(out var arr) && arr != null && arr.Length == count)
                return arr;
            return new PlanetData[count];
        }

        public static void ReturnPlanetArray(PlanetData[] arr)
        {
            if (arr == null || arr.Length == 0)
                return;
            System.Array.Clear(arr, 0, arr.Length);
            var bag = PlanetArrayPools.GetOrAdd(arr.Length, _ => new ConcurrentBag<PlanetData[]>());
            bag.Add(arr);
        }

        public static CudaPlanetNative.PlanetCoreF32Out[] RentCoreArray(int count)
        {
            if (count < 0) count = 0;
            if (count == 0)
                return System.Array.Empty<CudaPlanetNative.PlanetCoreF32Out>();

            var bag = CoreArrayPools.GetOrAdd(count, _ => new ConcurrentBag<CudaPlanetNative.PlanetCoreF32Out[]>());
            if (bag.TryTake(out var arr) && arr != null && arr.Length == count)
                return arr;
            return new CudaPlanetNative.PlanetCoreF32Out[count];
        }

        public static void ReturnCoreArray(CudaPlanetNative.PlanetCoreF32Out[] arr)
        {
            if (arr == null || arr.Length == 0)
                return;
            System.Array.Clear(arr, 0, arr.Length);
            var bag = CoreArrayPools.GetOrAdd(arr.Length, _ => new ConcurrentBag<CudaPlanetNative.PlanetCoreF32Out[]>());
            bag.Add(arr);
        }

        public static void ReleaseGalaxy(GalaxyData g)
        {
            if (g == null || g.stars == null)
                return;

            for (int si = 0; si < g.stars.Length; ++si)
            {
                var star = g.stars[si];
                if (star == null || star.planets == null)
                    continue;

                var planets = star.planets;
                for (int pi = 0; pi < planets.Length; ++pi)
                    ReturnPlanet(planets[pi]);

                star.planets = null;
                star.planetCount = 0;
                ReturnPlanetArray(planets);
            }
        }

        public static void ReleaseGalaxies(GalaxyData[] galaxies, int count)
        {
            if (galaxies == null || count <= 0)
                return;
            int max = count < galaxies.Length ? count : galaxies.Length;
            for (int i = 0; i < max; ++i)
            {
                ReleaseGalaxy(galaxies[i]);
                galaxies[i] = null;
            }
        }
    }
}
