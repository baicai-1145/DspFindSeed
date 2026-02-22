using System;
using UnityEngine;

namespace DspFindSeed
{
public static class StarGen
{
  public static float[] orbitRadius = new float[17]
  {
    0.0f,
    0.4f,
    0.7f,
    1f,
    1.4f,
    1.9f,
    2.5f,
    3.3f,
    4.3f,
    5.5f,
    6.9f,
    8.4f,
    10f,
    11.7f,
    13.5f,
    15.4f,
    17.5f
  };
  public static float specifyBirthStarMass = 0.0f;
  public static float specifyBirthStarAge = 0.0f;
  [ThreadStatic] private static double[] pGas;
  private const double PI = 3.14159265358979;

  public static StarData CreateStar(
    GalaxyData galaxy,
    VectorLF3 pos,
    int id,
    int seed,
    EStarType needtype,
    ESpectrType needSpectr = ESpectrType.X)
  {
    StarData star = new StarData()
    {
      galaxy = galaxy,
      index = id - 1
    };
    star.level = galaxy.starCount <= 1 ? 0.0f : (float) star.index / (float) (galaxy.starCount - 1);
    star.id = id;
    star.seed = seed;
    DotNet35Random dotNet35Random1 = new DotNet35Random(seed);
    int seed1 = dotNet35Random1.Next();
    int Seed = dotNet35Random1.Next();
    star.position = pos;
    float num1 = (float) pos.magnitude / 32f;
    if ((double) num1 > 1.0)
      num1 = Mathf.Log(Mathf.Log(Mathf.Log(Mathf.Log(Mathf.Log(num1) + 1f) + 1f) + 1f) + 1f) + 1f;
    star.resourceCoef = Mathf.Pow(7f, num1) * 0.6f;
    DotNet35Random dotNet35Random2 = new DotNet35Random(Seed);
    double r1 = dotNet35Random2.NextDouble();
    double r2 = dotNet35Random2.NextDouble();
    double num2 = dotNet35Random2.NextDouble();
    double rn = dotNet35Random2.NextDouble();
    double rt = dotNet35Random2.NextDouble();
    double num3 = (dotNet35Random2.NextDouble() - 0.5) * 0.2;
    double num4 = dotNet35Random2.NextDouble() * 0.2 + 0.9;
    double y = dotNet35Random2.NextDouble() * 0.4 - 0.2;
    double num5 = Math.Pow(2.0, y);
    float num6 = Mathf.Lerp(-0.98f, 0.88f, star.level);
    float averageValue = (double) num6 >= 0.0 ? num6 + 0.65f : num6 - 0.65f;
    float standardDeviation = 0.33f;
    if (needtype == EStarType.GiantStar)
    {
      averageValue = y > -0.08 ? -1.5f : 1.6f;
      standardDeviation = 0.3f;
    }
    float num7 = StarGen.RandNormal(averageValue, standardDeviation, r1, r2);
    switch (needSpectr)
    {
      case ESpectrType.M:
        num7 = -3f;
        break;
      case ESpectrType.O:
        num7 = 3f;
        break;
    }
    float p1 = (float) ((double) Mathf.Clamp((double) num7 <= 0.0 ? num7 * 1f : num7 * 2f, -2.4f, 4.65f) + num3 + 1.0);
    switch (needtype)
    {
      case EStarType.WhiteDwarf:
        star.mass = (float) (1.0 + r2 * 5.0);
        break;
      case EStarType.NeutronStar:
        star.mass = (float) (7.0 + r1 * 11.0);
        break;
      case EStarType.BlackHole:
        star.mass = (float) (18.0 + r1 * r2 * 30.0);
        break;
      default:
        star.mass = Mathf.Pow(2f, p1);
        break;
    }
    double d = 5.0;
    if ((double) star.mass < 2.0)
      d = 2.0 + 0.4 * (1.0 - (double) star.mass);
    star.lifetime = (float) (10000.0 * Math.Pow(0.1, Math.Log10((double) star.mass * 0.5) / Math.Log10(d) + 1.0) * num4);
    switch (needtype)
    {
      case EStarType.GiantStar:
        star.lifetime = (float) (10000.0 * Math.Pow(0.1, Math.Log10((double) star.mass * 0.58) / Math.Log10(d) + 1.0) * num4);
        star.age = (float) (num2 * 0.0399999991059303 + 0.959999978542328);
        break;
      case EStarType.WhiteDwarf:
      case EStarType.NeutronStar:
      case EStarType.BlackHole:
        star.age = (float) (num2 * 0.400000005960464 + 1.0);
        if (needtype == EStarType.WhiteDwarf)
        {
          star.lifetime += 10000f;
          break;
        }
        if (needtype == EStarType.NeutronStar)
        {
          star.lifetime += 1000f;
          break;
        }
        break;
      default:
        star.age = (double) star.mass >= 0.5 ? ((double) star.mass >= 0.8 ? (float) (num2 * 0.699999988079071 + 0.200000002980232) : (float) (num2 * 0.400000005960464 + 0.100000001490116)) : (float) (num2 * 0.119999997317791 + 0.0199999995529652);
        break;
    }
    float num8 = star.lifetime * star.age;
    if ((double) num8 > 5000.0)
      num8 = (float) (((double) Mathf.Log(num8 / 5000f) + 1.0) * 5000.0);
    if ((double) num8 > 8000.0)
      num8 = (float) (((double) Mathf.Log(Mathf.Log(Mathf.Log(num8 / 8000f) + 1f) + 1f) + 1.0) * 8000.0);
    star.lifetime = num8 / star.age;
    float num9 = (float) (1.0 - (double) Mathf.Pow(Mathf.Clamp01(star.age), 20f) * 0.5) * star.mass;
    star.temperature = (float) (Math.Pow((double) num9, 0.56 + 0.14 / (Math.Log10((double) num9 + 4.0) / Math.Log10(5.0))) * 4450.0 + 1300.0);
    double num10 = Math.Log10(((double) star.temperature - 1300.0) / 4500.0) / Math.Log10(2.6) - 0.5;
    if (num10 < 0.0)
      num10 *= 4.0;
    if (num10 > 2.0)
      num10 = 2.0;
    else if (num10 < -4.0)
      num10 = -4.0;
    star.spectr = (ESpectrType) Mathf.RoundToInt((float) num10 + 4f);
    star.color = Mathf.Clamp01((float) ((num10 + 3.5) * 0.200000002980232));
    star.classFactor = (float) num10;
    star.luminosity = Mathf.Pow(num9, 0.7f);
    star.radius = (float) (Math.Pow((double) star.mass, 0.4) * num5);
    star.acdiskRadius = 0.0f;
    float p2 = (float) num10 + 2f;
    star.habitableRadius = Mathf.Pow(1.7f, p2) + 0.25f * Mathf.Min(1f, star.orbitScaler);
    star.lightBalanceRadius = Mathf.Pow(1.7f, p2);
    star.orbitScaler = Mathf.Pow(1.35f, p2);
    if ((double) star.orbitScaler < 1.0)
      star.orbitScaler = Mathf.Lerp(star.orbitScaler, 1f, 0.6f);
    StarGen.SetStarAge(star, star.age, rn, rt);
    star.dysonRadius = star.orbitScaler * 0.28f;
    if ((double) star.dysonRadius * 40000.0 < (double) star.physicsRadius * 1.5)
      star.dysonRadius = (float) ((double) star.physicsRadius * 1.5 / 40000.0);
    star.uPosition = star.position * 2400000.0;
    star.name = NameGen.RandomStarName(seed1, star, galaxy);
    star.overrideName = "";
    return star;
  }

  public static StarData CreateBirthStar(GalaxyData galaxy, int seed)
  {
    StarData birthStar = new StarData();
    birthStar.galaxy = galaxy;
    birthStar.index = 0;
    birthStar.level = 0.0f;
    birthStar.id = 1;
    birthStar.seed = seed;
    birthStar.resourceCoef = 0.6f;
    DotNet35Random dotNet35Random1 = new DotNet35Random(seed);
    int seed1 = dotNet35Random1.Next();
    int Seed = dotNet35Random1.Next();
    birthStar.name = NameGen.RandomName(seed1);
    birthStar.overrideName = "";
    birthStar.position = VectorLF3.zero;
    DotNet35Random dotNet35Random2 = new DotNet35Random(Seed);
    double r1 = dotNet35Random2.NextDouble();
    double r2 = dotNet35Random2.NextDouble();
    double num1 = dotNet35Random2.NextDouble();
    double rn = dotNet35Random2.NextDouble();
    double rt = dotNet35Random2.NextDouble();
    double num2 = dotNet35Random2.NextDouble() * 0.2 + 0.9;
    double num3 = Math.Pow(2.0, dotNet35Random2.NextDouble() * 0.4 - 0.2);
    float p1 = Mathf.Clamp(StarGen.RandNormal(0.0f, 0.08f, r1, r2), -0.2f, 0.2f);
    birthStar.mass = Mathf.Pow(2f, p1);
    if ((double) StarGen.specifyBirthStarMass > 0.100000001490116)
      birthStar.mass = StarGen.specifyBirthStarMass;
    if ((double) StarGen.specifyBirthStarAge > 9.99999974737875E-06)
      birthStar.age = StarGen.specifyBirthStarAge;
    double d = 2.0 + 0.4 * (1.0 - (double) birthStar.mass);
    birthStar.lifetime = (float) (10000.0 * Math.Pow(0.1, Math.Log10((double) birthStar.mass * 0.5) / Math.Log10(d) + 1.0) * num2);
    birthStar.age = (float) (num1 * 0.4 + 0.3);
    if ((double) StarGen.specifyBirthStarAge > 9.99999974737875E-06)
      birthStar.age = StarGen.specifyBirthStarAge;
    float num4 = (float) (1.0 - (double) Mathf.Pow(Mathf.Clamp01(birthStar.age), 20f) * 0.5) * birthStar.mass;
    birthStar.temperature = (float) (Math.Pow((double) num4, 0.56 + 0.14 / (Math.Log10((double) num4 + 4.0) / Math.Log10(5.0))) * 4450.0 + 1300.0);
    double num5 = Math.Log10(((double) birthStar.temperature - 1300.0) / 4500.0) / Math.Log10(2.6) - 0.5;
    if (num5 < 0.0)
      num5 *= 4.0;
    if (num5 > 2.0)
      num5 = 2.0;
    else if (num5 < -4.0)
      num5 = -4.0;
    birthStar.spectr = (ESpectrType) Mathf.RoundToInt((float) num5 + 4f);
    birthStar.color = Mathf.Clamp01((float) ((num5 + 3.5) * 0.200000002980232));
    birthStar.classFactor = (float) num5;
    birthStar.luminosity = Mathf.Pow(num4, 0.7f);
    birthStar.radius = (float) (Math.Pow((double) birthStar.mass, 0.4) * num3);
    birthStar.acdiskRadius = 0.0f;
    float p2 = (float) num5 + 2f;
    birthStar.habitableRadius = Mathf.Pow(1.7f, p2) + 0.2f * Mathf.Min(1f, birthStar.orbitScaler);
    birthStar.lightBalanceRadius = Mathf.Pow(1.7f, p2);
    birthStar.orbitScaler = Mathf.Pow(1.35f, p2);
    if ((double) birthStar.orbitScaler < 1.0)
      birthStar.orbitScaler = Mathf.Lerp(birthStar.orbitScaler, 1f, 0.6f);
    StarGen.SetStarAge(birthStar, birthStar.age, rn, rt);
    birthStar.dysonRadius = birthStar.orbitScaler * 0.28f;
    if ((double) birthStar.dysonRadius * 40000.0 < (double) birthStar.physicsRadius * 1.5)
      birthStar.dysonRadius = (float) ((double) birthStar.physicsRadius * 1.5 / 40000.0);
    birthStar.uPosition = VectorLF3.zero;
    birthStar.name = NameGen.RandomStarName(seed1, birthStar, galaxy);
    birthStar.overrideName = "";
    return birthStar;
  }

  private static double _signpow(double x, double pow)
  {
    double num = x > 0.0 ? 1.0 : -1.0;
    return Math.Abs(Math.Pow(x, pow)) * num;
  }

  public static void CreateStarPlanets(GalaxyData galaxy, StarData star, GameDesc gameDesc)
  {
    DotNet35Random dotNet35Random1 = new DotNet35Random(star.seed);
    dotNet35Random1.Next();
    dotNet35Random1.Next();
    dotNet35Random1.Next();
    DotNet35Random dotNet35Random2 = new DotNet35Random(dotNet35Random1.Next());
    double num1 = dotNet35Random2.NextDouble();
    double num2 = dotNet35Random2.NextDouble();
    double num3 = dotNet35Random2.NextDouble();
    double num4 = dotNet35Random2.NextDouble();
    double num5 = dotNet35Random2.NextDouble();
    double num6 = dotNet35Random2.NextDouble() * 0.2 + 0.9;
    double num7 = dotNet35Random2.NextDouble() * 0.2 + 0.9;
    if (star.type == EStarType.BlackHole)
    {
      star.planetCount = 1;
      star.planets = new PlanetData[star.planetCount];
      int info_seed = dotNet35Random2.Next();
      int gen_seed = dotNet35Random2.Next();
      star.planets[0] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 0, 0, 3, 1, false, info_seed, gen_seed);
    }
    else if (star.type == EStarType.NeutronStar)
    {
      star.planetCount = 1;
      star.planets = new PlanetData[star.planetCount];
      int info_seed = dotNet35Random2.Next();
      int gen_seed = dotNet35Random2.Next();
      star.planets[0] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 0, 0, 3, 1, false, info_seed, gen_seed);
    }
    else if (star.type == EStarType.WhiteDwarf)
    {
      if (num1 < 0.699999988079071)
      {
        star.planetCount = 1;
        star.planets = new PlanetData[star.planetCount];
        int info_seed = dotNet35Random2.Next();
        int gen_seed = dotNet35Random2.Next();
        star.planets[0] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 0, 0, 3, 1, false, info_seed, gen_seed);
      }
      else
      {
        star.planetCount = 2;
        star.planets = new PlanetData[star.planetCount];
        if (num2 < 0.300000011920929)
        {
          int info_seed1 = dotNet35Random2.Next();
          int gen_seed1 = dotNet35Random2.Next();
          star.planets[0] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 0, 0, 3, 1, false, info_seed1, gen_seed1);
          int info_seed2 = dotNet35Random2.Next();
          int gen_seed2 = dotNet35Random2.Next();
          star.planets[1] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 1, 0, 4, 2, false, info_seed2, gen_seed2);
        }
        else
        {
          int info_seed3 = dotNet35Random2.Next();
          int gen_seed3 = dotNet35Random2.Next();
          star.planets[0] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 0, 0, 4, 1, true, info_seed3, gen_seed3);
          int info_seed4 = dotNet35Random2.Next();
          int gen_seed4 = dotNet35Random2.Next();
          star.planets[1] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 1, 1, 1, 1, false, info_seed4, gen_seed4);
        }
      }
    }
    else if (star.type == EStarType.GiantStar)
    {
      if (num1 < 0.300000011920929)
      {
        star.planetCount = 1;
        star.planets = new PlanetData[star.planetCount];
        int info_seed = dotNet35Random2.Next();
        int gen_seed = dotNet35Random2.Next();
        star.planets[0] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 0, 0, num3 > 0.5 ? 3 : 2, 1, false, info_seed, gen_seed);
      }
      else if (num1 < 0.800000011920929)
      {
        star.planetCount = 2;
        star.planets = new PlanetData[star.planetCount];
        if (num2 < 0.25)
        {
          int info_seed5 = dotNet35Random2.Next();
          int gen_seed5 = dotNet35Random2.Next();
          star.planets[0] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 0, 0, num3 > 0.5 ? 3 : 2, 1, false, info_seed5, gen_seed5);
          int info_seed6 = dotNet35Random2.Next();
          int gen_seed6 = dotNet35Random2.Next();
          star.planets[1] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 1, 0, num3 > 0.5 ? 4 : 3, 2, false, info_seed6, gen_seed6);
        }
        else
        {
          int info_seed7 = dotNet35Random2.Next();
          int gen_seed7 = dotNet35Random2.Next();
          star.planets[0] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 0, 0, 3, 1, true, info_seed7, gen_seed7);
          int info_seed8 = dotNet35Random2.Next();
          int gen_seed8 = dotNet35Random2.Next();
          star.planets[1] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 1, 1, 1, 1, false, info_seed8, gen_seed8);
        }
      }
      else
      {
        star.planetCount = 3;
        star.planets = new PlanetData[star.planetCount];
        if (num2 < 0.150000005960464)
        {
          int info_seed9 = dotNet35Random2.Next();
          int gen_seed9 = dotNet35Random2.Next();
          star.planets[0] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 0, 0, num3 > 0.5 ? 3 : 2, 1, false, info_seed9, gen_seed9);
          int info_seed10 = dotNet35Random2.Next();
          int gen_seed10 = dotNet35Random2.Next();
          star.planets[1] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 1, 0, num3 > 0.5 ? 4 : 3, 2, false, info_seed10, gen_seed10);
          int info_seed11 = dotNet35Random2.Next();
          int gen_seed11 = dotNet35Random2.Next();
          star.planets[2] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 2, 0, num3 > 0.5 ? 5 : 4, 3, false, info_seed11, gen_seed11);
        }
        else if (num2 < 0.75)
        {
          int info_seed12 = dotNet35Random2.Next();
          int gen_seed12 = dotNet35Random2.Next();
          star.planets[0] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 0, 0, num3 > 0.5 ? 3 : 2, 1, false, info_seed12, gen_seed12);
          int info_seed13 = dotNet35Random2.Next();
          int gen_seed13 = dotNet35Random2.Next();
          star.planets[1] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 1, 0, 4, 2, true, info_seed13, gen_seed13);
          int info_seed14 = dotNet35Random2.Next();
          int gen_seed14 = dotNet35Random2.Next();
          star.planets[2] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 2, 2, 1, 1, false, info_seed14, gen_seed14);
        }
        else
        {
          int info_seed15 = dotNet35Random2.Next();
          int gen_seed15 = dotNet35Random2.Next();
          star.planets[0] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 0, 0, num3 > 0.5 ? 4 : 3, 1, true, info_seed15, gen_seed15);
          int info_seed16 = dotNet35Random2.Next();
          int gen_seed16 = dotNet35Random2.Next();
          star.planets[1] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 1, 1, 1, 1, false, info_seed16, gen_seed16);
          int info_seed17 = dotNet35Random2.Next();
          int gen_seed17 = dotNet35Random2.Next();
          star.planets[2] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, 2, 1, 2, 2, false, info_seed17, gen_seed17);
        }
      }
    }
    else
    {
      if (StarGen.pGas == null)
        StarGen.pGas = new double[10];
      Array.Clear((Array) StarGen.pGas, 0, StarGen.pGas.Length);
      if (star.index == 0)
      {
        star.planetCount = 4;
        StarGen.pGas[0] = 0.0;
        StarGen.pGas[1] = 0.0;
        StarGen.pGas[2] = 0.0;
      }
      else if (star.spectr == ESpectrType.M)
      {
        star.planetCount = num1 >= 0.1 ? (num1 >= 0.3 ? (num1 >= 0.8 ? 4 : 3) : 2) : 1;
        if (star.planetCount <= 3)
        {
          StarGen.pGas[0] = 0.2;
          StarGen.pGas[1] = 0.2;
        }
        else
        {
          StarGen.pGas[0] = 0.0;
          StarGen.pGas[1] = 0.2;
          StarGen.pGas[2] = 0.3;
        }
      }
      else if (star.spectr == ESpectrType.K)
      {
        star.planetCount = num1 >= 0.1 ? (num1 >= 0.2 ? (num1 >= 0.7 ? (num1 >= 0.95 ? 5 : 4) : 3) : 2) : 1;
        if (star.planetCount <= 3)
        {
          StarGen.pGas[0] = 0.18;
          StarGen.pGas[1] = 0.18;
        }
        else
        {
          StarGen.pGas[0] = 0.0;
          StarGen.pGas[1] = 0.18;
          StarGen.pGas[2] = 0.28;
          StarGen.pGas[3] = 0.28;
        }
      }
      else if (star.spectr == ESpectrType.G)
      {
        star.planetCount = num1 >= 0.4 ? (num1 >= 0.9 ? 5 : 4) : 3;
        if (star.planetCount <= 3)
        {
          StarGen.pGas[0] = 0.18;
          StarGen.pGas[1] = 0.18;
        }
        else
        {
          StarGen.pGas[0] = 0.0;
          StarGen.pGas[1] = 0.2;
          StarGen.pGas[2] = 0.3;
          StarGen.pGas[3] = 0.3;
        }
      }
      else if (star.spectr == ESpectrType.F)
      {
        star.planetCount = num1 >= 0.35 ? (num1 >= 0.8 ? 5 : 4) : 3;
        if (star.planetCount <= 3)
        {
          StarGen.pGas[0] = 0.2;
          StarGen.pGas[1] = 0.2;
        }
        else
        {
          StarGen.pGas[0] = 0.0;
          StarGen.pGas[1] = 0.22;
          StarGen.pGas[2] = 0.31;
          StarGen.pGas[3] = 0.31;
        }
      }
      else if (star.spectr == ESpectrType.A)
      {
        star.planetCount = num1 >= 0.3 ? (num1 >= 0.75 ? 5 : 4) : 3;
        if (star.planetCount <= 3)
        {
          StarGen.pGas[0] = 0.2;
          StarGen.pGas[1] = 0.2;
        }
        else
        {
          StarGen.pGas[0] = 0.1;
          StarGen.pGas[1] = 0.28;
          StarGen.pGas[2] = 0.3;
          StarGen.pGas[3] = 0.35;
        }
      }
      else if (star.spectr == ESpectrType.B)
      {
        star.planetCount = num1 >= 0.3 ? (num1 >= 0.75 ? 6 : 5) : 4;
        if (star.planetCount <= 3)
        {
          StarGen.pGas[0] = 0.2;
          StarGen.pGas[1] = 0.2;
        }
        else
        {
          StarGen.pGas[0] = 0.1;
          StarGen.pGas[1] = 0.22;
          StarGen.pGas[2] = 0.28;
          StarGen.pGas[3] = 0.35;
          StarGen.pGas[4] = 0.35;
        }
      }
      else if (star.spectr == ESpectrType.O)
      {
        star.planetCount = num1 >= 0.5 ? 6 : 5;
        StarGen.pGas[0] = 0.1;
        StarGen.pGas[1] = 0.2;
        StarGen.pGas[2] = 0.25;
        StarGen.pGas[3] = 0.3;
        StarGen.pGas[4] = 0.32;
        StarGen.pGas[5] = 0.35;
      }
      else
        star.planetCount = 1;
      star.planets = new PlanetData[star.planetCount];
      int num8 = 0;
      int num9 = 0;
      int orbitAround = 0;
      int num10 = 1;
      for (int index = 0; index < star.planetCount; ++index)
      {
        int info_seed = dotNet35Random2.Next();
        int gen_seed = dotNet35Random2.Next();
        double num11 = dotNet35Random2.NextDouble();
        double num12 = dotNet35Random2.NextDouble();
        bool gasGiant = false;
        if (orbitAround == 0)
        {
          ++num8;
          if (index < star.planetCount - 1 && num11 < StarGen.pGas[index])
          {
            gasGiant = true;
            if (num10 < 3)
              num10 = 3;
          }
          for (; star.index != 0 || num10 != 3; ++num10)
          {
            int num13 = star.planetCount - index;
            int num14 = 9 - num10;
            if (num14 > num13)
            {
              float a = (float) num13 / (float) num14;
              float num15 = num10 <= 3 ? Mathf.Lerp(a, 1f, 0.15f) + 0.01f : Mathf.Lerp(a, 1f, 0.45f) + 0.01f;
              if (dotNet35Random2.NextDouble() < (double) num15)
                goto label_62;
            }
            else
              goto label_62;
          }
          gasGiant = true;
        }
        else
        {
          ++num9;
          gasGiant = false;
        }
label_62:
        star.planets[index] = PlanetGen.CreatePlanet(galaxy, star, gameDesc.savedThemeIds, index, orbitAround, orbitAround == 0 ? num10 : num9, orbitAround == 0 ? num8 : num9, gasGiant, info_seed, gen_seed);
        ++num10;
        if (gasGiant)
        {
          orbitAround = num8;
          num9 = 0;
        }
        if (num9 >= 1 && num12 < 0.8)
        {
          orbitAround = 0;
          num9 = 0;
        }
      }
    }
    int num16 = 0;
    int num17 = 0;
    int index1 = 0;
    for (int index2 = 0; index2 < star.planetCount; ++index2)
    {
      if (star.planets[index2].type == EPlanetType.Gas)
      {
        num16 = star.planets[index2].orbitIndex;
        break;
      }
    }
    for (int index3 = 0; index3 < star.planetCount; ++index3)
    {
      if (star.planets[index3].orbitAround == 0)
        num17 = star.planets[index3].orbitIndex;
    }
    if (num16 > 0)
    {
      int num18 = num16 - 1;
      bool flag = true;
      for (int index4 = 0; index4 < star.planetCount; ++index4)
      {
        if (star.planets[index4].orbitAround == 0 && star.planets[index4].orbitIndex == num16 - 1)
        {
          flag = false;
          break;
        }
      }
      if (flag && num4 < 0.2 + (double) num18 * 0.2)
        index1 = num18;
    }
    int index5 = num5 >= 0.2 ? (num5 >= 0.4 ? (num5 >= 0.8 ? 0 : num17 + 1) : num17 + 2) : num17 + 3;
    if (index5 != 0 && index5 < 5)
      index5 = 5;
    star.asterBelt1OrbitIndex = (float) index1;
    star.asterBelt2OrbitIndex = (float) index5;
    if (index1 > 0)
      star.asterBelt1Radius = StarGen.orbitRadius[index1] * (float) num6 * star.orbitScaler;
    if (index5 <= 0)
      return;
    star.asterBelt2Radius = StarGen.orbitRadius[index5] * (float) num7 * star.orbitScaler;
  }

  public static void SetStarAge(StarData star, float age, double rn, double rt)
  {
    float num1 = (float) (rn * 0.1 + 0.95);
    float num2 = (float) (rt * 0.4 + 0.8);
    float num3 = (float) (rt * 9.0 + 1.0);
    star.age = age;
    if ((double) age >= 1.0)
    {
      if ((double) star.mass >= 18.0)
      {
        star.type = EStarType.BlackHole;
        star.spectr = ESpectrType.X;
        star.mass *= 2.5f * num2;
        star.radius *= 1f;
        star.acdiskRadius = star.radius * 5f;
        star.temperature = 0.0f;
        star.luminosity *= 1f / 1000f * num1;
        star.habitableRadius = 0.0f;
        star.lightBalanceRadius *= 0.4f * num1;
        star.color = 1f;
      }
      else if ((double) star.mass >= 7.0)
      {
        star.type = EStarType.NeutronStar;
        star.spectr = ESpectrType.X;
        star.mass *= 0.2f * num1;
        star.radius *= 0.15f;
        star.acdiskRadius = star.radius * 9f;
        star.temperature = num3 * 1E+07f;
        star.luminosity *= 0.1f * num1;
        star.habitableRadius = 0.0f;
        star.lightBalanceRadius *= 3f * num1;
        star.orbitScaler *= 1.5f * num1;
        star.color = 1f;
      }
      else
      {
        star.type = EStarType.WhiteDwarf;
        star.spectr = ESpectrType.X;
        star.mass *= 0.2f * num1;
        star.radius *= 0.2f;
        star.acdiskRadius = 0.0f;
        star.temperature = num2 * 150000f;
        star.luminosity *= 0.04f * num2;
        star.habitableRadius *= 0.15f * num2;
        star.lightBalanceRadius *= 0.2f * num1;
        star.color = 0.7f;
      }
    }
    else
    {
      if ((double) age < 0.959999978542328)
        return;
      float num4 = (float) (Math.Pow(5.0, Math.Abs(Math.Log10((double) star.mass) - 0.7)) * 5.0);
      if ((double) num4 > 10.0)
        num4 = (float) (((double) Mathf.Log(num4 * 0.1f) + 1.0) * 10.0);
      float num5 = (float) (1.0 - (double) Mathf.Pow(star.age, 30f) * 0.5);
      star.type = EStarType.GiantStar;
      star.mass = num5 * star.mass;
      star.radius = num4 * num2;
      star.acdiskRadius = 0.0f;
      star.temperature = num5 * star.temperature;
      star.luminosity = 1.6f * star.luminosity;
      star.habitableRadius = 9f * star.habitableRadius;
      star.lightBalanceRadius = 3f * star.habitableRadius;
      star.orbitScaler = 3.3f * star.orbitScaler;
    }
  }

  private static float RandNormal(
    float averageValue,
    float standardDeviation,
    double r1,
    double r2)
  {
    return averageValue + standardDeviation * (float) (Math.Sqrt(-2.0 * Math.Log(1.0 - r1)) * Math.Sin(2.0 * Math.PI * r2));
  }
}
}
