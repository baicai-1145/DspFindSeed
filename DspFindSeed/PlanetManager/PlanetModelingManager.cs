using System;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;
using UnityEngine.Rendering;

namespace DspFindSeed
{


  public static class PlanetModelingManager
  {
  
    public static  int[]                                veinProducts;
    public static  int[]                                veinModelIndexs;
    public static  int[]                                veinModelCounts;
    public static  VeinProto[]                          veinProtos;

    // 只用于“矿点堆数统计”（不生成具体矿点位置/矿脉组），避免每颗行星重复分配数组导致 GC
    [ThreadStatic] private static int[] _veinSpotCountsBuffer;
    // 实验用途：把随机数比较也“强制 FP32”（将 NextDouble 截断到 float 后再比较）
    [ThreadStatic] private static int[] _veinSpotCountsBufferF32;
    
    private static Vector3[] verts200;
    private static Vector3[] verts80;
    private static int[]     indexMap200;
    private static int[]     indexMap80;
    public static Vector3[] poles = new Vector3[6]
    {
      Vector3.right,
      Vector3.left,
      Vector3.up,
      Vector3.down,
      Vector3.forward,
      Vector3.back
    };

   public static void Start ()
    {
      PrepareWorks ();
    }

    private static void PrepareWorks ()
    {
      int         length     = 0;
      VeinProto[] dataArray2 = LDB.veins.dataArray;
      for (int index = 0; index < dataArray2.Length; ++index)
        length = dataArray2[index].ID + 1;
      veinProducts    = new int[length];
      veinModelIndexs = new int[length];
      veinModelCounts = new int[length];
      veinProtos      = new VeinProto[length];
      for (int index = 0; index < dataArray2.Length; ++index)
      {
        veinProducts[dataArray2[index].ID]    = dataArray2[index].MiningItem;
        veinModelIndexs[dataArray2[index].ID] = dataArray2[index].ModelIndex;
        veinModelCounts[dataArray2[index].ID] = dataArray2[index].ModelCount;
        veinProtos[dataArray2[index].ID]      = dataArray2[index];
      }
    }

    public static PlanetAlgorithm Algorithm (PlanetData planet)
    {
      PlanetAlgorithm planetAlgorithm;
      switch (planet.algoId)
      {
        case 1:
          planetAlgorithm = (PlanetAlgorithm) new PlanetAlgorithm1();
          break;
        case 2:
          planetAlgorithm = (PlanetAlgorithm) new PlanetAlgorithm2();
          break;
        case 3:
          planetAlgorithm = (PlanetAlgorithm) new PlanetAlgorithm3();
          break;
        case 4:
          planetAlgorithm = (PlanetAlgorithm) new PlanetAlgorithm4();
          break;
        case 5:
          planetAlgorithm = (PlanetAlgorithm) new PlanetAlgorithm5();
          break;
        case 6:
          planetAlgorithm = (PlanetAlgorithm) new PlanetAlgorithm6();
          break;
        case 7:
          planetAlgorithm = (PlanetAlgorithm) new PlanetAlgorithm7();
          break;
        case 8:
          planetAlgorithm = (PlanetAlgorithm) new PlanetAlgorithm8();
          break;
        case 9:
          planetAlgorithm = (PlanetAlgorithm) new PlanetAlgorithm9();
          break;
        case 10:
          planetAlgorithm = (PlanetAlgorithm) new PlanetAlgorithm10();
          break;
        case 11:
          planetAlgorithm = (PlanetAlgorithm) new PlanetAlgorithm11();
          break;
        case 12:
          planetAlgorithm = (PlanetAlgorithm) new PlanetAlgorithm12();
          break;
        case 13:
          planetAlgorithm = (PlanetAlgorithm) new PlanetAlgorithm13();
          break;
        case 14:
          planetAlgorithm = (PlanetAlgorithm) new PlanetAlgorithm14();
          break;
        default:
          planetAlgorithm = (PlanetAlgorithm) new PlanetAlgorithm0();
          break;
      }
      planetAlgorithm?.Reset (planet.seed, planet);
      return planetAlgorithm;
    }

    public static int[] RefreshPlanetData (PlanetData planetData)
    {
      // 这里用于“种子筛选”的最小信息：只计算每种矿在该行星上的“矿点堆数（vein spot counts）”
      // 不生成具体矿点位置/矿脉组分布，从而避免地形/QueryHeight 等重逻辑。
      if (planetData == null)
        return null;
      if (planetData.type == EPlanetType.Gas)
        return null;

      ThemeProto themeProto = LDB.themes.Select(planetData.theme);
      if (themeProto == null)
        return null;

      // 缓冲区：长度按 veinProtos（由 PrepareWorks() 初始化）对齐
      int len = veinProtos != null ? veinProtos.Length : 0;
      if (len <= 0)
        return null;

      int[] counts = _veinSpotCountsBuffer;
      if (counts == null || counts.Length != len)
      {
        counts = new int[len];
        _veinSpotCountsBuffer = counts;
      }
      else
      {
        Array.Clear(counts, 0, counts.Length);
      }

      // 基础矿堆数来自主题表（索引从 1 开始，0 号不用）
      if (themeProto.VeinSpot != null)
        Array.Copy((Array)themeProto.VeinSpot, 0, (Array)counts, 1, Math.Min(themeProto.VeinSpot.Length, counts.Length - 1));

      // RNG 消费顺序对 determinism 至关重要：复制 PlanetAlgorithm.MyGenerateVeins 的“前半段”
      DotNet35Random rng = new DotNet35Random(planetData.seed);
      rng.Next();
      rng.Next();
      rng.Next();
      rng.Next();
      rng.Next(); // _birthSeed（这里只为保持消费顺序）
      _ = new DotNet35Random(rng.Next()); // dotNet35Random2（这里只为保持消费顺序）

      // 星系类型对稀有矿概率的 p 修正（float/Mathf，等价于原版路径）
      float p = 1f;
      ESpectrType spectr = planetData.star.spectr;
      switch (planetData.star.type)
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
          ++counts[9];
          ++counts[9];
          for (int i = 1; i < 12 && rng.NextDouble() < 0.449999988079071; ++i)
            ++counts[9];
          ++counts[10];
          ++counts[10];
          for (int i = 1; i < 12 && rng.NextDouble() < 0.449999988079071; ++i)
            ++counts[10];
          ++counts[12];
          for (int i = 1; i < 12 && rng.NextDouble() < 0.5; ++i)
            ++counts[12];
          break;
        case EStarType.NeutronStar:
          p = 4.5f;
          ++counts[14];
          for (int i = 1; i < 12 && rng.NextDouble() < 0.649999976158142; ++i)
            ++counts[14];
          break;
        case EStarType.BlackHole:
          p = 5f;
          ++counts[14];
          for (int i = 1; i < 12 && rng.NextDouble() < 0.649999976158142; ++i)
            ++counts[14];
          break;
      }

      // 主题稀有矿：只需要更新 counts（不需要 count/opacity，也不需要落点）
      if (themeProto.RareVeins != null && themeProto.RareSettings != null)
      {
        for (int idx = 0; idx < themeProto.RareVeins.Length; ++idx)
        {
          int rareVein = themeProto.RareVeins[idx];
          float appearBase = planetData.star.index == 0 ? themeProto.RareSettings[idx * 4] : themeProto.RareSettings[idx * 4 + 1];
          float chainProb = themeProto.RareSettings[idx * 4 + 2];

          float appearProb = 1f - Mathf.Pow(1f - appearBase, p);
          // 注意：这里的 NextDouble 比较如果强制改成“float 随机数”，会改变极少量边界情况，从而可能漏种子
          if (rng.NextDouble() < (double)appearProb)
          {
            ++counts[rareVein];
            for (int i = 1; i < 12 && rng.NextDouble() < (double)chainProb; ++i)
              ++counts[rareVein];
          }
        }
      }

      return counts;
    }

    /// <summary>
    /// 实验用途：把“概率分支的随机数比较”也强制 FP32（将 NextDouble() 截断成 float 再比较）。
    /// 为避免边界值因截断触发分支翻转，增加 tie-break：
    /// 当 (float)rv == threshold 时，回退使用 rv(double) 与 threshold(double) 比较。
    /// 这样可保持大部分 F32 行为，同时消除边界漏种子。
    /// </summary>
    public static int[] RefreshPlanetData_F32 (PlanetData planetData)
    {
      if (planetData == null)
        return null;
      if (planetData.type == EPlanetType.Gas)
        return null;

      ThemeProto themeProto = LDB.themes.Select(planetData.theme);
      if (themeProto == null)
        return null;

      int len = veinProtos != null ? veinProtos.Length : 0;
      if (len <= 0)
        return null;

      int[] counts = _veinSpotCountsBufferF32;
      if (counts == null || counts.Length != len)
      {
        counts = new int[len];
        _veinSpotCountsBufferF32 = counts;
      }
      else
      {
        Array.Clear(counts, 0, counts.Length);
      }

      if (themeProto.VeinSpot != null)
        Array.Copy((Array)themeProto.VeinSpot, 0, (Array)counts, 1, Math.Min(themeProto.VeinSpot.Length, counts.Length - 1));

      DotNet35Random rng = new DotNet35Random(planetData.seed);
      rng.Next();
      rng.Next();
      rng.Next();
      rng.Next();
      rng.Next();
      _ = new DotNet35Random(rng.Next());

      float p = 1f;
      ESpectrType spectr = planetData.star.spectr;
      switch (planetData.star.type)
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
          ++counts[9];
          ++counts[9];
          for (int i = 1; i < 12 && ProbHitF32WithTieBreak(rng, 0.449999988079071f); ++i)
            ++counts[9];
          ++counts[10];
          ++counts[10];
          for (int i = 1; i < 12 && ProbHitF32WithTieBreak(rng, 0.449999988079071f); ++i)
            ++counts[10];
          ++counts[12];
          for (int i = 1; i < 12 && ProbHitF32WithTieBreak(rng, 0.5f); ++i)
            ++counts[12];
          break;
        case EStarType.NeutronStar:
          p = 4.5f;
          ++counts[14];
          for (int i = 1; i < 12 && ProbHitF32WithTieBreak(rng, 0.649999976158142f); ++i)
            ++counts[14];
          break;
        case EStarType.BlackHole:
          p = 5f;
          ++counts[14];
          for (int i = 1; i < 12 && ProbHitF32WithTieBreak(rng, 0.649999976158142f); ++i)
            ++counts[14];
          break;
      }

      if (themeProto.RareVeins != null && themeProto.RareSettings != null)
      {
        for (int idx = 0; idx < themeProto.RareVeins.Length; ++idx)
        {
          int rareVein = themeProto.RareVeins[idx];
          float appearBase = planetData.star.index == 0 ? themeProto.RareSettings[idx * 4] : themeProto.RareSettings[idx * 4 + 1];
          float chainProb = themeProto.RareSettings[idx * 4 + 2];

          float appearProb = 1f - Mathf.Pow(1f - appearBase, p);
          if (ProbHitF32WithTieBreak(rng, appearProb))
          {
            ++counts[rareVein];
            for (int i = 1; i < 12 && ProbHitF32WithTieBreak(rng, chainProb); ++i)
              ++counts[rareVein];
          }
        }
      }

      return counts;
    }

    private static bool ProbHitF32WithTieBreak(DotNet35Random rng, float threshold)
    {
      double rv = rng.NextDouble();
      float rvf = (float)rv;
      return rvf < threshold || (rvf == threshold && rv < (double)threshold);
    }

    public static void CalcVerts (PlanetRawData planetRawData)
    {
      if (planetRawData.precision == 200 && verts200 != null)
      {
        Array.Copy ((Array) verts200, (Array) planetRawData.vertices, verts200.Length);
        Array.Copy ((Array) indexMap200, (Array) planetRawData.indexMap, indexMap200.Length);
      }
      else if (planetRawData.precision == 80 && verts80 != null)
      {
        Array.Copy ((Array) verts80, (Array) planetRawData.vertices, verts80.Length);
        Array.Copy ((Array) indexMap80, (Array) planetRawData.indexMap, indexMap80.Length);
      }
      else
      {
        for (int index = 0; index < planetRawData.indexMapDataLength; ++index)
          planetRawData.indexMap[index] = -1;
        int num1 = (planetRawData.precision + 1) * 2;
        int num2 = planetRawData.precision + 1;
        for (int index1 = 0; index1 < planetRawData.dataLength; ++index1)
        {
          int     num3  = index1 % num1;
          int     num4  = index1 / num1;
          int     num5  = num3 % num2;
          int     num6  = num4 % num2;
          int     num7  = ((num3 >= num2 ? 1 : 0) + (num4 >= num2 ? 1 : 0) * 2) * 2 + (num5 >= num6 ? 0 : 1);
          float   num8  = num5 >= num6 ? (float) (planetRawData.precision - num5) : (float) num5;
          float   num9  = num5 >= num6 ? (float) num6 : (float) (planetRawData.precision - num6);
          float   num10 = (float) planetRawData.precision - num9;
          float   t1    = num9 / (float) planetRawData.precision;
          float   t2    = (double) num10 > 0.0 ? num8 / num10 : 0.0f;
          Vector3 pole1;
          Vector3 pole2;
          Vector3 pole3;
          int     corner;
          switch (num7)
          {
            case 0:
              pole1  = PlanetRawData.poles[2];
              pole2  = PlanetRawData.poles[0];
              pole3  = PlanetRawData.poles[4];
              corner = 7;
              break;
            case 1:
              pole1  = PlanetRawData.poles[3];
              pole2  = PlanetRawData.poles[4];
              pole3  = PlanetRawData.poles[0];
              corner = 5;
              break;
            case 2:
              pole1  = PlanetRawData.poles[2];
              pole2  = PlanetRawData.poles[4];
              pole3  = PlanetRawData.poles[1];
              corner = 6;
              break;
            case 3:
              pole1  = PlanetRawData.poles[3];
              pole2  = PlanetRawData.poles[1];
              pole3  = PlanetRawData.poles[4];
              corner = 4;
              break;
            case 4:
              pole1  = PlanetRawData.poles[2];
              pole2  = PlanetRawData.poles[1];
              pole3  = PlanetRawData.poles[5];
              corner = 2;
              break;
            case 5:
              pole1  = PlanetRawData.poles[3];
              pole2  = PlanetRawData.poles[5];
              pole3  = PlanetRawData.poles[1];
              corner = 0;
              break;
            case 6:
              pole1  = PlanetRawData.poles[2];
              pole2  = PlanetRawData.poles[5];
              pole3  = PlanetRawData.poles[0];
              corner = 3;
              break;
            case 7:
              pole1  = PlanetRawData.poles[3];
              pole2  = PlanetRawData.poles[0];
              pole3  = PlanetRawData.poles[5];
              corner = 1;
              break;
            default:
              pole1  = PlanetRawData.poles[2];
              pole2  = PlanetRawData.poles[0];
              pole3  = PlanetRawData.poles[4];
              corner = 7;
              break;
          }
          planetRawData.vertices[index1] = Slerp (Slerp (pole1, pole3, t1), Slerp (pole2, pole3, t1), t2);
          int index2 = planetRawData.PositionHash (planetRawData.vertices[index1], corner);
          if (planetRawData.indexMap[index2] == -1)
            planetRawData.indexMap[index2] = index1;
        }
        int num11 = 0;
        for (int index = 1; index < planetRawData.indexMapDataLength; ++index)
        {
          if (planetRawData.indexMap[index] == -1)
          {
            planetRawData.indexMap[index] = planetRawData.indexMap[index - 1];
            ++num11;
          }
        }
        if (planetRawData.precision == 200)
        {
          if (verts200 != null)
            return;
          verts200    = new Vector3[planetRawData.vertices.Length];
          indexMap200 = new int[planetRawData.indexMap.Length];
          Array.Copy ((Array) planetRawData.vertices, (Array) verts200, planetRawData.vertices.Length);
          Array.Copy ((Array) planetRawData.indexMap, (Array) indexMap200, planetRawData.indexMap.Length);
        }
        else
        {
          if (planetRawData.precision != 80 || verts80 != null)
            return;
          verts80    = new Vector3[planetRawData.vertices.Length];
          indexMap80 = new int[planetRawData.indexMap.Length];
          Array.Copy ((Array) planetRawData.vertices, (Array) verts80, planetRawData.vertices.Length);
          Array.Copy ((Array) planetRawData.indexMap, (Array) indexMap80, planetRawData.indexMap.Length);
        }
      }
    }
    
    public static Vector3 Slerp(Vector3 a, Vector3 b, float t)
    {
      if (t <= 0)
      {
        return a;
      }
      else if (t >= 1)
      {
        return b;
      }

      Vector3 v = RotateTo(a, b, Vector3.Angle(a, b) * t);

      //向量的长度，跟线性插值一样计算
      float length = b.magnitude * t + a.magnitude * (1 - t);
      return v.normalized * length;
    }
    
    public static Vector3 RotateTo(Vector3 from, Vector3 to, float angle)
    {
      //如果两向量角度为0
      if (Vector3.Angle(from, to) == 0)
      {
        return from;
      }

      //旋转轴
      Vector3 n = Vector3.Cross(from, to);

      //旋转轴规范化
      n.Normalize();

      //旋转矩阵
      Matrix4x4 rotateMatrix = new Matrix4x4();

      //旋转的弧度
      double radian   = angle * Math.PI / 180;
      float  cosAngle = (float)Math.Cos(radian);
      float  sinAngle = (float)Math.Sin(radian);

      //矩阵的数据
      //这里看不懂的自行科普矩阵知识
      rotateMatrix.SetRow(0, new Vector4(n.x * n.x * (1 - cosAngle) + cosAngle, n.x * n.y * (1 - cosAngle) + n.z * sinAngle, n.x * n.z * (1 - cosAngle) - n.y * sinAngle, 0));
      rotateMatrix.SetRow(1, new Vector4(n.x * n.y * (1 - cosAngle) - n.z * sinAngle, n.y * n.y * (1 - cosAngle) + cosAngle, n.y * n.z * (1 - cosAngle) + n.x * sinAngle, 0));
      rotateMatrix.SetRow(2, new Vector4(n.x * n.z * (1 - cosAngle) + n.y * sinAngle, n.y * n.z * (1 - cosAngle) - n.x * sinAngle, n.z * n.z * (1 - cosAngle) + cosAngle, 0));
      rotateMatrix.SetRow(3, new Vector4(0, 0, 0, 1));

      Vector4  v      = ToVector4(from);
      Vector3 vector = new Vector3();
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; j++)
        {
          vector[i] += v[j] * rotateMatrix[j, i];
        }
      }
      return vector;
    }
    
    public static Vector4 ToVector4(Vector3 v)
    {
      return new Vector4(v.x, v.y, v.z, 0);
    }
    


   





    private enum ThreadFlag
    {
      Ended,
      Running,
      Ending,
    }

    private class ThreadFlagLock
    {
      private int obj;
    }
  }
}
