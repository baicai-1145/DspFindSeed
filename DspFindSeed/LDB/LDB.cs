using System;
using System.IO;
using System.Text;
using System.Xml.Serialization;
using Newtonsoft.Json;

namespace DspFindSeed
{
    public static class LDB
    {
        // 兼容多种资源目录布局：
        // - Prototypes/*.xml / Prototypes/*.json（期望路径）
        // - prototypes/*.xml（历史 release 打包中常见的小写目录）
        // - Release/Prototypes/*.json（仓库根目录 Release 导出的最新版 json）
        private static readonly string[] ProtoDirCandidates =
        {
            "Prototypes",
            "prototypes",
            Path.Combine("Release", "Prototypes"),
            Path.Combine("release", "Prototypes"),
        };

        private static string protoResDir = "Prototypes/";
        public static  ThemeProtoSet _themes;
        public static  VeinProtoSet  _veins;
        public static  ItemProtoSet  _items;

        private static void PostLoad(ProtoTable table)
        {
            if (table is VeinProtoSet veinProtoSet)
                veinProtoSet.OnAfterDeserialize();
            if (table is ItemProtoSet itemProtoSet)
                itemProtoSet.OnAfterDeserialize();
            if (table is ThemeProtoSet themeProtoSet)
                themeProtoSet.OnAfterDeserialize();
        }

        private static string ResolveProtoFilePath(string typeName, string extensionWithDot)
        {
            // 0) 允许从当前目录向上搜索，避免“运行目录不同导致找不到 Release/Prototypes”。
            // 最多向上 6 层，基本覆盖：
            // - repo 根目录运行
            // - bin/Release 运行
            // - IDE 启动目录差异
            string TryResolveIn(string baseDir)
            {
                if (string.IsNullOrWhiteSpace(baseDir))
                    return null;

                // a) 先用当前配置的 protoResDir（相对 baseDir）
                if (!string.IsNullOrWhiteSpace(protoResDir))
                {
                    var p = Path.Combine(baseDir, protoResDir, typeName + extensionWithDot);
                    if (File.Exists(p))
                        return p;
                }

                // b) 再扫候选目录（相对 baseDir）
                foreach (var dir in ProtoDirCandidates)
                {
                    var p = Path.Combine(baseDir, dir, typeName + extensionWithDot);
                    if (File.Exists(p))
                        return p;
                }

                return null;
            }

            var cwd = Directory.GetCurrentDirectory();
            var cur = new DirectoryInfo(cwd);
            for (int i = 0; i < 6 && cur != null; i++, cur = cur.Parent)
            {
                var found = TryResolveIn(cur.FullName);
                if (!string.IsNullOrEmpty(found))
                    return found;
            }

            // 1) 先用当前配置的 protoResDir（如果用户未来要手动指定目录）
            if (!string.IsNullOrWhiteSpace(protoResDir))
            {
                var p = Path.Combine(protoResDir, typeName + extensionWithDot);
                if (File.Exists(p))
                    return p;
            }

            // 2) 再扫常见候选目录
            foreach (var dir in ProtoDirCandidates)
            {
                var p = Path.Combine(dir, typeName + extensionWithDot);
                if (File.Exists(p))
                    return p;
            }

            return null;
        }

        private static T LoadFromJson<T>(string jsonPath) where T : ProtoTable
        {
            // json 由外部导出工具生成，通常是 UTF-8
            var text = File.ReadAllText(jsonPath, Encoding.UTF8);
            var obj  = JsonConvert.DeserializeObject<T>(text);
            if (obj == null)
                throw new InvalidOperationException($"反序列化失败: {jsonPath}");
            PostLoad(obj);
            return obj;
        }

        private static T LoadFromXml<T>(string xmlPath) where T : ProtoTable
        {
            var xmlSerializer = new XmlSerializer(typeof (T));
            using (var fs = File.OpenRead(xmlPath))
            {
                var obj = xmlSerializer.Deserialize(fs) as T;
                if (obj == null)
                    throw new InvalidOperationException($"反序列化失败: {xmlPath}");
                PostLoad(obj);
                return obj;
            }
        }

        private static T LoadTable<T>(ref T tmp) where T : ProtoTable
        {
            if ((object) tmp != null)
                return tmp;

            var typeName = typeof(T).Name;

            // 优先尝试 json（对应仓库根目录 Release/Prototypes/*.json 的最新版导出）
            var jsonPath = ResolveProtoFilePath(typeName, ".json");
            if (!string.IsNullOrEmpty(jsonPath))
            {
                tmp = LoadFromJson<T>(jsonPath);
                return tmp;
            }

            // 回退 xml（兼容旧版打包）
            var xmlPath = ResolveProtoFilePath(typeName, ".xml");
            if (!string.IsNullOrEmpty(xmlPath))
            {
                tmp = LoadFromXml<T>(xmlPath);
                return tmp;
            }

            throw new FileNotFoundException($"未找到原型表: {typeName}.json 或 {typeName}.xml。请放到 Prototypes/ 或 Release/Prototypes/ 目录下。");
        }

        public static ThemeProtoSet themes => LDB.LoadTable<ThemeProtoSet>(ref LDB._themes);

        public static VeinProtoSet veins => LDB.LoadTable<VeinProtoSet>(ref LDB._veins);

        public static ItemProtoSet items => LDB.LoadTable<ItemProtoSet>(ref LDB._items);
    }
    
    public abstract class ProtoTable
    {
        public string TableName;

        public abstract void Init(int length);

        public abstract Proto this[int index] { get; set; }

        public abstract int Length { get; }
    }
}