namespace SeedCli
{
    internal static class MixRuntimeFlags
    {
        // 仅用于 compare-pipeline-mix* 路径：
        // 跳过与签名无关的字段构造，减少 CPU 对象流负担。
        public static bool SignatureOnlyFastPath;

        // 仅用于 compare-pipeline-mix* 的单线程 chunk 路径：
        // 在一个 chunk 内复用同一个 core batch 作用域，减少每 seed Begin/Flush。
        public static bool ChunkWideCoreBatch;

        // ChunkWideCoreBatch 打开时，每多少个 seed 刷新一次批处理作用域。
        // 过大时会导致上下文集合过重，过小则 GPU 批处理收益下降。
        public static int ChunkWideCoreBatchSeedGroup = 64;
    }
}
