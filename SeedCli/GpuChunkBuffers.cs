namespace SeedCli
{
    internal sealed class GpuChunkBuffers
    {
        public int[] SeedBuffer;
        public GalaxyData[] GalaxyBuffer;

        public void EnsureCapacity(int chunkSize)
        {
            if (chunkSize < 1)
                chunkSize = 1;
            if (SeedBuffer == null || SeedBuffer.Length < chunkSize)
                SeedBuffer = new int[chunkSize];
            if (GalaxyBuffer == null || GalaxyBuffer.Length < chunkSize)
                GalaxyBuffer = new GalaxyData[chunkSize];
        }
    }
}
