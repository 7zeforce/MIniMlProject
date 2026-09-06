using MLs.Components;

namespace MLs
{
    public class ParseData
    {
        public DigitImage[] ReadLabels(string labelsPath, string imagesPath)
        {
            List<byte> labels = new List<byte>();
            DigitImage[] di;
            using (FileStream fsImages = new FileStream(imagesPath, FileMode.Open))
            using (BinaryReader brImages = new BinaryReader(fsImages))
            using (FileStream fsLabels = new FileStream(labelsPath, FileMode.Open))
            using (BinaryReader brLabels = new BinaryReader(fsLabels))
            {
                int magicImages = ReadBigEndianInt32(brImages);
                int numImages = ReadBigEndianInt32(brImages);
                int numRows = ReadBigEndianInt32(brImages);
                int numCols = ReadBigEndianInt32(brImages);
                int magicLabels = ReadBigEndianInt32(brLabels);
                int numLabels = ReadBigEndianInt32(brLabels);
                di = new DigitImage[numImages];
                for (int i = 0; i < numImages; i++)
                {
                    di[i] = new DigitImage { Label = brLabels.ReadByte(), Pixels = brImages.ReadBytes(numRows * numCols) };
                }
            }
            return di;
        }

        private int ReadBigEndianInt32(BinaryReader br) 
        {
            byte[]? bytes = br.ReadBytes(4);
            if (BitConverter.IsLittleEndian)
                Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }
    }
}
