
namespace MLs.Components
{
    public record ConfusionMatrix(int TruePositive, int FalsePositive,
                          int TrueNegative, int FalseNegative)
    {
        public double Accuracy => (double)(TruePositive + TrueNegative) / (double)(TruePositive + TrueNegative + FalseNegative + FalsePositive);
        public double Precision => TruePositive / (TruePositive + FalsePositive);
        public double Recall => TruePositive / (TruePositive + FalseNegative);
        public double F1 => (Precision + Recall) == 0 ? 0 : 2 * Precision * Recall / (Precision + Recall);
        
        static public ConfusionMatrix FormConsusionMatrix(double[] y, int[] pred)
        {
            int tp = 0, fp = 0, tn = 0, fn = 0;
            for(int i = 0; i < y.Length; i++)
            {
                if ((int)y[i] == 1 && pred[i] == 1) tp++;
                else if ((int)y[i] == 0 && pred[i] == 1) fp++;
                else if ((int)y[i] == 1 && pred[i] == 0) fn++;
                else if ((int)y[i] == 0 && pred[i] == 0) tn++;
            }
            return new ConfusionMatrix(tp, fp, tn, fp);
        }
    }
}
