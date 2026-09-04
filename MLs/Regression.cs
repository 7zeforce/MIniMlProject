
namespace MLs
{
    public class Regression
    {
        protected double[] _weights;
        protected double _bias;

        public virtual void Fit() { }

        public virtual double[] Predict(double[][] X)
        {
            double[] predictions = new double[X.Length];
            for(int i = 0; i < X.Length; i++)
            {
                for (int j = 0; j < _weights.Length; j++) predictions[i] += X[i][j] * _weights[j];
                predictions[i] += _bias;
            }
            return predictions;
        }

        public virtual (double[][] Normalized, double[] Mean, double[] Std) Standardize(double[][] X)
        {
            int n = X.Length;
            int m = X[0].Length;
            double[] means = new double[m];
            double[] stds = new double[m];
            double[][] X_norm = new double[n][];
            for (int j = 0; j < m; j++)
            {
                double[] col = new double[n];
                for (int i = 0; i < n; i++)
                    col[i] = X[i][j];
                double mean = col.Average();
                double std = Math.Sqrt(col.Sum(v => (v - mean) * (v - mean)) / n);
                means[j] = mean;
                stds[j] = std == 0 ? 1.0 : std;
                for (int i = 0; i < n; i++)
                {
                    if (X_norm[i] == null) X_norm[i] = new double[m];
                    X_norm[i][j] = (col[i] - mean) / stds[j];
                }
            }

            return (X_norm, means, stds);
        }
    }
}
