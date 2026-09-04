
namespace MLs
{
    public class LinearRegression : Regression
    {
        public double[] Weights { get => _weights; set => _weights = value; }
        public LinearRegression(double[] weights, double bias)
        {
            _weights = weights;
            _bias = bias;
        }

        public void Fit(double[][] X, double[] Y, double learningRate, int epochs = 1000)
        {
            for(int epoch = 1; epoch <= epochs; epoch++)
            {
                double[] predicts = Predict(X);
                double[] gradY = new double[_weights.Length]; double gradB = 0;
                for (int i = 0; i < X.Length; i++)
                {
                    double error = predicts[i] - Y[i];
                    for (int j = 0; j < _weights.Length; j++) gradY[j] += error * X[i][j];
                    gradB += error;
                }
                double factor = 2.0 / X.Length;
                for (int j = 0; j < _weights.Length; j++)
                {
                    gradY[j] *= factor;
                    _weights[j] -= learningRate * gradY[j];
                }
                gradB *= factor;
                _bias -= learningRate * gradB;
            }
        }
    }
}
