
namespace MLs
{
    public class LogicalRegression : Regression
    {
        public double[] Weights { get => _weights; set => _weights = value; }
        public LogicalRegression(double[] weight, double bias)
        {
            _weights = weight;
            _bias = bias;
        }

        public void Fit(double[][] X, double[] y, double learningRate, int epochs = 1000)
        {
            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                double[] predict = Predict(X).Select(p => Sigmoid(p)).ToArray();
                double[] gradW = new double[_weights.Length];double gradB = 0;
                for(int i = 0; i < X.Length; i++)
                {
                    double error = predict[i] - y[i];
                    gradB += error;
                    for(int j = 0; j < _weights.Length; j++) gradW[j] += error * X[i][j];
                }
                gradB *= (1.0 / X.Length);
                _bias -= gradB * learningRate;
                double loss = 0;
                for (int i = 0; i < X.Length; i++)
                {
                    double p = predict[i];
                    loss += -y[i] * Math.Log(p + 1e-15) - (1 - y[i]) * Math.Log(1 - p + 1e-15);
                }
                loss /= X.Length;
                //Console.WriteLine($"Epoch {epoch}: loss = {loss:F6}");
                for (int j = 0; j < _weights.Length; j++) _weights[j] -= (gradW[j]/X.Length) * learningRate;
            }
        }

        public double Sigmoid(double z) => 1.0 / (1.0 + Math.Exp(-z));
    }
}
