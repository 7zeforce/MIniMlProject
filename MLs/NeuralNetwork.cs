
namespace MLs
{
    public class NeuralNetwork
    {
        public Random rand = new Random(42);

        protected double[][] weights1;
        protected double[][] weights2;
        protected double[] bias1;
        protected double[] bias2;

        protected int _inputSize;
        protected int _hidenSize;
        protected int _outputSize;

        public NeuralNetwork(int inputSize, int hidenSize, int outputSize)
        {
            _inputSize = inputSize; _hidenSize = hidenSize; _outputSize = outputSize;
            weights1 = InitializeWeights(hidenSize, inputSize);
            bias1 = new double[hidenSize];
            weights2 = InitializeWeights(outputSize, hidenSize);
            bias2 = new double[outputSize];
        }

        public virtual void Fit(double[][] yOneHot, double[][] X, double learningRate, int epochs = 1000,int batchSize = 64)
        {
            for(int epoch = 1; epoch <= epochs; epoch++)
            {
                var indices = Enumerable.Range(0, X.Length).OrderBy(_ => rand.Next()).ToArray();
                X = indices.Select(i => X[i]).ToArray();
                yOneHot = indices.Select(i => yOneHot[i]).ToArray();
                for(int start = 0; start < X.Length; start += batchSize)
                {
                    int end = Math.Min(start + batchSize, X.Length);
                    int currentBatchSize = end - start;
                    double[][] gradW1 = new double[_hidenSize][];
                    for (int i = 0; i < gradW1.Length; i++) gradW1[i] = new double[_inputSize];
                    double[] gradB1 = new double[_hidenSize];
                    double[][] gradW2 = new double[_outputSize][];
                    for (int i = 0; i < gradW2.Length; i++) gradW2[i] = new double[_hidenSize];
                    double[] gradB2 = new double[_outputSize];
                    for(int idx = start; idx < end; idx++)
                    {
                        double[] Xi = X[idx];
                        double[] Y = yOneHot[idx];
                        double[] z1 = MultiplyMatrixVector(weights1, Xi, bias1);
                        double[] a1 = ReLU(z1);
                        double[] z2 = MultiplyMatrixVector(weights2, a1, bias2);
                        double[] a2 = SoftMax(z2);
                        double[] delta2 = a2.Zip(Y, (p, z) => p - z).ToArray();
                        for (int i = 0; i < gradW2.Length; i++)
                        {
                            for (int j = 0; j < a2.Length; j++) gradW2[i][j] += delta2[i] * a1[j];
                            gradB2[i] += delta2[i];
                        }
                        double[] delta1 = new double[_hidenSize];
                        for (int i = 0; i < delta1.Length; i++)
                        {
                            double sum = 0;
                            for (int j = 0; j < _outputSize; j++) sum += weights2[j][i] * delta2[j];
                            delta1[i] = sum * (z1[i] > 0 ? 1 : 0);
                        }
                        for (int i = 0; i < _hidenSize; i++)
                        {
                            for (int j = 0; j < _inputSize; j++)
                                gradW1[i][j] += delta1[i] * Xi[j];
                            gradB1[i] += delta1[i];
                        }
                    }
                    for (int i = 0; i < 128; i++)
                    {
                        for (int j = 0; j < 784; j++)
                            gradW1[i][j] /= currentBatchSize;
                        gradB1[i] /= currentBatchSize;
                    }
                    for (int i = 0; i < 10; i++)
                    {
                        for (int j = 0; j < 128; j++)
                            gradW2[i][j] /= currentBatchSize;
                        gradB2[i] /= currentBatchSize;
                    }
                    for (int i = 0; i < 128; i++)
                    {
                        bias1[i] -= learningRate * gradB1[i];
                        for (int j = 0; j < 784; j++)
                            weights1[i][j] -= learningRate * gradW1[i][j];
                    }
                    for (int i = 0; i < 10; i++)
                    {
                        bias2[i] -= learningRate * gradB2[i];
                        for (int j = 0; j < 128; j++)
                            weights2[i][j] -= learningRate * gradW2[i][j];
                    }
                }
            }
        }

        public virtual double[] ReLU(double[] z) => z.Select(x => Math.Max(0, x)).ToArray();

        public virtual double[] SoftMax(double[] z)
        {
            double max = z.Max();
            double[] exp = z.Select(x => Math.Exp(x - max)).ToArray();
            double sum = exp.Sum();
            return exp.Select(e => e / sum).ToArray();
        }

        public virtual double[] MultiplyMatrixVector(double[][] W, double[] x, double[] b)
        {
            double[] z = new double[W.Length];
            for (int i = 0; i < W.Length; i++)
            {
                double sum = 0;
                for (int j = 0; j < W[i].Length; j++) sum += W[i][j] * x[j];
                z[i] += sum + b[i];
            }
            return z;
        }

        public virtual double[] Predict(double[] x) => SoftMax(MultiplyMatrixVector(weights2, ReLU(MultiplyMatrixVector(weights1, x, bias1)), bias2));
        
        public virtual double ComputeAccuracy(double[][] X, double[][] Y)
        {
            int ans = 0;
            for(int i = 0; i < X.Length; i++)
            {
                double[] a2 = Predict(X[i]);
                if (AgrMax(a2) == AgrMax(Y[i])) ans++;
            }
            return ans/X.Length;
        }
        

        public double AgrMax(double[] m)
        {
            int maxAgr = 0;
            for (int i = 1; i < m.Length; i++) maxAgr = m[i] > m[maxAgr] ? i : maxAgr;
            return maxAgr;
        }


        private double[][] InitializeWeights(int n, int m)
        {
            double[][] W = new double[n][];
            for (int i = 0; i < n; i++)
            {
                W[i] = new double[m];
                for (int j = 0; j < m; j++) W[i][j] = (rand.NextDouble() - 0.5) * 0.01;
            }
            return W;
        }
    }
}
