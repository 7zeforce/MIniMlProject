using MLs.Components;

namespace MLs
{
    public sealed class SearchDirect
    {
        private List<Vector2> _points;
        
        public SearchDirect(List<Vector2> points)
        {
            _points = points;
        }

        public Dictionary<char, double> SearchParametrs()
        {
            Dictionary<char, double> parametrs = new Dictionary<char, double>(2);
            double meanX = _points.Average<Vector2>(p => p.X);
            double meanY = _points.Average<Vector2>(p => p.Y);
            double cov = 0, varX = 0;
            foreach(Vector2 v in _points)
            {
                cov += (v.X - meanX) * (v.Y - meanY);
                varX += Math.Pow((v.X - meanX), 2);
            }
            parametrs.Add('k', cov / varX);
            parametrs.Add('b', meanY - parametrs['k'] * meanX);
            return parametrs;
        }
        
        public (double, double) FitGradientDescent(double learningRate, int epochs = 1000)
        {
            double k = 0.0, b = 0.0;
            for (int epoch = 1; epoch < epochs; epoch++)
            {
                double gradK = 0, gradB = 0, loss = 0;
                for (int i = 0; i < _points.Count; i++)
                {
                    double predY = k * _points[i].X + b;
                    double error = predY - _points[i].Y;
                    gradB += error; gradK += error * _points[i].X;
                    loss += Math.Pow(error, 2);
                }
                gradB *= (2.0 / _points.Count);
                gradK *= (2.0 / _points.Count);
                double NumB = NumericGradient(p => MSE(k, p),b);
                double NumK = NumericGradient(p => MSE(p, b),k);
                Console.WriteLine($"Gradient b difference: {Math.Abs(gradB - NumB) <= 1e-7}\tk difference: {Math.Abs(gradK - NumK) <= 1e-7}");
                loss /= _points.Count;
                k -= learningRate * gradK;
                b -= learningRate * gradB;
            }
            return (k,b);
        }

        public (List<double> Normalized, double Mean, double Std) Standardize(List<double> values)
        {
            double mean = values.Average();
            double std = Math.Sqrt(values.Sum(x => (x - mean) * (x - mean))/values.Count);
            if (std == 0) return (values, mean, 1.0);
            return (values.Select(v => (v - mean) / std).ToList(), mean, std);
        }

        public double MSE(double k, double b)
        {
            double summ = 0;
            for (int i = 0; i < _points.Count; i++)
            {
                Vector2 v = _points[i];
                summ += Math.Pow((v.Y - (v.X * k + b)),2);
            }
            return summ / _points.Count;
        }

        private double NumericGradient(Func<double, double> f, double param, double h = 1e-5) => (f(param + h) - f(param - h)) / (2 * h);
    }
}
