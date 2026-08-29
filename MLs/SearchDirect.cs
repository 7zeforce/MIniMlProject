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

        public double MSE(Dictionary<char, double> predict)
        {
            double summ = 0;
            for (int i = 0; i < _points.Count; i++)
            {
                Vector2 v = predict['k'] != 0 ? new Vector2(_points[i].X + (predict['b'] / predict['k']), _points[i].Y) : 
                    new Vector2(_points[i].X, _points[i].Y);
                v.TurnAngle(-Math.Atan(predict['k']));
                summ += v.Y * v.Y;
            }
            return summ / _points.Count;
        }
    }
}
