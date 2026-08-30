using MLs;
using MLs.Components;

List<Vector2> points = new List<Vector2>()
{ new (1,2), new (2,4), new (3,5), new (4,4), new (5,5) };
SearchDirect searhDirect = new SearchDirect(points);
Random rand = new Random(42);
double[][] X = new double[200][];
double[] y = new double[200];
double[] tW = { 3, -2 };
double b = 5;
for (int i = 0; i < y.Length; i++)
{
    X[i] = new double[] { rand.NextDouble() * 20 - 10, rand.NextDouble() * 20 - 10 };
    y[i] = Formul(tW, b, X[i][0], X[i][1], (rand.NextDouble() - 0.5) * 0.5);
}

LinearRegression model = new LinearRegression(new double[2], 0);
model.Fit(X, y, 0.01);
Console.WriteLine($"koefficents: ({model.Weights[0]},{model.Weights[1]})");
double Formul(double[] ws, double b, double x1, double x2, double noise) => ws[0] * x1 + ws[1] * x2 + b + noise;