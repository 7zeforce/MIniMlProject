using MLs;
using MLs.Components;

List<Vector2> points = new List<Vector2>()
{ new (1,2), new (2,4), new (3,5), new (4,4), new (5,5) };
SearchDirect searhDirect = new SearchDirect(points);
Random rand = new Random(42);
double[][] X = new double[200][];
double[] y = new double[200];
double[] tW = { 3 };
double b = -2;
LogicalRegression model = new LogicalRegression(new double[1], 0);

(double,double) Test(double[] y)
{
    int a = 0, b = 0;
    for(int i = 0; i < y.Length; i++)
    {
        if (y[i] == 1) a++;
        else b++;
    }
    return (a, b);
}