using MLs;
using MLs.Components;

Random rand = new Random(42);
Console.ForegroundColor = ConsoleColor.Red;

ParseData pd = new ParseData();
string pathTrainLabels = @"C:\ProjectC#\MIniMlProject\MLs\DataForLearning\mnist-master\train-labels.idx1-ubyte";
string pathTrainImage = @"C:\ProjectC#\MIniMlProject\MLs\DataForLearning\mnist-master\train-images.idx3-ubyte";
string pathTestImage = @"C:\ProjectC#\MIniMlProject\MLs\DataForLearning\mnist-master\t10k-images.idx1-ubyte";
string pathTestLabels = @"C:\ProjectC#\MIniMlProject\MLs\DataForLearning\mnist-master\t10k-labels.idx1-ubyte";
DigitImage[] train = pd.ReadLabels(pathTrainLabels, pathTrainImage);
DigitImage[] test = pd.ReadLabels(pathTestLabels, pathTestImage);
double[][] X_train = train.Select(x => x.Pixels.Select(p => p / 255.0).ToArray()).ToArray();
double[][] Y_train = new double[X_train.Length][];
for(int i = 0; i < Y_train.Length; i++)
{
    Y_train[i] = new double[10];
    Y_train[i][train[i].Label] = 1;
}
double[][] X_test = test.Select(x => x.Pixels.Select(p => p / 255.0).ToArray()).ToArray();
double[][] Y_test = new double[X_test.Length][];
for (int i = 0; i < Y_test.Length; i++)
{
    Y_test[i] = new double[10];
    Y_test[i][test[i].Label] = 1;
}
double[][] W1 = new double[128][];
double[] B1 = new double[128];

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