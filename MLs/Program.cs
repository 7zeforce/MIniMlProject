using System.Text;
using MLs;
using MLs.Components;

Random rand = new Random(42);
Console.ForegroundColor = ConsoleColor.Red;
ParseData pd = new ParseData();
string pathTrainLabels = @"C:\ProjectC#\MIniMlProject\MLs\DataForLearning\mnist-master\train-labels.idx1-ubyte";
string pathTrainImage = @"C:\ProjectC#\MIniMlProject\MLs\DataForLearning\mnist-master\train-images.idx3-ubyte";
string pathTestImage = @"C:\ProjectC#\MIniMlProject\MLs\DataForLearning\mnist-master\t10k-images.idx3-ubyte";
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
NeuralNetwork model = new NeuralNetwork(X_train[0].Length,128,10);
Console.WriteLine($"{ComputeAverageLoss(X_train, Y_train)}");
double[][] X_small = X_train.Take(10).ToArray();
double[][] Y_small = Y_train.Take(10).ToArray();
model.Fit(Y_train, X_train, 0.1, 30, 64);
char[,] buff = new char[28,28];
Dictionary<double, char> symbols = new Dictionary<double,char> { {0.0, ' '}, {0.1, ' '}, { 0.2, '.' }, { 0.3, ':' }, 
    { 0.4, '-' }, { 0.5, '=' }, { 0.6, '+' }, { 0.7, '*' }, { 0.8, '#' }, { 0.9, '%' }, { 1.0, '@' } };
for (int i = 0; i < X_small.Length; i++)
{
    for (int j = 0; j < 28; j++)
    {
        for(int k = 0; k < 28; k++)
        {
            buff[j, k] = symbols.TryGetValue(Math.Round(X_small[i][j * 28 + k], 1), out char value) ? value : ' ';
        }
    }
    var sb = new StringBuilder();
    for (int k = 0; k < 28; k++)
    {
        for (int m = 0; m < 28; m++) sb.Append(buff[k, m]);
        sb.AppendLine();
    }
    double[] a2 = model.Predict(X_small[i]);
    sb.Append($"NeuralNetwork ansver = {model.AgrMax(a2)}\n");
    Console.Write(sb.ToString());
}
double ComputeAverageLoss(double[][] X, double[][] yOneHot)
{
    double totalLoss = 0.0;
    int n = X.Length;
    for (int i = 0; i < n; i++)
    {
        double[] a2 = model.Predict(X[i]);
        totalLoss += CrossEntropy(a2, yOneHot[i]);
    }
    return totalLoss / n;
}

double CrossEntropy(double[] prod, double[] y)
{
    double loss = 0.0;
    for(int i = 0; i < prod.Length; i++)
    {
        loss += y[i] * Math.Log(prod[i] + 1e-15);
    }
    return -loss;
}