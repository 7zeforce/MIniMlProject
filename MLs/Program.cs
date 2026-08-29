using MLs;
using MLs.Components;

List<Vector2> points = new List<Vector2>()
{ new (1,2), new (2,4), new (3,5), new (4,4), new (5,5) };
SearchDirect searhDirect = new SearchDirect(points);
Dictionary<char, double> param = searhDirect.SearchParametrs();
Console.WriteLine($"k = {param['k']}\nb = {param['b']}\nMSE = {searhDirect.MSE(param)}");