
namespace MLs.Components
{
    public struct Vector2
    {
        public double X; public double Y;

        public Vector2(double x, double y)
        {
            X = x; Y = y;
        }

        public Vector2 TurnAngle(double angle)
        {
            Matrix2x2 m = new Matrix2x2((float)Math.Cos(angle), (float)-Math.Sin(angle), (float)Math.Sin(angle), (float)Math.Cos(angle));
            return this * m;
        }
    }
}
