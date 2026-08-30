using MLs.Components;

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
            Matrix2x2 m = new Matrix2x2(Math.Cos(angle), -Math.Sin(angle), Math.Sin(angle), Math.Cos(angle));
            return this * m;
        }
    }
}
