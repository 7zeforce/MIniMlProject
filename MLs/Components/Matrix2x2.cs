
namespace MLs.Components
{
    public struct Matrix2x2
    {
        public double M11; public double M12;
        public double M21; public double M22;

        public Matrix2x2(float m11, float m12, float m21, float m22)
        {
            M11 = m11; M12 = m12;
            M21 = m21; M22 = m22;
        }

        public static Vector2 operator *(Vector2 v, Matrix2x2 m)
        {
            return new Vector2(v.X * m.M11 + v.Y * m.M12, v.X * m.M21 + v.Y * m.M22);
        }
    }
}
