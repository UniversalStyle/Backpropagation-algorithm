using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace Programm
{
    class CalculatingMatrix
    {
        //
        //  Вычисление ошибки сети 
        //
        public static double Emse(Matrix<double> vector)
        {
            double result = (double)0.0;
            for (int countOfRows = 0; countOfRows < vector.RowCount; countOfRows++)
            {
                result += Math.Pow(vector[countOfRows, 0], 2);
            }
            return result / 3;
        }

        //
        //  Наивное умножение векторов
        //
        public static Matrix<double> NaiveMultiplication(Matrix<double> v1, Matrix<double> v2)
        {
            Matrix<double> result = Matrix<double>.Build.Dense(v1.RowCount, 0);
            for (int countOfRows = 0; countOfRows < v1.RowCount; countOfRows++)
            {
                result[countOfRows, 0] = v1[countOfRows, 0] * v2[countOfRows, 0];
            }
            return result;
        }

        //
        //  Вычисление do*(1 - do)
        //
        public static Matrix<double> FHatch(Matrix<double> vector)
        {
            for (int countOfRows = 0; countOfRows < vector.RowCount; countOfRows++)
            {
                vector[countOfRows, 0] *= (1 - vector[countOfRows, 0]);
            }
            return vector;
        }

        //
        //  Транспонирование вектора
        //
        public static Matrix<double> TransposeVector(Matrix<double> vector)
        {
            Matrix<double> transpose = Matrix<double>.Build.Dense(1, vector.RowCount);
            for (int countOfRows = 0; countOfRows < vector.RowCount; countOfRows++)
            {
                transpose[0, countOfRows] = vector[countOfRows, 0];
            }
            return transpose;
        }

        //  Вычисление градиента:
        //  1. для каждого входа нейронов выходного слоя
        //  2. для каждого входа нейронов скрытого слоя
        public static Matrix<double> Grad(Matrix<double> delta, Matrix<double> vector)
        {
            Matrix<double> result = Matrix<double>.Build.Dense(delta.RowCount, vector.RowCount);
            delta.Multiply(TransposeVector(vector), result);
            return result;
        }

        //
        //  Вычисление дельт для нейронов скрытого слоя
        //
        public static Matrix<double> DeltaHidden(Matrix<double> w2, Matrix<double> dh, Matrix<double> y, Matrix<double> d0)
        {
            Matrix<double> multiplyResult = Matrix<double>.Build.Dense(y.RowCount, y.ColumnCount);
            w2.Transpose().Multiply(DeltaOut(DeltaE(y, d0), FHatch(d0)), multiplyResult);
            return NaiveMultiplication(multiplyResult, FHatch(dh));
        }

        //
        //  Вычисление дельт для нейронов выходного слоя: (y - dO) * f'(dO) = e * f'(dO)
        //
        public static Matrix<double> DeltaOut(Matrix<double> y, Matrix<double> d0)
        {
            return NaiveMultiplication(DeltaE(y, d0), FHatch(d0));
        }

        //
        //  Вычисление e = y - d0
        //
        public static Matrix<double> DeltaE(Matrix<double> y, Matrix<double> d0)
        {

            Matrix<double> result = Matrix<double>.Build.Dense(y.RowCount, 0);
            for (int countOfRows = 0; countOfRows < y.RowCount; countOfRows++)
            {
                result[countOfRows, 0] = y[countOfRows, 0] - d0[countOfRows, 0];
            }
            return result;
        }

        //
        // Вычисление сигмоиды для вектора (вычисляется dh, d0)
        //
        public static Matrix<double> VectorSigmoid(Matrix<double> vector)
        {
            for (int countOfRows = 0; countOfRows < vector.RowCount; countOfRows++)
            {
                vector[countOfRows, 0] = Sigmoid(vector[countOfRows, 0]);
            }
            return vector;
        }

        //
        // Вычисление сигмоиды
        //
        public static double Sigmoid(double value)
        {
            double k = Math.Exp(value);
            return k / ((double)1.0 + k);
        }

        //
        // Умножение матрицы на константу
        //

        public static Matrix<double> MultiplicationWithConst(Matrix<double> matrix, double constant)
        {
            for(int countOfRows = 0; countOfRows < matrix.RowCount; countOfRows++)
            {
                for(int countOfColumns = 0; countOfColumns < matrix.ColumnCount; countOfColumns++)
                {
                    matrix[countOfRows, countOfColumns] *= constant;
                }
            }
            return matrix;
        }

        //
        // Вычисление весовых коэффициентов
        // n - буква греческого алфавита "эта", dh и d0 берутся из метода VectorSigmoid()
        public static List<List<Matrix<double>>> Weights(Matrix<double> w1, Matrix<double> w2, Matrix<double> dh, Matrix<double> d0, Matrix<double> x, Matrix<double> y, int iteration, double n)
        {
            List<List<Matrix<double>>> deltaMatrix = new List<List<Matrix<double>>>();
            List<List<Matrix<double>>> matrix = new List<List<Matrix<double>>>();
            matrix[0][0] = w1;
            matrix[1][0] = w2;
            for(int t = 0; t < iteration; t++)
            {
                if(t == 0)
                {
                    deltaMatrix[0][0] = MultiplicationWithConst(Grad(DeltaHidden(w2, dh, y, d0), TransposeVector(x)), n);
                    deltaMatrix[1][0] = MultiplicationWithConst(Grad(DeltaOut(y, d0), TransposeVector(dh)), n);
                }
                if(t > 0)
                {
                    MultiplicationWithConst(Grad(DeltaHidden(w2, dh, y, d0), TransposeVector(x)), n).Add(deltaMatrix[0][t - 1], deltaMatrix[0][t]);
                    MultiplicationWithConst(Grad(DeltaOut(y, d0), TransposeVector(dh)), n).Add(deltaMatrix[1][t - 1], deltaMatrix[1][t]);
                    matrix[0][t - 1].Add(deltaMatrix[0][t],matrix[0][t]);
                    matrix[1][t - 1].Add(deltaMatrix[1][t],matrix[1][t]);
                }
            }
            return matrix;
        }
    }

    class Program
    {
        public static void CalculationMatrix(int countOfColumns, int countOfRows)
        {
            Matrix<double> dh;
            Matrix<double> d0;
            Matrix<double> x = Matrix<double>.Build.Random(countOfRows, countOfColumns);
            Matrix<double> y = Matrix<double>.Build.Random(countOfRows, countOfColumns);
            Matrix<double> w1 = Matrix<double>.Build.Random(countOfRows, countOfColumns);
            Matrix<double> w2 = Matrix<double>.Build.Random(countOfRows, countOfColumns);
        }

        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
        }
    }
}