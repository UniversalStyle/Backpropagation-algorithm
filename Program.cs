using System.Collections.Generic;
using System;
using LinearAlgebra;

namespace Programm
{
    class CalculatingMatrix
    {
        /// <summary>
        /// Вычисление ошибки сети
        /// </summary>
        /// <param name="vector"></param>
        /// <returns></returns>
        public static double Emse(Matrix<double> vector)
        {
            double result = (double)0.0;
            for (int countOfRows = 0; countOfRows < vector.RowsCount(); countOfRows++)
            {
                result += Math.Pow(vector[countOfRows, 0], 2);
            }
            return result / vector.RowsCount();
        }

        /// <summary>
        /// Наивное умножение векторов
        /// </summary>
        /// <param name="vector1"></param>
        /// <param name="vector2"></param>
        /// <returns></returns>
        public static Matrix<double> NaiveMultiplication(Matrix<double> vector1, Matrix<double> vector2)
        {
            Matrix<double> result = new Matrix<double>(vector1.RowsCount(), 1);
            if (vector1.RowsCount() == vector2.RowsCount())
            {
                for (int countOfRows = 0; countOfRows < vector1.RowsCount(); countOfRows++)
                {
                    result[countOfRows, 0] = vector1[countOfRows, 0] * vector2[countOfRows, 0];
                }
                return result;
            }
            else
            {
                throw new Exception("Error: the dimensions of the vectors do not match." +
                "\nYour vectors:" +
                "\nVector vector1: " + vector1.RowsCount() + " rows and " + vector1.ColumnsCount() + " columns" +
                "\nVector vector2: " + vector2.RowsCount() + " rows and " + vector2.ColumnsCount() + " columns");
            }

        }

        /// <summary>
        /// Вычисление do*(1 - do) или dh*(1 - dh)
        /// </summary>
        /// <param name="vector"></param>
        /// <returns></returns>
        public static Matrix<double> FHatch(Matrix<double> vector)
        {
            for (int countOfRows = 0; countOfRows < vector.RowsCount(); countOfRows++)
            {
                vector[countOfRows, 0] *= (1 - vector[countOfRows, 0]);
            }
            return vector;
        }

        /// <summary>
        /// Вычисление градиента:
        /// 1. для каждого входа нейронов скрытого слоя (DeltaHidden, x)
        /// 2. для каждого входа нейронов выходного слоя (DeltaOut, dh)
        /// </summary>
        /// <param name="delta"></param>
        /// <param name="vector"></param>
        /// <returns></returns>
        public static Matrix<double> Grad(Matrix<double> delta, Matrix<double> vector)
        {
            Matrix<double> result = new Matrix<double>(delta.RowsCount(), vector.ColumnsCount());
            result = delta.Multiply(vector.Transpose());
            return result;
        }

        /// <summary>
        /// Вычисление дельт для нейронов скрытого слоя
        /// </summary>
        /// <param name="w2"></param>
        /// <param name="dh"></param>
        /// <param name="y"></param>
        /// <param name="d0"></param>
        /// <returns></returns>
        public static Matrix<double> DeltaHidden(Matrix<double> w1, Matrix<double> w2, Matrix<double> x, Matrix<double> y)
        {
            return NaiveMultiplication(w2.Transpose().Multiply(DeltaOut(w1, w2, x, y)), FHatch(VectorSigmoid(w1.Multiply(x))));
        }

        /// <summary>
        /// Вычисление дельт для нейронов выходного слоя: (y - dO) * f'(dO) = e * f'(dO)
        /// </summary>
        /// <param name="y"></param>
        /// <param name="d0"></param>
        /// <returns></returns>
        public static Matrix<double> DeltaOut(Matrix<double> w1, Matrix<double> w2, Matrix<double> x, Matrix<double> y)
        {
            return NaiveMultiplication(DeltaE(y,  VectorSigmoid(w2.Multiply(VectorSigmoid(w1.Multiply(x))))), FHatch( VectorSigmoid(w2.Multiply(VectorSigmoid(w1.Multiply(x))))));
        }

        /// <summary>
        /// Вычисление e = y - d0
        /// </summary>
        /// <param name="y"></param>
        /// <param name="d0"></param>
        /// <returns></returns>
        public static Matrix<double> DeltaE(Matrix<double> y, Matrix<double> d0)
        {
            Matrix<double> result = new Matrix<double>(y.RowsCount(), y.ColumnsCount());
            if (y.RowsCount() == d0.RowsCount())
            {
                for (int countOfRows = 0; countOfRows < d0.RowsCount(); countOfRows++)
                {
                    result[countOfRows, 0] = y[countOfRows, 0] - d0[countOfRows, 0];
                }
                return result;
            }
            else
            {
                throw new Exception("Error: the dimensions of the vectors do not match." +
                "\nYour vectors:" +
                "\nVector y: " + y.RowsCount() + " rows and " + y.ColumnsCount() + " columns" +
                "\nVector d0: " + d0.RowsCount() + " rows and " + d0.ColumnsCount() + " columns");
            }
        }

        /// <summary>
        /// Вычисление сигмоиды для вектора (вычисляется dh, d0)
        /// </summary>
        /// <param name="vector"></param>
        /// <returns></returns>
        public static Matrix<double> VectorSigmoid(Matrix<double> vector)
        {
            for (int countOfRows = 0; countOfRows < vector.RowsCount(); countOfRows++)
            {
                vector[countOfRows, 0] = Sigmoid(vector[countOfRows, 0]);
            }
            return vector;
        }

        /// <summary>
        /// Вычисление сигмоиды
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double Sigmoid(double value)
        {
            double k = (double)Math.Exp(value);
            return k / ((double)1.0 + k);
        }

        /// <summary>
        /// Вычисление весовых коэффициентов:
        /// n - буква греческого алфавита "эта", dh и d0 берутся из метода VectorSigmoid()
        /// </summary>
        /// <param name="w1"></param>
        /// <param name="w2"></param>
        /// <param name="dh"></param>
        /// <param name="d0"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="iteration"></param>
        /// <param name="n"></param>
        /// <param name="alfa"></param>
        /// <returns></returns>
        public static List<List<Matrix<double>>> Weights(Matrix<double> w1, Matrix<double> w2, Matrix<double> x, Matrix<double> y, int iteration, double n, double alfa)
        {
            List<List<Matrix<double>>> data = new List<List<Matrix<double>>>();
            List<Matrix<double>> deltaW1 = new List<Matrix<double>>();
            List<Matrix<double>> deltaW2 = new List<Matrix<double>>();
            List<Matrix<double>> matrixW1 = new List<Matrix<double>>();
            List<Matrix<double>> matrixW2 = new List<Matrix<double>>();
            double oldAlfa = alfa;
            int dimension = 0;
            double lastError = 0, error = 0;

            for (int i = 0; i < 2; i++)
            {
                deltaW1.Add(w1);
                deltaW2.Add(w2);
                matrixW1.Add(w1);
                matrixW2.Add(w2);
            }

            if (matrixW1[1] != null && matrixW2[1] != null && deltaW1[1] != null && deltaW2[1] != null)
            {
                for (int t = 0; t < iteration && (error < lastError + (double)0.0000000000000001 || error < lastError - (double)0.0000000000000001); t++)
                {
                    alfa = t == 0 ? 0 : oldAlfa;
                    dimension = t % 2 == 0 ? 0 : 1;
                    lastError = Emse(DeltaE(y, VectorSigmoid(matrixW2[dimension].Multiply(VectorSigmoid(matrixW1[dimension].Multiply(x))))));
                    if (t == 0)
                    {
                        deltaW1[0] = (Grad(DeltaHidden(matrixW1[0], matrixW2[0], x, y), x).Multiply(n)).Add(deltaW1[0].Multiply(alfa));
                        deltaW2[0] = (Grad(DeltaOut(matrixW1[0], matrixW2[0], x, y), VectorSigmoid(matrixW1[0].Multiply(x))).Multiply(n)).Add(deltaW2[0].Multiply(alfa));
                        matrixW1[0] = matrixW1[0].Add(deltaW1[0]);
                        matrixW2[0] = matrixW2[0].Add(deltaW2[0]);
                    }
                    else
                    {
                        if (t % 2 != 0)
                        {
                            deltaW1[1] = (Grad(DeltaHidden(matrixW1[0], matrixW2[0], x, y), x).Multiply(n)).Add(deltaW1[0].Multiply(alfa));
                            deltaW2[1] = (Grad(DeltaOut(matrixW1[0], matrixW2[0], x, y), VectorSigmoid(matrixW1[0].Multiply(x))).Multiply(n)).Add(deltaW2[0].Multiply(alfa));
                            matrixW1[1] = matrixW1[0].Add(deltaW1[1]);
                            matrixW2[1] = matrixW2[0].Add(deltaW2[1]);
                        }
                        else
                        {
                            deltaW1[0] = (Grad(DeltaHidden(matrixW1[1], matrixW2[1], x, y), x).Multiply(n)).Add(deltaW1[1].Multiply(alfa));
                            deltaW2[0] = (Grad(DeltaOut(matrixW1[1], matrixW2[1], x, y), VectorSigmoid(matrixW1[1].Multiply(x))).Multiply(n)).Add(deltaW2[1].Multiply(alfa));
                            matrixW1[0] = matrixW1[1].Add(deltaW1[0]);
                            matrixW2[0] = matrixW2[1].Add(deltaW2[0]);
                        }
                    }
                    error = Emse(DeltaE(y, VectorSigmoid(matrixW2[dimension].Multiply(VectorSigmoid(matrixW1[dimension].Multiply(x))))));
                }

                data.Add(deltaW1);
                data.Add(deltaW2);
                data.Add(matrixW1);
                data.Add(matrixW2);

                return data;
            }
            else
            {
                throw new Exception("No lists have been created.");
            }


        }
    }

    class Program
    {
        /// <summary>
        /// Вывод матрицы на конслоль
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="str"></param>
        public static void PrintMatrix(Matrix<double> matrix, string str)
        {
            for (int countOfRows = 0; countOfRows < matrix.RowsCount(); countOfRows++)
            {
                Console.Write(str);
                for (int countOfColumns = 0; countOfColumns < matrix.ColumnsCount(); countOfColumns++)
                {
                    Console.Write(matrix[countOfRows, countOfColumns] + "\t");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Заполнение матрицы случайным числом
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public static Matrix<double> CreateMatrix(Matrix<double>  matrix)
        {
            Random value = new Random();
            for (int countOfRows = 0; countOfRows < matrix.RowsCount(); countOfRows++)
            {
                for (int countOfColumns = 0; countOfColumns < matrix.ColumnsCount(); countOfColumns++)
                {
                    matrix[countOfRows, countOfColumns] = value.NextDouble();
                }
            }
            return matrix;
        }

        /// <summary>
        /// Вычисление матрицы
        /// </summary>
        public static void CalculationMatrix()
        {
            int countOfColumns, countOfRows, iteration;
            double lastError, error, alfa, n;
            Console.Write("Введите количество столбцов матрицы: ");
            countOfColumns = int.Parse(Console.ReadLine());
            Console.Write("Введите количество строк матрицы: ");
            countOfRows = int.Parse(Console.ReadLine());
            Console.Write("Введите количество итераций, но помните, что при iter > 1 выводиться будет подробное решение только для последней итерации: ");
            iteration = int.Parse(Console.ReadLine());
            Console.Write("Введите параметр скорости обучения(буква греческого алфавита 'эта'): ");
            n = double.Parse(Console.ReadLine());
            Console.Write("Введите момент(альфа): ");
            alfa = double.Parse(Console.ReadLine());

            // Создание матриц
            Matrix<double>  dh = new Matrix<double> (countOfRows, 1);
            Matrix<double>  d0 = new Matrix<double> (countOfRows, 1);
            Matrix<double>  x = new Matrix<double> (countOfRows, 1);
            Matrix<double>  y = new Matrix<double> (countOfRows, 1);
            Matrix<double>  w1 = new Matrix<double> (countOfRows, countOfColumns);
            Matrix<double>  w2 = new Matrix<double> (countOfRows, countOfColumns);
            Matrix<double>  deltaW1 = new Matrix<double> (countOfRows, countOfColumns);
            Matrix<double>  deltaW2 = new Matrix<double> (countOfRows, countOfColumns);
            Matrix<double>  deltaHidden = new Matrix<double> (countOfRows, 1);
            Matrix<double>  deltaOut = new Matrix<double> (countOfRows, 1);

            // Генерация случайных значений матриц
            x = CreateMatrix(x);
            y = CreateMatrix(y);
            w1 = CreateMatrix(w1);
            w2 = CreateMatrix(w2);

            // Вывод созданных матриц
            Console.WriteLine("\nW1:"); PrintMatrix(w1, "\t");
            Console.WriteLine("W2:"); PrintMatrix(w2, "\t");
            Console.WriteLine("X:"); PrintMatrix(x, "\t");
            Console.WriteLine("Y:"); PrintMatrix(y, "\t");

            Console.WriteLine("Матричный расчет выходов скрытого и выходного слоя:");
            Console.WriteLine("\tdh = W1 x X");
            dh = CalculatingMatrix.VectorSigmoid(w1.Multiply(x));
            d0 = CalculatingMatrix.VectorSigmoid(w2.Multiply(dh));
            PrintMatrix(dh, "\t\t");

            Console.WriteLine("\td0 = W2 x dh");
            PrintMatrix(d0, "\t\t");

            Console.WriteLine("\te = y - d0");
            PrintMatrix(CalculatingMatrix.DeltaE(y, CalculatingMatrix.VectorSigmoid(w2.Multiply(CalculatingMatrix.VectorSigmoid(w1.Multiply(x))))), "\t\t");

            Console.Write("Первичная ошибка равна: ");
            lastError = CalculatingMatrix.Emse(CalculatingMatrix.DeltaE(y, CalculatingMatrix.VectorSigmoid(w2.Multiply(dh))));
            Console.WriteLine(lastError);

            Console.WriteLine("Новые значения весовых коэффициентов:");
            deltaOut = CalculatingMatrix.DeltaOut(w1, w2, x, y);
            deltaHidden = CalculatingMatrix.DeltaHidden(w1, w2, x, deltaOut);

            Console.WriteLine("Вычисление дельт для нейронов выходного слоя: ");
            PrintMatrix(deltaOut, "\t\t");

            Console.WriteLine("Вычисление дельт для нейронов скрытого слоя: ");
            PrintMatrix(deltaHidden, "\t\t");

            Console.WriteLine("Вычисление градиента для каждого входа нейронов выходного и скрытого слоев: ");
            Console.WriteLine("\tGradW1:");
            PrintMatrix(CalculatingMatrix.Grad(deltaHidden, x), "\t\t");
            Console.WriteLine("\tGradW2:");
            PrintMatrix(CalculatingMatrix.Grad(deltaOut, CalculatingMatrix.VectorSigmoid(w1.Multiply(x))), "\t\t");

            List<List<Matrix<double> >> list = CalculatingMatrix.Weights(w1, w2, x, y, iteration, n, alfa);
            if (iteration % 2 == 0)
            {
                deltaW1 = list[0][1];
                deltaW2 = list[1][1];
                w1 = list[2][1];
                w2 = list[3][1];
            }
            else
            {
                deltaW1 = list[0][0];
                deltaW2 = list[1][0];
                w1 = list[2][0];
                w2 = list[3][0];
            }

            Console.WriteLine("Вычислениe корректирующих значений весовых коэффициентов:");
            Console.WriteLine("\tdeltaW1:");
            PrintMatrix(deltaW1, "\t\t");
            Console.WriteLine("\tdeltaW2:");
            PrintMatrix(deltaW2, "\t\t");

            Console.WriteLine("Новые значения весовых коэффициентов:");
            Console.WriteLine("\tW1:"); 
            PrintMatrix(w1, "\t\t");
            Console.WriteLine("\tW2:"); 
            PrintMatrix(w2, "\t\t");

            Console.Write("Вычислим конечную ошибку: ");
            dh = CalculatingMatrix.VectorSigmoid(w1.Multiply(x));
            d0 = CalculatingMatrix.VectorSigmoid(w2.Multiply(dh));
            error = CalculatingMatrix.Emse(CalculatingMatrix.DeltaE(y, CalculatingMatrix.VectorSigmoid(w2.Multiply(CalculatingMatrix.VectorSigmoid(w1.Multiply(x))))));
            if((100 - ((error / lastError) * 100)) < 0)
            {
                throw new Exception("Error < 0");
            }
            else {
                Console.WriteLine(error);
                Console.WriteLine("Сравнение ошибок: " + "ошибка упала на " + (100 - ((error / lastError) * 100)) + "%");
            }
        }

        /// <summary>
        /// Точка входа
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            CalculationMatrix();
        }
    }
}