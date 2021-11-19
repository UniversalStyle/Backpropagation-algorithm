using System.Collections.Generic;
using System;
using Math;

namespace Programm
{
    class CalculatingMatrix
    {
        ///
        /// Вычисление ошибки сети 
        ///
        public static double Emse(Matrix vector)
        {
            double result = (double)0.0;
            for (int countOfRows = 0; countOfRows < vector.RowsCount(); countOfRows++)
            {
                result += System.Math.Pow(vector[countOfRows, 0], 2);
            }
            return result / 3;
        }

        //
        //  Наивное умножение векторов
        //
        public static Matrix NaiveMultiplication(Matrix v1, Matrix v2)
        {
            Matrix result = new Matrix(v1.RowsCount(), 1);
            for (int countOfRows = 0; countOfRows < v1.RowsCount(); countOfRows++)
            {
                result[countOfRows, 0] = v1[countOfRows, 0] * v2[countOfRows, 0];
            }
            return result;
        }

        //
        //  Вычисление do*(1 - do)
        //
        public static Matrix FHatch(Matrix vector)
        {
            for (int countOfRows = 0; countOfRows < vector.RowsCount(); countOfRows++)
            {
                vector[countOfRows, 0] *= (1 - vector[countOfRows, 0]);
            }
            return vector;
        }

        //  Вычисление градиента:
        //  1. для каждого входа нейронов выходного слоя
        //  2. для каждого входа нейронов скрытого слоя
        public static Matrix Grad(Matrix delta, Matrix vector)
        {
            Matrix result = new Matrix(delta.RowsCount(), vector.ColumnsCount());
            result = delta.Multiply(vector.Transpose());
            return result;
        }

        //
        //  Вычисление дельт для нейронов скрытого слоя
        //
        public static Matrix DeltaHidden(Matrix w2, Matrix dh, Matrix y, Matrix d0)
        {
            return NaiveMultiplication(w2.Transpose().Multiply(DeltaOut(DeltaE(y, d0), FHatch(d0))), FHatch(dh));
        }

        //
        //  Вычисление дельт для нейронов выходного слоя: (y - dO) * f'(dO) = e * f'(dO)
        //
        public static Matrix DeltaOut(Matrix y, Matrix d0)
        {
            return NaiveMultiplication(DeltaE(y, d0), FHatch(d0));
        }

        //
        //  Вычисление e = y - d0
        //
        public static Matrix DeltaE(Matrix y, Matrix d0)
        {
            Matrix result = new Matrix(y.RowsCount(), y.ColumnsCount());
            for (int countOfRows = 0; countOfRows < d0.RowsCount(); countOfRows++)
            {
                result[countOfRows, 0] = y[countOfRows, 0] - d0[countOfRows, 0];
            }
            return result;
        }

        //
        // Вычисление сигмоиды для вектора (вычисляется dh, d0)
        //
        public static Matrix VectorSigmoid(Matrix vector)
        {
            for (int countOfRows = 0; countOfRows < vector.RowsCount(); countOfRows++)
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
            double k = System.Math.Exp(value);
            return k / ((double)1.0 + k);
        }

        //
        // Вычисление весовых коэффициентов
        // n - буква греческого алфавита "эта", dh и d0 берутся из метода VectorSigmoid()
        public static List<List<List<Matrix>>>  Weights(Matrix w1, Matrix w2, Matrix dh, Matrix d0, Matrix x, Matrix y, int iteration, double n, double alfa)
        {
            List<List<List<Matrix>>> all = new List<List<List<Matrix>>>();
            List<List<Matrix>> allDeltaMatrix = new List<List<Matrix>>();
            List<List<Matrix>> allMatrix = new List<List<Matrix>>();
            List<Matrix> deltaW1 = new List<Matrix>();
            List<Matrix> deltaW2 = new List<Matrix>();
            List<Matrix> matrixW1 = new List<Matrix>();
            List<Matrix> matrixW2 = new List<Matrix>();
            

            for(int t = 0; t < iteration; t++)
            {
                if(t == 0)
                {
                    deltaW1.Add(Grad(DeltaHidden(w2, dh, y, d0), x.Transpose()).Multiply(n));
                    deltaW2.Add(Grad(DeltaOut(y, d0), dh.Transpose()).Multiply(n));
                    matrixW1.Add(w1.Add(deltaW1[t]));
                    matrixW2.Add(w2.Add(deltaW2[t]));
                }
                if(t > 0)
                {
                    deltaW1.Add(Grad(DeltaHidden(w2, dh, y, d0), x.Transpose()).Multiply(n).Add(deltaW1[t - 1].Multiply(alfa)));
                    deltaW2.Add(Grad(DeltaOut(y, d0), dh.Transpose()).Multiply(n).Add(deltaW2[t - 1].Multiply(alfa)));
                    matrixW1.Add(matrixW1[t - 1].Add(deltaW1[t]));
                    matrixW2.Add(matrixW2[t - 1].Add(deltaW2[t]));
                }
                allDeltaMatrix.Add(deltaW1);
                allDeltaMatrix.Add(deltaW2);
                allMatrix.Add(matrixW1);
                allMatrix.Add(matrixW2);
                all.Add(allDeltaMatrix);
                all.Add(allMatrix);
            }

            return all;
        }
    }

    class Program
    {
        // Вывод матрицы на конслоль
        public static void PrintMatrix(Matrix matrix, string str)
        {
            for(int countOfRows = 0; countOfRows < matrix.RowsCount(); countOfRows++)
            {
                Console.Write(str);
                for(int countOfColumns = 0; countOfColumns < matrix.ColumnsCount(); countOfColumns++)
                {
                    Console.Write(matrix[countOfRows, countOfColumns] + "   ");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }

        // Заполнение матрицы случайным числом
        public static Matrix CreateMatrix(Matrix matrix)
        {
            Random value = new Random();
            for(int countOfRows = 0; countOfRows < matrix.RowsCount(); countOfRows++)
            {
                for(int countOfColumns = 0; countOfColumns < matrix.ColumnsCount(); countOfColumns++)
                {
                    matrix[countOfRows, countOfColumns] = value.NextDouble();
                }
            }

            return matrix;
        }

        public static void CalculationMatrix()
        {
            int countOfColumns, countOfRows, iteration;
            double lastError, error, alfa, n;
            Console.Write("Введите количество столбцов матрицы: "); 
            countOfColumns = int.Parse(Console.ReadLine());
            Console.Write("Введите количество строк матрицы: "); 
            countOfRows = int.Parse(Console.ReadLine());
            Console.Write("Введите количество итераций, но помните, что при iter > 1 выводиться будут начальные матрицы, конечные матрицы, и конечная ошибка: ");
            //iteration = int.Parse(Console.ReadLine());
            Console.Write("Введите параметр скорости обучения(буква греческого алфавита 'эта'): "); 
            //n = double.Parse(Console.ReadLine());
            Console.Write("Введите момент(альфа): ");  
            //alfa = double.Parse(Console.ReadLine());

            // Создание матриц
            Matrix dh = new Matrix(countOfRows, 1);
            Matrix d0 = new Matrix(countOfRows, 1);
            Matrix x = new Matrix(countOfRows, 1);
            Matrix y = new Matrix(countOfRows, 1);
            Matrix w1 = new Matrix(countOfRows, countOfColumns);
            Matrix w2 = new Matrix(countOfRows, countOfColumns);

            // Генерация случайных значений матриц
            x = CreateMatrix(x);
            y = CreateMatrix(y);
            w1 = CreateMatrix(w1);
            w2 =CreateMatrix(w2);

            // Вывод созданных матриц
            Console.WriteLine("W1:"); PrintMatrix(w1, "\t");
            Console.WriteLine("W2:"); PrintMatrix(w2, "\t");
            Console.WriteLine("X:"); PrintMatrix(x, "\t");
            Console.WriteLine("Y:"); PrintMatrix(y, "\t");

            Console.WriteLine("Матричный расчет выходов скрытого и выходного слоя:");
            Console.WriteLine("\tdh = W1 x X");
            dh = CalculatingMatrix.VectorSigmoid(w1.Multiply(x));
            PrintMatrix(dh, "\t\t");

            d0 = CalculatingMatrix.VectorSigmoid(w2.Multiply(dh));
            Console.WriteLine("\td0 = W2 x dh");
            PrintMatrix(d0, "\t\t");

            Console.WriteLine("\te = y - d0");
            PrintMatrix(CalculatingMatrix.DeltaE(y, d0), "\t\t");
            
            Console.Write("Ошибка равна: ");
            lastError = CalculatingMatrix.Emse(CalculatingMatrix.DeltaE(y, d0));
            Console.WriteLine(lastError);

            Console.WriteLine("Вычисление дельт для нейронов выходного слоя: ");
            PrintMatrix(CalculatingMatrix.DeltaOut(y, d0), "\t\t");

            Console.WriteLine("Вычисление дельт для нейронов скрытого слоя: ");
            PrintMatrix(CalculatingMatrix.DeltaHidden(w2, dh, y, d0), "\t\t");

            Console.WriteLine("Вычисление градиента для каждого входа нейронов выходного и скрытого слоев: ");
            Console.WriteLine("\tW1:");
            PrintMatrix(CalculatingMatrix.Grad(CalculatingMatrix.DeltaHidden(w2, dh, y, d0), x.Transpose()), "\t\t");
            Console.WriteLine("\tW2:");
            PrintMatrix(CalculatingMatrix.Grad(CalculatingMatrix.DeltaOut(y, d0), dh.Transpose()), "\t\t");
            
        }

        static void Main(string[] args)
        {
            CalculationMatrix();
        }
    }
}