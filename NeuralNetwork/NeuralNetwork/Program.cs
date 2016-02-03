using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    class Program
    {
        static void ColorTest()
        {
            var data = new List<List<double>>
            {
                new List<double> { 255, 255, 255 },
                new List<double> { 255, 0, 0 },
                new List<double> { 0, 255, 0 },
                new List<double> { 0, 0, 255 },
                new List<double> { 255, 255, 0 },
                new List<double> { 255, 0, 255 },
                new List<double> { 0, 255, 255 },
                new List<double> { 120, 120, 120 },
                new List<double> { 0, 0, 0 }
            };

            string[] labels =
            {
                "White",
                "Red",
                "Green",
                "Blue",
                "Yellow",
                "Purple",
                "Cyan",
                "Gray",
                "Black"
            };

            var tests = new List<List<double>>
            {
                new List<double> { 235, 250, 246 },
                new List<double> { 235, 15, 92 },
                new List<double> { 93, 255, 94 },
                new List<double> { 35, 64, 249 },
                new List<double> { 249, 250, 3 },
                new List<double> { 245, 0, 245 },
                new List<double> { 0, 255, 255 },
                new List<double> { 123, 111, 130 },
                new List<double> { 8, 8, 8 },
                new List<double> { 25, 15, 38 }
            };

            var neuralNetwork = new NeuralNetwork<string>(data, labels, 9, 2, 0.01);

            foreach (var test in tests)
            {
                Console.WriteLine(string.Join(" ", test) + " is " + neuralNetwork.Predict(test));
            }

            Console.Read();
        }

        static void XORTest()
        {
            var data = new List<List<double>>
            {
                new List<double> { 0, 0 },
                new List<double> { 0, 1 },
                new List<double> { 1, 0 },
                new List<double> { 1, 1 }
            };

            int[] labels =
            {
                0,
                1,
                1,
                0
            };

            var tests = new List<List<double>>
            {
                new List<double> { 0, 0 },
                new List<double> { 0, 1 },
                new List<double> { 1, 0 },
                new List<double> { 1, 1 }
            };

            var neuralNetwork = new NeuralNetwork<int>(data, labels);

            foreach (var test in tests)
            {
                Console.WriteLine(string.Join(" ", test) + " is " + neuralNetwork.Predict(test));
            }

            Console.Read();
        }

        static void Main(string[] args)
        {
            ColorTest();
        }
    }
}
