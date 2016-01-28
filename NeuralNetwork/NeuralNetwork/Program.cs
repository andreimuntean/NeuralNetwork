using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
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
                new List<double> { 235, 15, 92 },
                new List<double> { 35, 64, 249 },
                new List<double> { 25, 15, 48 },
                new List<double> { 123, 100, 130 }
            };

            var neuralNetwork = new NeuralNetwork<string>(data, labels);

            foreach (var test in tests)
            {
                Console.WriteLine(string.Join(" ", test) + " is " + neuralNetwork.Predict(test));
            }

            Console.Read();
        }
    }
}
