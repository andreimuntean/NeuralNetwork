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
                new List<double> { 255, 245, 245 },
                new List<double> { 245, 242, 250 },

                new List<double> { 255, 0, 0 },
                new List<double> { 240, 4, 3 },
                new List<double> { 250, 10, 11 },

                new List<double> { 0, 255, 0 },
                new List<double> { 12, 243, 10 },
                new List<double> { 4, 250, 3 },

                new List<double> { 0, 0, 255 },
                new List<double> { 12, 10, 235 },
                new List<double> { 8, 11, 240 },

                new List<double> { 255, 255, 0 },
                new List<double> { 254, 245, 10 },
                new List<double> { 248, 249, 7 },

                new List<double> { 255, 0, 255 },
                new List<double> { 235, 10, 240 },
                new List<double> { 241, 8, 233 },
                new List<double> { 200, 10, 240 },
                new List<double> { 160, 4, 200 },
                new List<double> { 153, 7, 160 },

                new List<double> { 0, 255, 255 },
                new List<double> { 15, 240, 241 },
                new List<double> { 7, 231, 226 },

                new List<double> { 103, 61, 35 },
                new List<double> { 145, 87, 49 },
                new List<double> { 101, 58, 31 },

                new List<double> { 123, 120, 121 },
                new List<double> { 131, 131, 132 },
                new List<double> { 120, 120, 120 },

                new List<double> { 11, 6, 13 },
                new List<double> { 3, 4, 2 },
                new List<double> { 0, 0, 0 }
            };

            string[] labels =
            {
                "White",
                "White",
                "White",

                "Red",
                "Red",
                "Red",

                "Green",
                "Green",
                "Green",

                "Blue",
                "Blue",
                "Blue",

                "Yellow",
                "Yellow",
                "Yellow",

                "Purple",
                "Purple",
                "Purple",
                "Purple",
                "Purple",
                "Purple",

                "Cyan",
                "Cyan",
                "Cyan",

                "Brown",
                "Brown",
                "Brown",

                "Gray",
                "Gray",
                "Gray",

                "Black",
                "Black",
                "Black"
            };

            var tests = new List<List<double>>
            {
                new List<double> { 250, 250, 250 },
                new List<double> { 235, 250, 246 },
                new List<double> { 255, 26, 26 },
                new List<double> { 235, 15, 92 },
                new List<double> { 68, 184, 31 },
                new List<double> { 93, 255, 94 },
                new List<double> { 35, 64, 249 },
                new List<double> { 15, 29, 202 },
                new List<double> { 249, 250, 3 },
                new List<double> { 255, 251, 40 },
                new List<double> { 245, 0, 245 },
                new List<double> { 250, 20, 250 },
                new List<double> { 0, 255, 255 },
                new List<double> { 45, 215, 223 },
                new List<double> { 123, 111, 130 },
                new List<double> { 121, 121, 121 },
                new List<double> { 8, 8, 8 },
                new List<double> { 25, 15, 38 }
            };

            var neuralNetwork = new NeuralNetwork<string>(data, labels, 500, 1, 4);

            foreach (var test in tests)
            {
                Console.WriteLine(string.Join(" ", test) + " is " + neuralNetwork.Predict(test));
            }

            string input;

            while ((input = Console.ReadLine()) != "")
            {
                var color = input.Split(' ');
                double red = double.Parse(color[0]);
                double green = double.Parse(color[1]);
                double blue = double.Parse(color[2]);
                var test = new List<double> { red, green, blue };

                Console.WriteLine(string.Join(" ", test) + " is " + neuralNetwork.Predict(test));
            }
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
