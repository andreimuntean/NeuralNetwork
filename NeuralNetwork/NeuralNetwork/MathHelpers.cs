using System;

namespace NeuralNetwork
{
    static class MathHelpers
    {
        /// <summary>
        /// Computes the logistic sigmoid function.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>A value between 0 and 1.</returns>
        public static double Sigmoid(double value)
        {
            return 1 / (1 + Math.Exp(-value));
        }

        /// <summary>
        /// Computes the gradient of the logistic sigmoid function.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>A value between 0 and 1.</returns>
        public static double SigmoidGradient(double value)
        {
            return Sigmoid(value) * (1 - Sigmoid(value));
        }
    }
}
