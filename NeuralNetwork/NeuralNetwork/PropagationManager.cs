using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    static class PropagationManager
    {
        public static ForwardPropagationResult PropagateForwards(List<double[,]> weights, IEnumerable<double> data)
        {
            var sums = new List<List<double>>();
            var activations = new List<List<double>>();

            // Stores the activation of the input layer.
            activations.Add(data.ToList());

            // Iterates through the weights of every layer.
            foreach (var currentWeights in weights)
            {
                var previousActivation = activations.Last();

                sums.Add(new List<double>());
                activations.Add(new List<double>());

                // Prepends the bias term.
                previousActivation.Insert(0, 1);

                // Sums the input nodes for every output node.
                for (int outputNodeIndex = 0; outputNodeIndex < currentWeights.GetLength(0); ++outputNodeIndex)
                {
                    double sum = 0;

                    for (int inputNodeIndex = 0; inputNodeIndex < currentWeights.GetLength(1); ++inputNodeIndex)
                    {
                        sum += previousActivation[inputNodeIndex] * currentWeights[outputNodeIndex, inputNodeIndex];
                    }

                    // Stores the sum and activation for this output node.
                    sums.Last().Add(sum);
                    activations.Last().Add(MathHelpers.Sigmoid(sum));
                }
            }

            return new ForwardPropagationResult(sums, activations);
        }

        public static List<double[,]> PropagateBackwards(List<double[,]> weights, List<ForwardPropagationResult> results, IEnumerable<int> labels,
            double learningRate, double regularization)
        {
            var batchActivations = new List<List<List<double>>>();
            var batchErrors = new List<List<List<double>>>();

            for (int index = 0; index < results.Count; ++index)
            {
                var activations = results[index].Activations;
                var errors = GetErrors(weights, activations, results[index].Sums, labels.ElementAt(index));

                batchActivations.Add(activations);
                batchErrors.Add(errors);
            }

            // Averages and regularizes the gradients.
            var regularizedGradients = GetGradients(weights, batchActivations, batchErrors, regularization);

            return GetUpdatedWeights(weights, regularizedGradients, learningRate);
        }

        private static List<double[,]> GetUpdatedWeights(List<double[,]> weights, List<double[,]> gradient, double learningRate)
        {
            var updatedWeights = new List<double[,]>();

            for (int layerIndex = 0; layerIndex < weights.Count; ++layerIndex)
            {
                int outputNodeCount = weights[layerIndex].GetLength(0);
                int inputNodeCount = weights[layerIndex].GetLength(1);

                updatedWeights.Add(new double[outputNodeCount, inputNodeCount]);

                for (int outputNodeIndex = 0; outputNodeIndex < outputNodeCount; ++outputNodeIndex)
                {
                    for (int inputNodeIndex = 0; inputNodeIndex < inputNodeCount; ++inputNodeIndex)
                    {
                        var originalWeight = weights[layerIndex][outputNodeIndex, inputNodeIndex];
                        var gradientUpdate = learningRate * gradient[layerIndex][outputNodeIndex, inputNodeIndex];

                        updatedWeights.Last()[outputNodeIndex, inputNodeIndex] = originalWeight - gradientUpdate;
                    }
                }
            }

            return updatedWeights;
        }

        private static List<List<double>> GetErrors(List<double[,]> weights, List<List<double>> activations, List<List<double>> sums,
            int label)
        {
            var errors = new List<List<double>>();

            // Initializes the error list.
            for (int index = 0; index < weights.Count; ++index)
            {
                errors.Add(new List<double>());
            }

            // Calculates the errors for the output layer.
            foreach (var activation in activations[activations.Count - 1])
            {
                errors.Last().Add(activation);
            }

            errors.Last()[label] -= 1;

            // Calculates the errors for the hidden layers.
            for (int index = errors.Count - 2; index >= 0; --index)
            {
                int outputNodeCount = weights[index + 1].GetLength(0);
                int inputNodeCount = weights[index + 1].GetLength(1);

                for (int inputNodeIndex = 1; inputNodeIndex < inputNodeCount; ++inputNodeIndex)
                {
                    double error = 0;

                    for (int outputNodeIndex = 0; outputNodeIndex < outputNodeCount; ++outputNodeIndex)
                    {
                        error += weights[index + 1][outputNodeIndex, inputNodeIndex] * errors[index + 1][outputNodeIndex];
                    }

                    // The tertiary operator acts as if the bias term has been prepended to the sums.
                    var sum = sums[index][inputNodeIndex - 1];

                    errors[index].Add(error * MathHelpers.SigmoidGradient(sum));
                }
            }

            return errors;
        }

        private static List<double[,]> GetGradients(List<double[,]> weights, List<List<List<double>>> batchActivations, List<List<List<double>>> batchErrors,
            double regularization)
        {
            var gradients = new List<double[,]>();
            int exampleCount = batchActivations.Count;

            for (int layerIndex = 0; layerIndex < weights.Count; ++layerIndex)
            {
                int outputNodeCount = weights[layerIndex].GetLength(0);
                int inputNodeCount = weights[layerIndex].GetLength(1);
                var layerGradients = new double[outputNodeCount, inputNodeCount];

                for (int outputNodeIndex = 0; outputNodeIndex < outputNodeCount; ++outputNodeIndex)
                {
                    for (int inputNodeIndex = 0; inputNodeIndex < inputNodeCount; ++inputNodeIndex)
                    {
                        for (int exampleIndex = 0; exampleIndex < exampleCount; ++exampleIndex)
                        {
                            var error = batchErrors[exampleIndex][layerIndex][outputNodeIndex];
                            var activation = batchActivations[exampleIndex][layerIndex][inputNodeIndex];

                            layerGradients[outputNodeIndex, inputNodeIndex] += error * activation;
                        }

                        layerGradients[outputNodeIndex, inputNodeIndex] /= exampleCount;

                        if (inputNodeIndex > 0)
                        {
                            layerGradients[outputNodeIndex, inputNodeIndex] += regularization / exampleCount * weights[layerIndex][outputNodeIndex, inputNodeIndex];
                        }
                    }
                }

                gradients.Add(layerGradients);
            }

            return gradients;
        }
    }
}
