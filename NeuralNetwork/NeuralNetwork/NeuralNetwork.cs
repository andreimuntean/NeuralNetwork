using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    /// <summary>
    /// A nonlinear classifier.
    /// </summary>
    public class NeuralNetwork<T>
    {
        private List<T> knownLabels;
        private List<double[,]> weights;
        private Random random = new Random();

        /// <summary>
        /// Initializes a neural network.
        /// </summary>
        /// <param name="examples">The training data. Each example is represented as a list of features.</param>
        /// <param name="labels">A list of labels, one for every example.</param>
        public NeuralNetwork(IEnumerable<IEnumerable<double>> examples, IEnumerable<T> labels)
        {
            // Stores the unique labels.
            knownLabels = labels.Distinct().ToList();
            
            // Trains the network.
            Train(examples, GetIntegerLabel(labels.ToArray()), 2 * examples.First().Count(), 1, 0.1);
        }

        /// <summary>
        /// Initializes a neural network.
        /// </summary>
        /// <param name="examples">The training data. Each example is represented as a list of features.</param>
        /// <param name="labels">A list of labels, one for every example.</param>
        /// <param name="seed">The seed used to generate randomness.</param>
        public NeuralNetwork(IEnumerable<IEnumerable<double>> examples, IEnumerable<T> labels, int seed)
        {
            random = new Random(seed);

            // Stores the unique labels.
            knownLabels = labels.Distinct().ToList();

            // Trains the network.
            Train(examples, GetIntegerLabel(labels.ToArray()), 2 * examples.First().Count(), 1, 0.1);
        }

        /// <summary>
        /// Initializes a neural network.
        /// </summary>
        /// <param name="examples">The training data. Each example is represented as a list of features.</param>
        /// <param name="labels">A list of labels, one for every example.</param>
        /// <param name="hiddenLayerSize">The number of nodes in each hidden layer.</param>
        /// <param name="hiddenLayerCount">The number of hidden layers.</param>
        /// <param name="regularization">The regularization parameter. Used to prevent overfitting.</param>
        public NeuralNetwork(IEnumerable<IEnumerable<double>> examples, IEnumerable<T> labels, int hiddenLayerSize,
            int hiddenLayerCount, double regularization = 0.1)
        {
            // Stores the unique labels.
            knownLabels = labels.Distinct().ToList();

            // Trains the network.
            Train(examples, GetIntegerLabel(labels.ToArray()), hiddenLayerSize, hiddenLayerCount, regularization);
        }

        /// <summary>
        /// Initializes a neural network.
        /// </summary>
        /// <param name="examples">The training data. Each example is represented as a list of features.</param>
        /// <param name="labels">A list of labels, one for every example.</param>
        /// <param name="seed">The seed used to generate randomness.</param>
        /// <param name="hiddenLayerSize">The number of nodes in each hidden layer.</param>
        /// <param name="hiddenLayerCount">The number of hidden layers.</param>
        /// <param name="regularization">The regularization parameter. Used to prevent overfitting.</param>
        public NeuralNetwork(IEnumerable<IEnumerable<double>> examples, IEnumerable<T> labels, int seed,
            int hiddenLayerSize, int hiddenLayerCount, double regularization = 0.1)
        {
            random = new Random(seed);

            // Stores the unique labels.
            knownLabels = labels.Distinct().ToList();

            // Trains the network.
            Train(examples, GetIntegerLabel(labels.ToArray()), hiddenLayerSize, hiddenLayerCount, regularization);
        }

        private IEnumerable<int> GetIntegerLabel(params T[] labels)
        {
            return labels.Select(label => knownLabels.IndexOf(label));
        }

        private IEnumerable<T> GetOriginalLabel(params int[] labels)
        {
            return labels.Select(label => knownLabels[label]);
        }

        // Generates weights for a layer with the specified number of input nodes and the specified number of output nodes.
        private double[,] GenerateWeights(int inputNodeCount, int outputNodeCount)
        {
            // Restricts the weights to +-epsilon.
            const double epsilon = 0.12;

            // Increments the inputNodeCount to take the bias node into consideration.
            var weights = new double[outputNodeCount, inputNodeCount + 1];

            for (int outputNodeIndex = 0; outputNodeIndex < weights.GetLength(0); ++outputNodeIndex)
            {
                for (int inputNodeIndex = 0; inputNodeIndex < weights.GetLength(1); ++inputNodeIndex)
                {
                    weights[outputNodeIndex, inputNodeIndex] = epsilon * (2 * random.NextDouble() - 1);
                }
            }

            return weights;
        }

        // Generates weights for a neural network with the specified configuration.
        private List<double[,]> GenerateWeights(int inputLayerSize, int hiddenLayerSize, int outputLayerSize,
            int hiddenLayerCount)
        {
            var weights = new List<double[,]>();

            // Determines whether hidden layers exist.
            if (hiddenLayerCount == 0)
            {
                // Maps the input layer directly to the output layer.
                weights.Add(GenerateWeights(inputLayerSize, outputLayerSize));
            }
            else
            {
                // Maps the input layer to the first hidden layer.
                weights.Add(GenerateWeights(inputLayerSize, hiddenLayerSize));

                while (--hiddenLayerCount > 0)
                {
                    // Maps this hidden layer to the next hidden layer.
                    weights.Add(GenerateWeights(hiddenLayerSize, hiddenLayerSize));
                }

                // Maps the last hidden layer to the output layer.
                weights.Add(GenerateWeights(hiddenLayerSize, outputLayerSize));
            }

            return weights;
        }
        
        private ForwardPropagationResult PropagateForwards(IEnumerable<double> data)
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

        private void PropagateBackwards(List<ForwardPropagationResult> forwardPropagationResults, IEnumerable<int> labels, double learningRate,
            double regularization)
        {
            var batchActivations = new List<List<List<double>>>();
            var batchErrors = new List<List<List<double>>>();

            for (int index = 0; index < forwardPropagationResults.Count; ++index)
            {
                var activations = forwardPropagationResults[index].Activations;
                var errors = GetErrors(activations, forwardPropagationResults[index].Sums, labels.ElementAt(index));

                batchActivations.Add(activations);
                batchErrors.Add(errors);
            }

            // Averages and regularizes the gradients.
            var regularizedGradients = GetGradients(batchActivations, batchErrors, regularization);

            UpdateWeights(regularizedGradients, learningRate);
        }

        private void UpdateWeights(List<double[,]> gradient, double learningRate)
        {
            for (int layerIndex = 0; layerIndex < weights.Count; ++layerIndex)
            {
                int outputNodeCount = weights[layerIndex].GetLength(0);
                int inputNodeCount = weights[layerIndex].GetLength(1);

                for (int outputNodeIndex = 0; outputNodeIndex < outputNodeCount; ++outputNodeIndex)
                {
                    for (int inputNodeIndex = 0; inputNodeIndex < inputNodeCount; ++inputNodeIndex)
                    {
                        var update = learningRate * gradient[layerIndex][outputNodeIndex, inputNodeIndex];

                        weights[layerIndex][outputNodeIndex, inputNodeIndex] -= update;
                    }
                }
            }
        }

        private double GetCost(List<double> outputLayerActivations, double label)
        {
            if (outputLayerActivations.Count != knownLabels.Count)
            {
                throw new ArgumentException("Invalid argument: outputLayerActivations.");
            }

            double cost = 0;

            for (int index = 0; index < knownLabels.Count; ++index)
            {
                var activation = outputLayerActivations[index];

                if (label == index)
                {
                    cost -= Math.Log(activation);
                }
                else
                {
                    cost -= Math.Log(1 - activation);
                }
            }

            return cost;
        }

        private double GetRegularizedCost(List<double> costs, double regularization)
        {
            double sum = 0;

            foreach (var currentWeights in weights)
            {
                for (int outputNodeIndex = 0; outputNodeIndex < currentWeights.GetLength(0); ++outputNodeIndex)
                {
                    for (int inputNodeIndex = 1; inputNodeIndex < currentWeights.GetLength(1); ++inputNodeIndex)
                    {
                        sum += Math.Pow(currentWeights[outputNodeIndex, inputNodeIndex], 2);
                    }
                }
            }

            return (costs.Sum() + regularization / 2 * sum) / knownLabels.Count;
        }

        private List<List<double>> GetErrors(List<List<double>> activations, List<List<double>> sums, int label)
        {
            var errors = new List<List<double>>();

            // Initializes the error list.
            foreach (var weights in weights)
            {
                errors.Add(new List<double>());
            }

            // Calculates the errors for the output layer.
            foreach (var activation in activations[activations.Count - 1])
            {
                errors[errors.Count - 1].Add(activation);
            }

            errors[errors.Count - 1][label] -= 1;

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

                    errors[index].Add(error + MathHelpers.SigmoidGradient(sum));
                }
            }

            return errors;
        }

        private List<double[,]> GetGradients(List<List<List<double>>> batchActivations, List<List<List<double>>> batchErrors, double regularization)
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

                        if (inputNodeIndex == 0)
                        {
                            layerGradients[outputNodeIndex, inputNodeIndex] /= exampleCount;
                        }
                        else
                        {
                            layerGradients[outputNodeIndex, inputNodeIndex] *= regularization / exampleCount * weights[layerIndex][outputNodeIndex, inputNodeIndex];
                        }
                    }
                }

                gradients.Add(layerGradients);
            }

            return gradients;
        }

        private void Train(IEnumerable<IEnumerable<double>> examples, IEnumerable<int> labels, int hiddenLayerSize,
            int hiddenLayerCount, double regularization)
        {
            const double maximumIterations = 100000;
            const double epsilon = 0;

            // Determines the speed at which the weights are updated.
            double learningRate = 1;

            // Keeps track of previous costs. Used to modify the learning rate.
            var costHistory = new List<double>();

            // Determines the sizes of the input and output layers.
            var inputLayerSize = examples.First().Count();
            var outputLayerSize = knownLabels.Count();

            // Randomly initializes the weights of the network.
            weights = GenerateWeights(inputLayerSize, hiddenLayerSize, outputLayerSize, hiddenLayerCount);

            for (int iteration = 0; iteration < maximumIterations; ++iteration)
            {
                var forwardPropagationResults = new List<ForwardPropagationResult>();
                var costs = new List<double>();

                for (int index = 0; index < examples.Count(); ++index)
                {
                    var exampleResult = PropagateForwards(examples.ElementAt(index));
                    var exampleCost = GetCost(exampleResult.Activations.Last(), labels.ElementAt(index));

                    forwardPropagationResults.Add(exampleResult);
                    costs.Add(exampleCost);
                }

                var cost = GetRegularizedCost(costs, regularization);

                if (costHistory.Count > 0 && cost > costHistory.Last())
                {
                    // Decreases the learning rate to prevent divergence.
                    learningRate /= 3;
                }
                else if (costHistory.Count > 0 && costHistory.Last() - cost < epsilon)
                {
                    break;
                }

                if (iteration % 10000 == 0)
                {
                    Console.WriteLine("Cost: " + cost + "\t\tLearning rate: " + learningRate);
                }

                costHistory.Add(cost);
                PropagateBackwards(forwardPropagationResults, labels, learningRate, regularization);
            }
        }

        /// <summary>
        /// Predicts a class for the specified data.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <returns>The predicted class.</returns>
        public T Predict(IEnumerable<double> data)
        {
            var label = PropagateForwards(data).Prediction;

            return GetOriginalLabel(label).First();
        }
    }
}
