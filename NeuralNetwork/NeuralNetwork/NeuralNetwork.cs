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
            Train(examples, GetIntegerLabel(labels.ToArray()), 2 * examples.First().Count(), 1, 1.0);
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
            Train(examples, GetIntegerLabel(labels.ToArray()), 2 * examples.First().Count(), 1, 1.0);
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
            int hiddenLayerCount, double regularization = 1.0)
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
            int hiddenLayerSize, int hiddenLayerCount, double regularization = 1.0)
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

        // Gets the activations of every layer.
        private ForwardPropagationResult PropagateForwards(IEnumerable<double> data)
        {
            var sums = new List<List<double>>();
            var activations = new List<List<double>>();

            // Stores the activation of the input layer.
            activations.Add(data.ToList());

            // Iterates through the weights of every layer.
            foreach (var weights in weights)
            {
                var previousActivation = activations.Last();

                sums.Add(new List<double>());
                activations.Add(new List<double>());

                // Prepends the bias term.
                previousActivation.Insert(0, 1);

                // Sums the input nodes for every output node.
                for (int outputNodeIndex = 0; outputNodeIndex < weights.GetLength(0); ++outputNodeIndex)
                {
                    double sum = 0;

                    for (int inputNodeIndex = 0; inputNodeIndex < weights.GetLength(1); ++inputNodeIndex)
                    {
                        sum += previousActivation[inputNodeIndex] * weights[outputNodeIndex, inputNodeIndex];
                    }

                    // Stores the sum and activation for this output node.
                    sums.Last().Add(sum);
                    activations.Last().Add(MathHelpers.Sigmoid(sum));
                }
            }

            return new ForwardPropagationResult(sums, activations);
        }

        private BackwardPropagationResult PropagateBackwards(ForwardPropagationResult forwardPropagationResult, int label)
        {
            var errors = new List<List<double>>();
            var activations = forwardPropagationResult.Activations;
            var sums = forwardPropagationResult.Sums;

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
                // TO-DO.
            }

            return new BackwardPropagationResult(errors);
        }

        private void Train(IEnumerable<IEnumerable<double>> examples, IEnumerable<int> labels, int hiddenLayerSize,
            int hiddenLayerCount, double regularization)
        {
            var inputLayerSize = examples.First().Count();
            var outputLayerSize = labels.Count();

            weights = GenerateWeights(inputLayerSize, hiddenLayerSize, outputLayerSize, hiddenLayerCount);

            for (int index = 0; index < examples.Count(); ++index)
            {
                var forwardPropagationResult = PropagateForwards(examples.ElementAt(index));
                var backwardPropagationResult = PropagateBackwards(forwardPropagationResult, labels.ElementAt(index));

                // TO-DO.
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
