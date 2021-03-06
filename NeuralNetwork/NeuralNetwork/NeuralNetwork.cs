﻿using System;
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
            Train(examples, GetIntegerLabel(labels.ToArray()), 3 * examples.First().Count(), 1, 0);
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
            Train(examples, GetIntegerLabel(labels.ToArray()), 3 * examples.First().Count(), 1, 0);
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
            int hiddenLayerCount, double regularization)
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
            int hiddenLayerSize, int hiddenLayerCount, double regularization)
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

        private void Train(IEnumerable<IEnumerable<double>> examples, IEnumerable<int> labels, int hiddenLayerSize,
            int hiddenLayerCount, double regularization)
        {
            const int maximumIterations = 10000;
            const double minimumCostDifference = 1e-100;
            const double minimumCost = 1e-4;
            const double learningRateAcceleration = 1.001;
            const double learningRateDeceleration = 0.3;

            // Determines the speed at which the weights are updated.
            double learningRate = 1;

            // Determines the sizes of the input and output layers.
            var inputLayerSize = examples.First().Count();
            var outputLayerSize = knownLabels.Count();

            // Randomly initializes the weights of the network.
            weights = GenerateWeights(inputLayerSize, hiddenLayerSize, outputLayerSize, hiddenLayerCount);

            var newWeights = weights;
            List<ForwardPropagationResult> previousResults = null;
            var previousCost = double.MaxValue;

            for (int iteration = 1; iteration <= maximumIterations; ++iteration)
            {
                var results = new List<ForwardPropagationResult>();
                var exampleCosts = new List<double>();

                for (int index = 0; index < examples.Count(); ++index)
                {
                    var exampleResult = PropagationManager.PropagateForwards(newWeights, examples.ElementAt(index));
                    var exampleCost = CostManager.GetCost(exampleResult, labels.ElementAt(index));

                    results.Add(exampleResult);
                    exampleCosts.Add(exampleCost);
                }

                var cost = CostManager.GetBatchCost(newWeights, exampleCosts, regularization);

                // Determines whether the cost is increasing.
                if (previousCost - cost < 0)
                {
                    // Reduces the learning rate and iterates again.
                    learningRate *= learningRateDeceleration;
                    newWeights = PropagationManager.PropagateBackwards(weights, previousResults, labels, learningRate, regularization);

                    continue;
                }

                // Determines whether the training should stop.
                if (previousCost - cost < minimumCostDifference || cost < minimumCost)
                {
                    // Stops the training.
                    break;
                }

                // Increases the learning rate and updates the weights.
                learningRate *= learningRateAcceleration;
                weights = newWeights;

                // Stores data for the next iteration.
                newWeights = PropagationManager.PropagateBackwards(weights, results, labels, learningRate, regularization);
                previousResults = results;
                previousCost = cost;

                if ((iteration - 1) % 100 == 0)
                {
                    Console.Write("Iteration: {0} \tCost: {1}\tLearning rate: {2}\n", iteration, cost, learningRate);
                }
            }
        }

        /// <summary>
        /// Predicts a class for the specified data.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <returns>The predicted class.</returns>
        public T Predict(IEnumerable<double> data)
        {
            var result = PropagationManager.PropagateForwards(weights, data);
            var label = result.Prediction;

            return GetOriginalLabel(label).First();
        }
    }
}
