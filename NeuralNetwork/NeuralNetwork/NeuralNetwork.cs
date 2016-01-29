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
        
        /// <summary>
        /// Initializes a neural network.
        /// </summary>
        /// <param name="data">The training data. Each example is represented as a list of features.</param>
        /// <param name="labels">A list of labels, one for every example.</param>
        /// <param name="hiddenLayerSize">The number of nodes in each hidden layer.</param>
        /// <param name="hiddenLayerCount">The number of hidden layers.</param>
        /// <param name="regularization">The regularization parameter. It is used to avoid overfitting.</param>
        public NeuralNetwork(IEnumerable<IEnumerable<double>> data, IEnumerable<T> labels, int hiddenLayerSize = 1,
            int hiddenLayerCount = 1, double regularization = 1.0)
        {
            // Stores the unique labels.
            knownLabels = labels.Distinct().ToList();

            // Trains the network.
            Train(data, GetIntegerLabel(labels.ToArray()), hiddenLayerSize, hiddenLayerCount, regularization);
        }

        private IEnumerable<int> GetIntegerLabel(params T[] labels)
        {
            return labels.Select(label => knownLabels.IndexOf(label));
        }

        private IEnumerable<T> GetOriginalLabel(params int[] labels)
        {
            return labels.Select(label => knownLabels[label]);
        }

        private static IEnumerable<double> GenerateWeights(int inputNodes, int outputNodes)
        {
            return null;
        }

        private void Train(IEnumerable<IEnumerable<double>> data, IEnumerable<int> labels, int hiddenLayerSize,
            int hiddenLayerCount, double regularization)
        {
        
        }

        /// <summary>
        /// Predicts a class for the specified data.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <returns>The predicted class.</returns>
        public T Predict(IEnumerable<double> data)
        {
            return GetOriginalLabel(0).First();
        }
    }
}
