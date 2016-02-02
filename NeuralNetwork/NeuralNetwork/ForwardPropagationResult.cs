using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    struct ForwardPropagationResult
    {
        public List<List<double>> Sums { get; }

        public List<List<double>> Activations { get; }

        public int Prediction { get; }

        public ForwardPropagationResult(List<List<double>> sums, List<List<double>> activations)
        {
            Sums = sums;
            Activations = activations;

            // The predicted label is the index of the node with the highest activation.
            Prediction = Activations.Last().IndexOfHighestValue();
        }
    }
}
