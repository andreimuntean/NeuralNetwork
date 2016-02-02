using System.Collections.Generic;

namespace NeuralNetwork
{
    struct BackwardPropagationResult
    {
        public List<List<double>> Errors { get; }

        public BackwardPropagationResult(List<List<double>> errors)
        {
            Errors = errors;
        }
    }
}
