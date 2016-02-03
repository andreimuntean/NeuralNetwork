using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    static class CostManager
    {
        public static double GetCost(ForwardPropagationResult forwardPropagationResult, double label)
        {
            var outputLayerActivations = forwardPropagationResult.Activations.Last(); 
            double cost = 0;

            for (int index = 0; index < outputLayerActivations.Count; ++index)
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

        public static double GetBatchCost(List<double[,]> weights, List<double> costs, double regularization)
        {
            var examplesCount = costs.Count;
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

            return (costs.Sum() + regularization / 2 * sum) / examplesCount;
        }
    }
}
