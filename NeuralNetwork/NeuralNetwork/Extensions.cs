using System.Collections.Generic;

namespace NeuralNetwork
{
    static class Extensions
    {
        public static int IndexOfHighestValue(this List<double> values)
        {
            int highestIndex = values.Count - 1;
            double highestValue = values[highestIndex];

            for (int index = 0; index < values.Count - 1; ++index)
            {
                if (highestValue.CompareTo(values[index]) < 0)
                {
                    highestIndex = index;
                    highestValue = values[index];
                }
            }

            return highestIndex;
        }
    }
}
