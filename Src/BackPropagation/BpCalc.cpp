#include "BackPropagator.h"
#include "DebugLib.h"

#include <cmath>

using namespace std;

/**
  Calculate the delta value of each node in the last(output) layer.
  Delta = (Desired - Actual) * f'(Actual), where f(x) is the activation function and
  f'(x) is the derivative of activation function.

  @param[in]  DesiredOutput  A matrix representing the desired output values.

**/
matrix
BackPropagator::CalculateLastLayerDelta (
  matrix  DesiredOutput
  )
{
  matrix        DesiredGap;
  unsigned int  LastLayerIndex;

  LastLayerIndex = (unsigned int)(Network.GetLayout().size() - 1);

  DesiredGap = Substract (
                 DesiredOutput,
                 Network.GetActivationByLayer (LastLayerIndex)
                 );

  return HadamardProduct (
           DesiredGap,
           Network.GetDerivativeActivationByLayer (LastLayerIndex)
           );
}

/**
  Calculate the delta value of each node in a specific middle layer.
  Delta = (Weight^T * NextLayerDelta) * f'(CurrentLayerActivationValue), where f(x) is the activation function and
  f'(x) is the derivative of activation function.

  @param[in]  Layer

**/
matrix
BackPropagator::CalculateMidLayerDelta (
  unsigned int  Layer
  )
{
  matrix  WeightedError;

  if (Layer > (unsigned int)(Network.GetLayout().size() - 2)) {
    DEBUG_LOG ("Layer " << Layer << " is not a middle layer.");
    throw runtime_error ("Layer passed into CalculateMidLayerDelta() is out of range.");
  }

  WeightedError = multiply (
                    transpose (Network.GetWeightByLayer(Layer)),
                    NodeDelta[Layer + 1]
                    );

  return HadamardProduct (
           WeightedError,
           Network.GetDerivativeActivationByLayer (Layer)
           );
}

/**
  Calculate the delta value of each node in all layers except the first(input) layer.

  @param[in]  DesiredOutput  A matrix representing the desired output values.

**/
void BackPropagator::NodeDeltaCalculation (
  const matrix  &DesiredOutput
  )
{
  matrix        NodeDeltaOfLayer;
  unsigned int  LastLayerIndex = (unsigned int)(Network.GetLayout().size() - 1);

  NodeDeltaOfLayer = CalculateLastLayerDelta (DesiredOutput);
  NodeDelta[LastLayerIndex] = NodeDeltaOfLayer;

  //
  // Calculate delta for all nodes in all layer except last layer.
  // Note: It's unnecessary to calculate the delta value of the first(input) layer(LayerIdx = 0).
  //
  for (unsigned int LayerIdx = LastLayerIndex - 1;
       LayerIdx > 0;
       LayerIdx--) {
    NodeDeltaOfLayer = CalculateMidLayerDelta (LayerIdx);
    NodeDelta[LayerIdx] = NodeDeltaOfLayer;
  }

  // DEBUG_START()
  // PrintNodeDelta ();
  // DEBUG_END()
}

/**
  Calculate the delta value of each node in all layers except the first(input) layer.

  @param[in]  DesiredOutput  A matrix representing the desired output values.

**/
void
BackPropagator::DeltaWeightsCalculation (
  double  LearningRate
  )
{
  unsigned int  WeightsLayerCount = ((unsigned int)Network.GetLayout().size() - 1);

  DeltaWeights.clear();

  for (unsigned int LayerIdx = 0; LayerIdx < WeightsLayerCount; LayerIdx++) {
    matrix  CurrentLayerActivation_T = transpose (Network.GetActivationByLayer (LayerIdx));
    matrix  NextLayerDelta           = NodeDelta[LayerIdx + 1];

    matrix  Gradient = multiply (NextLayerDelta, CurrentLayerActivation_T);
  
    matrix  DeltaWeight = multiplyBy (Gradient, LearningRate);

    DeltaWeights.push_back (DeltaWeight);
  }
}

/**
  Update the weights of the network by applying the calculated delta weights.

**/
void
BackPropagator::UpdateWeights (
  void
  )
{
  if (DeltaWeights.empty () ||
      (DeltaWeights.size () != Network.GetLayout().size() - 1)) {
    DEBUG_LOG ("DeltaWeights Size = " << DeltaWeights.size() << " , Network Layout size = " << Network.GetLayout().size());
    DEBUG_LOG ("Delta Weights are not ready, Failed to update");
  }

  Network.UpdateWeight (DeltaWeights);
}

/**
  Perform the backward pass of back propagation algorithm, which includes following steps:
  1. Calculate node deltas
  2. Calculate delta weights
  3. Update the network weights

  @param[in]  DesiredOutput  A matrix representing the desired output values.
  @param[in]  LearningRate   A double representing the learning rate for weight updates.

**/
void
BackPropagator::BackwardPass (
  const matrix &DesiredOutput,
  const double LearningRate
  )
{
  NodeDeltaCalculation (DesiredOutput);

  DeltaWeightsCalculation (LearningRate);

  UpdateWeights ();
}

double
InternalSquare (
  double  x
  )
{
  return pow(x, 2.0);
}

/**
  Calculate the loss value of the network based on the desired output.
  Here we use Mean Square Error(MSE) as the loss function.
  The formula is: Loss = Sum( (Desired - Actual)^2 ) / N, where N is the number of output nodes.

  @param[in]  DesiredOutput  A matrix representing the desired output values.

  @return A double representing the calculated loss value.

**/
double
BackPropagator::LossMeanSquareError (
  const matrix  &DesiredOutput
  )
{
  matrix                   DesiredGap;
  function<double(double)> Square = InternalSquare;

  DesiredGap = Substract (
                 DesiredOutput,
                 Network.GetActivationByLayer (Network.GetLayout().size() - 1)
                 );

  return DesiredGap.ApplyElementWise (Square).Sum() / DesiredOutput.getrow();
}
