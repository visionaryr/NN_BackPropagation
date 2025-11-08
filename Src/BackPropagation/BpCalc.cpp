#include "BackPropagator.h"
#include "DebugLib.h"

using namespace std;

/**
  Calculate the delta value of each node in the last(output) layer.
  Delta = (Desired - Actual) * f'(Actual), where f(x) is the activation function and
  f'(x) is the derivative of activation function.

  @param[in]  DesiredOutput  A matrix representing the desired output values.

**/
void
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

  NodeDelta[LastLayerIndex] = HadamardProduct (
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
void
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

  NodeDelta[Layer] = HadamardProduct (
                       WeightedError,
                       Network.GetDerivativeActivationByLayer (Layer)
                       );
}

/**
  Calculate the delta value of each node in all layers except the first(input) layer.

  @param[in]  DesiredOutput  A matrix representing the desired output values.

**/
void BackPropagator::NodeDeltaCalculation (
  matrix  DesiredOutput
  )
{
  CalculateLastLayerDelta (DesiredOutput);

  //
  // Calculate delta for all nodes in all layer except last layer.
  // Note: It's unnecessary to calculate the delta value of the first(input) layer(LayerIdx = 0).
  //
  for (unsigned int LayerIdx = (unsigned int)(Network.GetLayout().size() - 2);
       LayerIdx > 0;
       LayerIdx--) {
    CalculateMidLayerDelta (LayerIdx);
  }

  DEBUG_START()
  PrintNodeDelta ();
  DEBUG_END()
}