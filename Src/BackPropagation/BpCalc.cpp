#include "BackPropagator.h"
#include "DebugLib.h"

#include <cmath>

using namespace std;

/**

**/
static
matrix
CalculateDerivativeActivation (
  const matrix     Activation,
  ACTIVATION_TYPE  ActivationType
  )
{
  function<double (double)>  DerivativeActivation = GetDeriativeActivationFunction (ActivationType);

  return Activation.ApplyElementWise (DerivativeActivation);
}

/**
  Calculate the delta value of each node in the last(output) layer.
  Delta = (Desired - Actual) * f'(Actual), where f(x) is the activation function and
  f'(x) is the derivative of activation function.

  @param[in]  DesiredOutput  A matrix representing the desired output values.

**/
matrix
BackPropagator::CalculateLastLayerDelta (
  const matrix        &DesiredOutput,
  ComputationContext  &Context
  )
{
  matrix        DesiredGap;
  unsigned int  LastLayerIndex;
  matrix        DerivativeActivation;

  LastLayerIndex = (unsigned int)(Network.GetLayout().size() - 1);

  DesiredGap = Substract (
                 DesiredOutput,
                 Context.GetActivationByLayer (LastLayerIndex)
                 );

  DerivativeActivation = CalculateDerivativeActivation (
                           Context.GetActivationByLayer (LastLayerIndex),
                           Network.GetActivationType ()
                           );

  return HadamardProduct (
           DesiredGap,
           DerivativeActivation
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
  unsigned int        Layer,
  ComputationContext  &Context
  )
{
  matrix  WeightedError;
  matrix  DerivativeActivation;

  if (Layer > (unsigned int)(Network.GetLayout().size() - 2)) {
    DEBUG_LOG ("Layer " << Layer << " is not a middle layer.");
    throw runtime_error ("Layer passed into CalculateMidLayerDelta() is out of range.");
  }

  WeightedError = multiply (
                    transpose (Network.GetWeightByLayer(Layer)),
                    Context.GetNodeDeltaByLayer (Layer + 1)
                    );

  DerivativeActivation = CalculateDerivativeActivation (
        Context.GetActivationByLayer (Layer),
        Network.GetActivationType ()
        );

  return HadamardProduct (
           WeightedError,
           DerivativeActivation
           );
}

/**
  Calculate the delta value of each node in all layers except the first(input) layer.

  @param[in]  DesiredOutput  A matrix representing the desired output values.

**/
void
BackPropagator::NodeDeltaCalculation (
  const matrix        &DesiredOutput,
  ComputationContext  &Context
  )
{
  matrix        NodeDeltaOfLayer;
  unsigned int  LastLayerIndex = (unsigned int)(Network.GetLayout().size() - 1);

  NodeDeltaOfLayer = CalculateLastLayerDelta (DesiredOutput, Context);
  Context.SetActivationByLayer (LastLayerIndex, NodeDeltaOfLayer);

  //
  // Calculate delta for all nodes in all layer except last layer.
  // Note: It's unnecessary to calculate the delta value of the first(input) layer(LayerIdx = 0).
  //
  for (unsigned int LayerIdx = LastLayerIndex - 1;
       LayerIdx > 0;
       LayerIdx--) {
    NodeDeltaOfLayer = CalculateMidLayerDelta (LayerIdx, Context);
    Context.SetNodeDeltaByLayer (LayerIdx, NodeDeltaOfLayer);
  }

  // DEBUG_START()
  // PrintNodeDelta ();
  // DEBUG_END()
}

/**
  Calculate the delta value of each node in all layers except the first(input) layer.

**/
void
BackPropagator::DeltaWeightsCalculation (
  vector<matrix>      DeltaWeights,
  ComputationContext  &Context
  )
{
  unsigned int  WeightsLayerCount = ((unsigned int)Network.GetLayout().size() - 1);

  for (unsigned int LayerIdx = 0; LayerIdx < WeightsLayerCount; LayerIdx++) {
    matrix  CurrentLayerActivation_T = transpose (Context.GetActivationByLayer (LayerIdx));
    matrix  NextLayerDelta           = Context.GetNodeDeltaByLayer (LayerIdx + 1);

    matrix  Gradient = multiply (NextLayerDelta, CurrentLayerActivation_T);
  
    matrix  DeltaWeight = multiplyBy (Gradient, LearningRate);

    DeltaWeights[LayerIdx] = add (DeltaWeights[LayerIdx], DeltaWeight);
  }
}

/**
  Update the weights of the network by applying the calculated delta weights.

**/
void
BackPropagator::UpdateWeights (
  vector<matrix>  DeltaWeights
  )
{
  if (DeltaWeights.empty () ||
      (DeltaWeights.size () != Network.GetLayout().size() - 1)) {
    DEBUG_LOG ("DeltaWeights Size = " << DeltaWeights.size() << " , Network Layout size = " << Network.GetLayout().size());
    DEBUG_LOG ("Delta Weights are not ready, Failed to update");
    return;
  }

  Network.UpdateWeight (DeltaWeights);
}

/**
  Update the batch mode delta weights by adding the current delta weights.

**/
void
BackPropagator::UpdateBatchDeltaWeights (
  vector<matrix>  &DeltaWeights
  )
{
  if (DeltaWeights.size() != BatchDeltaWeights.size()) {
    DEBUG_LOG ("DeltaWeights has size = " << DeltaWeights.size());
    DEBUG_LOG ("BatchDeltaWeights has size = " << BatchDeltaWeights.size());
    throw runtime_error ("Failed to update batch delta weights.");
  }

  for (unsigned int Index = 0; Index < BatchDeltaWeights.size(); Index++) {
    BatchDeltaWeights[Index] = add (BatchDeltaWeights[Index], DeltaWeights[Index]);
  }
}

/**
  Calculate the average of delta weights of a batch.
  The count of training data samples in the batch is specified by TotalTrainDataSetCount.

  @param[in]  TotalTrainDataSetCount  Total number of training data samples.

**/
void
BackPropagator::AverageBatchDeltaWeights (
  unsigned int  TotalTrainDataSetCount
  )
{
  if (TotalTrainDataSetCount == 0) {
    DEBUG_LOG ("Total train data set count is 0, failed to calculate average.");
    throw runtime_error ("Failed to calculate average if dividing 0.");
  }

  if (TotalTrainDataSetCount == 1) {
    return;
  }

  for (unsigned int Index = 0; Index < (unsigned int)BatchDeltaWeights.size(); Index++) {
    BatchDeltaWeights[Index] = multiplyBy (BatchDeltaWeights[Index], 1 / (double)TotalTrainDataSetCount);
  }
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
  const matrix        &DesiredOutput,
  vector<matrix>      &DeltaWeights,
  ComputationContext  &Context
  )
{
  NodeDeltaCalculation (DesiredOutput, Context);

  DeltaWeightsCalculation (DeltaWeights, Context);
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
  const matrix        &DesiredOutput,
  ComputationContext  &Context
  )
{
  matrix                   DesiredGap;
  function<double(double)> Square = InternalSquare;

  DesiredGap = Substract (
                 DesiredOutput,
                 Context.GetActivationByLayer (Network.GetLayout().size() - 1)
                 );

  return DesiredGap.ApplyElementWise (Square).Sum() / DesiredOutput.getrow();
}
