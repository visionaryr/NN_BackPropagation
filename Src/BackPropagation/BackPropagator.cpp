#include "BackPropagator.h"
#include "DebugLib.h"

using namespace std;

/**
  Constructor for BackPropagator class.

  @param  FCN   A reference to a FullyConnectedNetwork object to be associated with this BackPropagator.

**/
BackPropagator::BackPropagator (
  FullyConnectedNetwork &FCN
  ) : Network (FCN)
{
  InitNodeDelta ();

  InitTrainingParams ();
}

/**
  Initialize delta values of each layer to zero.

**/
void
BackPropagator::InitNodeDelta ()
{
  vector<unsigned int> Layout = Network.GetLayout ();

  for(int Index = 0; Index < (int)Layout.size(); Index++) {
    matrix LayerDelta (Layout[Index], 1);
    NodeDelta.push_back (LayerDelta);
  }
}

/**
  Set the delta value of a specific node in a specific layer.

  @param  Layer   An unsigned integer representing the layer index.
  @param  Number  An unsigned integer representing the node index within the layer.
  @param  Delta   A double representing the delta value to be set for the specified node.

  @throw std::runtime_error if the Layer or Number index is out of range.

**/
void BackPropagator::SetNodeDelta (
  unsigned int Layer,
  unsigned int Number,
  double       Delta
  )
{
  vector<unsigned int> Layout = Network.GetLayout ();

  if (Layer >= (unsigned int)Layout.size()) {
    DEBUG_LOG ("Layer: " << Layer << ", Layout size: " << Layout.size());
    throw std::runtime_error("Error: Layer index out of range in SetNodeDelta().");
  }
  if (Number >= (unsigned int)Layout[Layer]) {
    DEBUG_LOG ("Number: " << Number << ", Nodes in layer: " << Layout[Layer]);
    throw std::runtime_error("Error: Node number index out of range in SetNodeDelta().");
  }

  NodeDelta[Layer].SetValue (Number, 0, Delta);
}

/**
  [**Only for internal debug use**]
  Print the delta values of all nodes in each layer.

**/
void
BackPropagator::PrintNodeDelta(
  void
  )
{
  for(int Index = 0; Index < (int)NodeDelta.size(); Index++) {
    cout << "Delta in layer " << Index<< ":" <<endl;
    NodeDelta[Index].show();
  }
}

/**
  [**Only for internal debug use**]
  Print the delta weights between each layers.

**/
void
BackPropagator::PrintDeltaWeights(
  void
  )
{
  for(int Index = 0; Index < (int)DeltaWeights.size(); Index++) {
    cout << "Delta weights between layer" << Index << " and layer " << Index + 1 << endl;
    DeltaWeights[Index].show();
  }
}

/**
  Initialize delta weights between each layers to zero.

**/
void BackPropagator::InitDeltaWeights ()
{
  vector<unsigned int> Layout = Network.GetLayout ();

  for(unsigned int Index = 0; Index < (unsigned int)Layout.size() - 1; Index++) {
    matrix LayerDeltaWeights (Layout[Index + 1], Layout[Index]);
    DeltaWeights.push_back (LayerDeltaWeights);
  }
}

/**

**/
void
BackPropagator::InitBatchDeltaWeights (
  void
  )
{
  vector<unsigned int> Layout = Network.GetLayout ();

  BatchDeltaWeights.clear();

  for (unsigned int Index = 0; Index < (unsigned int)Layout.size() - 1; Index++) {
    matrix LayerDeltaWeights (Layout[Index + 1], Layout[Index]);
    BatchDeltaWeights.push_back (LayerDeltaWeights);
  }
}

/**
  Initialize training mode related settings.
  No initialization is required for PATTERN_MODE.

**/
void
BackPropagator::InitTrainingMode (
  void
  )
{
  InitBatchDeltaWeights ();
}

/**
  Train the network with one data sample, including forward pass and backward pass.

  @param[in]  InputData     A matrix representing the input data.
  @param[in]  DesiredOutput A matrix representing the desired output values.
  @param[in]  LearningRate  A double representing the learning rate for weight updates.

  @return A double representing the loss value after training with this data sample.

**/
double
BackPropagator::TrainOneData (
  const matrix  &InputData,
  const matrix  &DesiredOutput,
  const double  LearningRate
  )
{
  double  Loss;

  Network.Forward (InputData);

  Loss = LossMeanSquareError (DesiredOutput);

  BackwardPass (DesiredOutput, LearningRate);

  return Loss;
}

/**
  Train the network for one epoch over the entire dataset.

  @param[in]  InputDataSet     A vector of matrices representing the input data samples.
  @param[in]  DesiredOutputSet A vector of matrices representing the desired output values for each sample.
  @param[in]  LearningRate     A double representing the learning rate for weight updates.

  @return A double representing the average loss over the epoch.

**/
double
BackPropagator::TrainOneEpoch (
  const vector<matrix> &InputDataSet,
  const vector<matrix> &DesiredOutputSet,
  const double         LearningRate
  )
{
  double        EpochLoss = 0.0;
  unsigned int  TrainedDataCount;

  TrainedDataCount = 0;
  for (unsigned int DataIndex = 0; DataIndex < (unsigned int)InputDataSet.size(); DataIndex++) {
    EpochLoss += TrainOneData (
                   InputDataSet[DataIndex],
                   DesiredOutputSet[DataIndex],
                   LearningRate
                   );

    TrainedDataCount++;

    //
    // Update weights in batch mode after processing a batch of data samples.
    //
    if (TrainedDataCount == BatchSize) {
      AverageBatchDeltaWeights (BatchSize);
      UpdateWeights (BatchDeltaWeights);

      TrainedDataCount = 0;
      InitBatchDeltaWeights ();
    }
  }

  return (double)(EpochLoss / InputDataSet.size());
}

void
BackPropagator::Train (
  vector<matrix>  &InputDataSet,
  vector<matrix>  &DesiredOutputSet
  )
{
  if (InputDataSet.size() != DesiredOutputSet.size()) {
    DEBUG_LOG (__FUNCTION__ << ": InputData count = " << InputDataSet.size() << ", DesiredOutput count = " << DesiredOutputSet.size());
    throw runtime_error ("Amount of InputData and DesiredOutput isn't match.");
  }
  if (BatchSize > (unsigned int)InputDataSet.size()) {
    DEBUG_LOG (__FUNCTION__ << ": Batch size = " << BatchSize << ", InputData count = " << InputDataSet.size());
    throw runtime_error ("Batch size can't be larger than total training data set size.");
  }

  InitTrainingMode ();

  ShowTrainingParams ();

  double        EpochLoss;
  clock_t       StartTime;
  clock_t       EndTime;

  for (unsigned int Epoch = 1; Epoch <= Epochs; Epoch++) {
    StartTime = clock ();
  
    cout << "Training Epoch #" << Epoch << endl;

    EpochLoss = TrainOneEpoch (
                  InputDataSet,
                  DesiredOutputSet,
                  LearningRate
                  );

    EndTime = clock ();

    cout << "Epoch #" << Epoch << ": " << endl;
    cout << "  Loss = " << EpochLoss << endl;
    cout << "  Consume time = " << (double)(EndTime - StartTime) / CLOCKS_PER_SEC << " seconds" << endl;

    if (EpochLoss < TargetLoss) {
      DEBUG_LOG ("Loss of this epoch is lower than target loss(" << TargetLoss << ")");
      break;
    }

    //
    // TODO: Preturb(shake) weights if loss between each rounds doesn't have much difference.
    //
  }
}