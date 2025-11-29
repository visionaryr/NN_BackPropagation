#include "BackPropagator.h"
#include "DebugLib.h"

#include <set>
#include <cmath>

using namespace std;

#define  MAX_EPOCHS_TO_TRACK_LOSS  10
#define  SHAKE_WEIGHT_THRESHOLD    0.001

/**
  Calculate the standard deviation of a vector of double values.

  @param[in]  Values   A vector of double values.

  @retval   The standard deviation of the input values.

  @throws   std::invalid_argument  If the input vector is empty or has less than 2 values.
  @throws   std::runtime_error     If the input vector contains NaN or Inf values,
                                   or if the standard deviation calculation results in NaN or Inf.
**/
double
CalculateStandardDeviation (
  const vector<double>& Values
  )
{
  if (Values.empty()) {
    throw std::invalid_argument("Input vector is empty. Cannot calculate standard deviation of an empty vector.");
  }
  
  if (Values.size() == 1) {
    throw std::invalid_argument("Input vector must contain at least 2 values to calculate standard deviation.");
  }

  //
  // Check for NaN or Inf values
  //
  for (size_t Index = 0; Index < Values.size(); ++Index) {
    if (std::isnan(Values[Index]) || std::isinf(Values[Index])) {
      throw std::runtime_error("Input vector contains NaN or Inf values.");
    }
  }

  //
  // Calculate mean
  //
  double Sum = 0.0;
  for (size_t Index = 0; Index < Values.size(); ++Index) {
    Sum += Values[Index];
  }
  double Mean = Sum / Values.size();

  //
  // Calculate variance
  //
  double Variance = 0.0;
  for (size_t Index = 0; Index < Values.size(); ++Index) {
    double Diff = Values[Index] - Mean;
    Variance += Diff * Diff;
  }
  Variance /= Values.size();

  //
  // Calculate and return standard deviation
  //
  double StdDev = std::sqrt(Variance);
  
  if (std::isnan(StdDev) || std::isinf(StdDev)) {
    throw std::runtime_error("Standard deviation calculation resulted in NaN or Inf.");
  }
  
  return StdDev;
}

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
  double             EpochLoss = 0.0;
  set<unsigned int>  TrainedDataIndex;
  unsigned int       RandIndex;

  while (TrainedDataIndex.size() < InputDataSet.size()) {
    RandIndex = rand() % InputDataSet.size();
    if (TrainedDataIndex.count (RandIndex) != 0) {
      continue;
    }

    TrainedDataIndex.insert (RandIndex);

    EpochLoss += TrainOneData (
                   InputDataSet[RandIndex],
                   DesiredOutputSet[RandIndex],
                   LearningRate
                   );

    //
    // Update weights in batch mode after processing a batch of data samples.
    //
    if ((TrainedDataIndex.size() % BatchSize) == 0) {
      AverageBatchDeltaWeights (BatchSize);
      UpdateWeights (BatchDeltaWeights);

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

  double          EpochLoss;
  clock_t         StartTime;
  clock_t         EndTime;
  vector<double>  Last10EpochsLoss;
  double          StdDev = 0.0;

  for (unsigned int Epoch = 1; Epoch <= Epochs; Epoch++) {
    StartTime = clock ();
  
    cout << "Training Epoch #" << Epoch << endl;

    EpochLoss = TrainOneEpoch (
                  InputDataSet,
                  DesiredOutputSet,
                  LearningRate
                  );

    EndTime = clock ();

    if (EpochLoss < TargetLoss) {
      DEBUG_LOG ("Loss of this epoch is lower than target loss(" << TargetLoss << ")");
      break;
    }

    //
    // Calculate the standard deviation of the loss over the last 10 epochs.
    //
    Last10EpochsLoss.push_back (EpochLoss);

    if (Last10EpochsLoss.size() > MAX_EPOCHS_TO_TRACK_LOSS) {
      Last10EpochsLoss.erase(Last10EpochsLoss.begin());
    }

    //
    // Calculate standard deviation when you have at least 2 values
    //
    if (Last10EpochsLoss.size () >= 2) {
      try {
        StdDev = CalculateStandardDeviation (Last10EpochsLoss);
      }
      catch (const std::exception& Exception) {
        cerr << "Error calculating standard deviation: " << Exception.what() << endl;
      }
    }

    cout << "Epoch #" << Epoch << ": " << endl;
    cout << "  Loss = " << EpochLoss << endl;
    cout << "  Consume time = " << (double)(EndTime - StartTime) / CLOCKS_PER_SEC << " seconds" << endl;
    if (Last10EpochsLoss.size () >= 2) {
      cout << "  StdDev of last " << Last10EpochsLoss.size() << " epochs loss = " << StdDev << endl;
    }

    //
    // If standard deviation is smaller than threshold, means loss hasn't
    // change much in last 10 epochs, shake weights.
    //
    if ((StdDev < SHAKE_WEIGHT_THRESHOLD) &&
        (StdDev != 0.0)) {
      Network.PerturbWeight();
    }
  }
}