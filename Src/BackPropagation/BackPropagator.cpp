/**
  BackPropagator class base implementation.

  Copyright (c) 2025, visionaryr
  Licensed under the MIT License. See the accompanying 'LICENSE' file for details.
**/

#include "BackPropagator.h"
#include "DebugLib.h"

#include <set>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>

using namespace std;

#define  MAX_EPOCHS_TO_TRACK_LOSS  10
#define  SHAKE_WEIGHT_THRESHOLD    0.001

/**
  Constructor for BackPropagator class.

  @param  FCN   A reference to a FullyConnectedNetwork object to be associated with this BackPropagator.

**/
BackPropagator::BackPropagator (
  FullyConnectedNetwork &FCN
  ) : Network (FCN)
{
  EpochLoss = 0.0;

  InitTrainingParams ();
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

  @return A double representing the loss value after training with this data sample.

**/
double
BackPropagator::TrainOneData (
  const matrix        &InputData,
  const matrix        &DesiredOutput,
  vector<matrix>      &DeltaWeights,
  ComputationContext  &Context
  )
{
  double  Loss;

  Network.Forward (InputData, Context);

  Loss = LossMeanSquareError (DesiredOutput, Context);

  BackwardPass (DesiredOutput, DeltaWeights, Context);

  return Loss;
}

/**

**/
void
BackPropagator::TrainOneSubBatch (
  const vector<matrix>        &InputData,
  const vector<matrix>        &DesiredOutput,
  const vector<unsigned int>  IndexToTrain
  )
{
  NETWORK_LAYOUT      Layout = Network.GetLayout ();
  ComputationContext  Context (Layout);
  double              TotalLoss;
  unsigned int        DataIndex;
  vector<matrix>      DeltaWeights;


  //
  // Initialize Delta Weights.
  //
  for(unsigned int Index = 0; Index < (unsigned int)Layout.size() - 1; Index++) {
    matrix LayerDeltaWeights (Layout[Index + 1], Layout[Index]);
    DeltaWeights.push_back (LayerDeltaWeights);
  }

  TotalLoss = 0.0;
  for (unsigned int Index = 0; Index < (unsigned int)IndexToTrain.size(); Index++) {
    DataIndex = IndexToTrain[Index];
    TotalLoss += TrainOneData (
                   InputData[DataIndex],
                   DesiredOutput[DataIndex],
                   DeltaWeights,
                   Context
                   );
  }

  unique_lock<mutex> lock (DeltaWeightsMutex);
  UpdateBatchDeltaWeights (DeltaWeights);
  EpochLoss += TotalLoss;
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
  const vector<matrix> &DesiredOutputSet
  )
{
  vector<unsigned int>  TrainSequence;
  unsigned int          NumOfThreads;
  unsigned int          BatchStartIndex;
  unsigned int          BatchEndIndex;

  EpochLoss = 0.0;

  NumOfThreads = TrainingThreads.GetNumOfThreads ();

  //
  // Step 1:
  //   Generate training order randomly.
  //   This is to make sure training between each epoch is running in different sequence.
  //
  TrainSequence = GenerateUniqueRandomSequence (InputDataSet.size());

  BatchStartIndex = 0;
  BatchEndIndex   = min (BatchSize, (unsigned int)TrainSequence.size());
  while (BatchStartIndex < TrainSequence.size()) {
    //
    // Step 2:
    //   Divide one batch into multiple sub-batches for multi-threading training.
    //
    unsigned int BatchLen = BatchEndIndex - BatchStartIndex; // actual number of elements in this batch

    if (BatchLen == 0) {
      break; // nothing to do
    }

    // Distribute BatchLen across threads as evenly as possible
    unsigned int BaseSubBatchSize = BatchLen / NumOfThreads;
    unsigned int Extra            = BatchLen % NumOfThreads;

    for (unsigned int ThreadId = 0; ThreadId < NumOfThreads; ++ThreadId) {
      //
      // Compute start offset in the batch for this thread
      //
      unsigned int SubStartIndex = BatchStartIndex + ThreadId * BaseSubBatchSize;;

      //
      // Compute how many elements this thread should process
      // Add all extra to the last thread
      //
      unsigned int ThisSubBatchSize = BaseSubBatchSize + ((ThreadId == NumOfThreads - 1) ? Extra : 0u);

      //
      // Compute end index (exclusive) and clamp to BatchEndIndex to be safe
      //
      unsigned int SubEndIndex = std::min(SubStartIndex + ThisSubBatchSize, BatchEndIndex);

      // If no work for this thread, continue
      if (SubStartIndex >= SubEndIndex) {
        continue;
      }

      vector<unsigned int>  IndexToTrain;

      IndexToTrain = ExtractSubVectorFromTrainSequence (
                      TrainSequence,
                      SubStartIndex,
                      SubEndIndex
                      );

      //
      // Step 3:
      //   Enqueue training of each sub-batch task into the task queue of ThreadPool.
      //
      auto task = [this, &InputDataSet, &DesiredOutputSet, IndexToTrain]() mutable {
        // When this lambda is called, it executes the worker function
        // using the values it captured.
        this->TrainOneSubBatch(InputDataSet, DesiredOutputSet, IndexToTrain);
      };

      // The lambda captures (copies) sub_X and sub_Y, making them part of the task object.
      TrainingThreads.Enqueue(task);
    }

    //
    // Step 4:
    //   Wait until training of all sub-batch is complete.
    //
    TrainingThreads.WaitForAllTasksDone ();

    //
    // Step 5:
    //   Average the batch delta weights and update the network weights.
    //   Re-initialize the batch delta weights after updating.
    //
    unique_lock<mutex> lock (DeltaWeightsMutex);
    AverageBatchDeltaWeights (BatchSize);
    UpdateWeights (BatchDeltaWeights);

    InitBatchDeltaWeights ();

    lock.unlock();

    //
    // Step 6:
    //   Move to next batch.
    //
    BatchStartIndex = BatchEndIndex;
    BatchEndIndex   = min (BatchStartIndex + BatchSize, (unsigned int)TrainSequence.size());
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
  vector<double>  Last10EpochsLoss;
  double          StdDev = 0.0;

  for (unsigned int Epoch = 1; Epoch <= Epochs; Epoch++) {
    auto StartTime = chrono::high_resolution_clock::now();
  
    cout << "Training Epoch #" << Epoch << endl;

    EpochLoss = TrainOneEpoch (
                  InputDataSet,
                  DesiredOutputSet
                  );

    auto EndTime = chrono::high_resolution_clock::now();

    //
    // Calculate time taken for this epoch.
    //
    auto   DurationInMs = chrono::duration_cast<std::chrono::milliseconds>(EndTime - StartTime);
    double Duration = DurationInMs.count() / 1000.0; // in seconds

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
    cout << "  Consume time = " << Duration << " seconds" << endl;
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
  Generate a vector containing unique integers in the range [0, MaxNumber-1]
  in a random order.

  @param[in]  MaxNumber  The exclusive upper bound of generated numbers.

  @return A vector<unsigned int> containing all integers from 0 to MaxNumber-1,
          shuffled randomly.

  @throws std::invalid_argument If MaxNumber == 0.
  @throws std::runtime_error    If the RNG produces an error (very unlikely).

**/
vector<unsigned int>
GenerateUniqueRandomSequence (
  unsigned int MaxNumber
  )
{
  if (MaxNumber == 0) {
    throw std::invalid_argument("MaxNumber must be greater than 0.");
  }

  // Prepare the sequential list 0..MaxNumber-1
  vector<unsigned int> Sequence;
  Sequence.reserve(MaxNumber);
  for (unsigned int Index = 0; Index < MaxNumber; ++Index) {
    Sequence.push_back(Index);
  }

  // Shuffle using a non-deterministic random device where available
  try {
    std::random_device RandomDev;
    std::mt19937 Generator(RandomDev());
    std::shuffle(Sequence.begin(), Sequence.end(), Generator);
  }
  catch (const std::exception &Exception) {
    // Fall back to deterministic generator if random_device fails
    try {
      std::mt19937 Generator((unsigned int)time(nullptr));
      std::shuffle(Sequence.begin(), Sequence.end(), Generator);
    }
    catch (...) {
      throw std::runtime_error("Failed to generate random sequence.");
    }
  }

  return Sequence;
}

/**
  Extract a sub-vector of indices from TrainSequence based on 
  a start and end index.

  @param[in]  TrainSequence  A vector of unsigned int indices (training order).
  @param[in]  StartIndex     The start position in TrainSequence (inclusive).
  @param[in]  EndIndex       The end position in TrainSequence (exclusive).

  @return A vector<unsigned int> containing the sub-sequence of indices.

  @throws std::invalid_argument If indices are out of range or invalid.
**/
std::vector<unsigned int>
ExtractSubVectorFromTrainSequence (
  const std::vector<unsigned int> &TrainSequence,
  unsigned int                     StartIndex,
  unsigned int                     EndIndex
  )
{
  // Input validation
  if (StartIndex >= TrainSequence.size()) {
    throw std::invalid_argument("StartIndex is out of range.");
  }
  
  if (EndIndex > TrainSequence.size()) {
    throw std::invalid_argument("EndIndex is out of range.");
  }
  
  if (StartIndex >= EndIndex) {
    throw std::invalid_argument("StartIndex must be less than EndIndex.");
  }
  
  // Extract sub-vector
  std::vector<unsigned int> SubVector;
  SubVector.reserve(EndIndex - StartIndex);
  
  for (unsigned int Index = StartIndex; Index < EndIndex; ++Index) {
    SubVector.push_back(TrainSequence[Index]);
  }
  
  return SubVector;
}