#ifndef _BACK_PROPAGATOR_H_
#define _BACK_PROPAGATOR_H_

#include "matrix.h"
#include "FullyConnectedNetwork.h"
#include "ThreadPool.h"

#include <vector>
#include <functional>
#include <mutex>

typedef enum {
  BATCH_MODE = 0,
  PATTERN_MODE,
  TRAINING_MODE_MAX
} TRAINING_MODE;

class BackPropagator
{
  public:
    BackPropagator (FullyConnectedNetwork &FCN);

    //
    // Public functions to configure training parameters.
    //
    void SetLearningRate (
      const double  LearningRate
      );

    void SetEpochs (
      const unsigned int  Epochs
      );

    void SetTargetLoss (
      const double  TargetLoss
      );

    void SetTrainingMode (
      const TRAINING_MODE  TrainingMode,
      const unsigned int   BatchSize
      );

    void
    ShowTrainingParams (
      void
      );

    //
    // Function to start training process.
    //
    void Train (
      std::vector<matrix>  &InputDataSet,
      std::vector<matrix>  &DesiredOutputSet
      );

  private:
    void
    NodeDeltaCalculation (
      const matrix        &DesiredOutput,
      ComputationContext  &Context
      );

    matrix
    CalculateLastLayerDelta (
      const matrix        &DesiredOutput,
      ComputationContext  &Context
      );
    matrix
    CalculateMidLayerDelta (
      unsigned int        Layer,
      ComputationContext  &Context
      );

    void
    DeltaWeightsCalculation (
      vector<matrix>      DeltaWeights,
      ComputationContext  &Context
      );

    void
    BackwardPass (
      const matrix        &DesiredOutput,
      vector<matrix>      &DeltaWeights,
      ComputationContext  &Context
      );

    void  UpdateWeights (
      std::vector<matrix>  DeltaWeights
      );

    void
    InitTrainingMode (
      void
      );

    void
    InitBatchDeltaWeights (
      void
      );

    void
    UpdateBatchDeltaWeights (
      vector<matrix>  &DeltaWeights
      );

    void
    AverageBatchDeltaWeights (
      unsigned int  TotalTrainDataSetCount
    );

    double
    LossMeanSquareError (
      const matrix        &DesiredOutput,
      ComputationContext  &Context
      );

    double
    TrainOneData (
      const matrix        &InputData,
      const matrix        &DesiredOutput,
      vector<matrix>      &DeltaWeights,
      ComputationContext  &Context
      );

    void  TrainOneSubBatch (
      const std::vector<matrix>        &InputData,
      const std::vector<matrix>        &DesiredOutput,
      const std::vector<unsigned int>  IndexToTrain
      );

    double  TrainOneEpoch (
      const std::vector<matrix>  &InputData,
      const std::vector<matrix>  &DesiredOutput
      );

    // std::optional<std::reference_wrapper<FullyConnectedNetwork>>  Network;
    FullyConnectedNetwork          &Network;

    std::vector<matrix>            BatchDeltaWeights;

    //
    // Training parameters
    //
    void
    InitTrainingParams (
      void
      );

    //
    // Temp data during training process.
    //
    double  EpochLoss;

    //
    // Training parameters.
    //
    double                 LearningRate;
    unsigned int           Epochs;
    double                 TargetLoss;
    TRAINING_MODE          TrainingMode;
    unsigned int           BatchSize;

    //
    // Multi-threading computation of training.
    //
    ThreadPool  TrainingThreads;
    std::mutex  DeltaWeightsMutex;
};

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
  const std::vector<double>&  Values
  );

/**
  Generate a vector containing unique integers in the range [0, MaxNumber-1]
  in a random order.

  @param[in]  MaxNumber  The exclusive upper bound of generated numbers.

  @return A vector<unsigned int> containing all integers from 0 to MaxNumber-1,
          shuffled randomly.

  @throws std::invalid_argument If MaxNumber == 0.
  @throws std::runtime_error    If the RNG produces an error (very unlikely).

**/
std::vector<unsigned int>
GenerateUniqueRandomSequence (
  unsigned int MaxNumber
  );

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
  );

#endif
