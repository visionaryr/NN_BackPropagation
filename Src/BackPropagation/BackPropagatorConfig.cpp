/**
  BackPropagator configuration functions implementation.

  Copyright (c) 2025, visionaryr
  Licensed under the MIT License. See the accompanying 'LICENSE' file for details.
**/

#include "BackPropagator.h"
#include "DebugLib.h"

using namespace std;

void
BackPropagator::InitTrainingParams (
  void
  )
{
  LearningRate = 0.1;
  Epochs       = 10;
  TargetLoss   = 0.5;
  TrainingMode = BATCH_MODE;
  BatchSize    = 200;
}

void
BackPropagator::SetLearningRate (
  const double  LearningRate
  )
{
  this->LearningRate = LearningRate;
}

void
BackPropagator::SetEpochs (
  const unsigned int  Epochs
  )
{
  if (Epochs == 0) {
    DEBUG_LOG ("Epochs should at least be 1.");
    throw invalid_argument ("BackPropagator::SetEpochs(): Invalid Epochs.");
  }

  this->Epochs = Epochs;
}

void
BackPropagator::SetTargetLoss (
  const double  TargetLoss
  )
{
  this->TargetLoss = TargetLoss;
}

void
BackPropagator::SetTrainingMode (
  const TRAINING_MODE  TrainingMode
  )
{
  if (TrainingMode >= TRAINING_MODE_MAX) {
    DEBUG_LOG ("Training mode = " << TrainingMode << " is unsupported.");
    throw invalid_argument ("BackPropagator::SetTrainingMode (): Unsupported training mode.");
  }
  if (BatchSize == 0) {
    DEBUG_LOG ("Size of a batch should at least be 1.");
    throw invalid_argument ("BackPropagator::SetTrainingMode (): Invalid batch size.");
  }

  this->TrainingMode = TrainingMode;

  if (TrainingMode == PATTERN_MODE) {
    this->BatchSize = 1;
  }
}

void
BackPropagator::SetBatchSize (
  const unsigned int   BatchSize
  )
{
  if (BatchSize == 0) {
    DEBUG_LOG ("Size of a batch should at least be 1.");
    throw invalid_argument ("BackPropagator::SetBatchSize (): Invalid batch size.");
  }
  if (this->TrainingMode != BATCH_MODE) {
    DEBUG_LOG ("Cannot set batch size when training mode is not BATCH_MODE.");
    throw invalid_argument ("BackPropagator::SetBatchSize (): Invalid operation in current training mode.");
  }

  this->BatchSize = BatchSize;
}

void
BackPropagator::ShowTrainingParams (
  void
  )
{
  cout << endl << "=========Training Parameters:=========" << endl;
  cout << "  Learning Rate : " << LearningRate << endl;
  cout << "  Epochs        : " << Epochs << endl;
  cout << "  Target Loss   : " << TargetLoss << endl;
  cout << "  Training Mode : " << ((TrainingMode == BATCH_MODE) ? "BATCH_MODE" : "PATTERN_MODE") << endl;
  if (TrainingMode == BATCH_MODE) {
    cout << "  Batch Size    : " << BatchSize << endl;
  }

  cout << "======================================" << endl;
}