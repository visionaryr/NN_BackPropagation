/**
  ConfigurationData class implementation for Back Propagation Neural Network.

  Copyright (c) 2026, visionaryr
  Licensed under the MIT License. See the accompanying 'LICENSE' file for details.

**/

#include "ConfigurationData.h"

ConfigurationData::ConfigurationData (
  void
  )
{
  // Initialize with default values
  Network.Layout.clear();
  Training.LearningRate = 0;
  Training.Epochs = 0;
  Training.TargetLoss = 0;
  Training.BatchSize = 0;
  Training.TrainingMode = BATCH_MODE;
  Data.TrainingCategories.clear();
  Valid = false;  // Set to false until properly populated by config parser
}