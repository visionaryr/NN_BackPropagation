/**
  ConfigurationData class implementation for Back Propagation Neural Network.

  Copyright (c) 2026, visionaryr
  Licensed under the MIT License. See the accompanying 'LICENSE' file for details.

**/

#ifndef _CONFIGURATION_DATA_H_
#define _CONFIGURATION_DATA_H_

#include <vector>
#include <string>

typedef enum {
  BATCH_MODE = 0,
  PATTERN_MODE,
  TRAINING_MODE_MAX
} TRAINING_MODE;

typedef struct {
  std::vector<unsigned int> Layout;
} NETWORK_CONFIG;

typedef struct {
  double LearningRate;
  unsigned int Epochs;
  double TargetLoss;
  unsigned int BatchSize;
  TRAINING_MODE TrainingMode;
}TRAINING_CONFIG;

typedef struct {
  std::vector<unsigned int> TrainingCategories;
} DATA_CONFIG;

class ConfigurationData {
  public:
    ConfigurationData (
      void
      );

    bool IsValid() const {
      return Valid;
    }

    void SetValid() {
      Valid = true;
    }

    NETWORK_CONFIG   Network;
    TRAINING_CONFIG  Training;
    DATA_CONFIG      Data;

  private:
    bool  Valid;
};

#endif