/**
  Main entry point for Back Propagation training on MNIST dataset.

  Copyright (c) 2026, visionaryr
  Licensed under the MIT License. See the accompanying 'LICENSE' file for details.
**/

#include "matrix.h"
#include "BackPropagator.h"
#include "FullyConnectedNetwork.h"
#include "PreProcess.h"
#include "PngIo.h"
#include "MnistDataSet.h"
#include "ConfigParser.h"

#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <queue>
#include <cmath>
#include <iomanip>
#include <set>
#include <cstring>
#include <filesystem>

#define ARRAY_SIZE(Array) \
  (sizeof(Array) / sizeof(Array[0]))

#define PAUSE printf("Press Enter key to continue..."); fgetc(stdin);
#define train_start 0
//#define train_images 60
#define test_images 0
#define last_save 15

using namespace std;

int mDefaultTrainingCategories[] = {
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9
};

int mDefaultNetworkLayout[] = {
  784,  // Input layer
  30,   // Hidden layer
  10    // Output layer
};

/**
  Get the root path for MNIST dataset files.

  This function returns the root path where the MNIST dataset files are located.
  If ROOT_PATH is defined during compilation, it returns that path.
  Otherwise, it returns the current working directory.

  @return  The root path as a string.

**/
string
GetRootPath (
  void
  )
{
#ifdef ROOT_PATH
  string  RootPath(ROOT_PATH);

  return RootPath;
#else
  return filesystem::current_path().string() + "/";
#endif
}

vector<matrix>
ConvertLabelsToNetworkOutput (
  LABELS                &LabelSet,
  vector<unsigned int>  &TrainingLabels
  )
{
  vector<matrix>  DesiredOutputs;

  for (unsigned int Index = 0; Index < (unsigned int)LabelSet.size(); Index ++) {
    matrix  DesiredOutput = ConvertOutputValueToMatrix (
                              LabelSet[Index],
                              TrainingLabels
                              );
    DesiredOutputs.push_back (DesiredOutput);
  }

  return DesiredOutputs;
}

vector<matrix>
ConvertDataToNetworkInput (
  DATA_SET  &DataSet
  )
{
  vector<matrix>  DataInputs;

  for (unsigned int Index = 0; Index < (unsigned int)DataSet.size(); Index++) {
    vector<double>  DataInput1dVector = DataSet[Index].ConvertToVector ();

    matrix  DataInput (DataInput1dVector.size(), 1, DataInput1dVector);

    DataInputs.push_back (DataInput);
  }

  return DataInputs;
}

/**

**/
int
main (
  int argc,
  char **argv
  )
{
  DATA_SET        DataSet;
  LABELS          LabelSet;
  LABELS          TestLabelSet;
  vector<matrix>  DesiredOutputs;
  vector<matrix>  DataInputs;
  vector<matrix>  TestDataInputs;
  string          Filename;
  std::string     ConfigFilePath;
  ConfigurationData AppConfig;
  ConfigParser      ConfigParser;

  //
  // Initialize random generator.
  //
  srand (time (NULL));

  // Parse command line arguments for config file (-f <file>)
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-f") == 0 && (i + 1) < argc) {
      ConfigFilePath = argv[++i];
    }
  }

  // Only load configuration if the user explicitly provided a config file via -f
  if (!ConfigFilePath.empty()) {
    try {
      AppConfig = ConfigParser.ParseConfigFile(GetRootPath() + ConfigFilePath);
    } catch (const std::exception &e) {
      cout << "Config file load error: " << e.what() << "\nUsing defaults." << endl;
      // keep AppConfig defaults
    }
  } else {
    // No config file provided; use built-in hardcoded defaults
    cout << "No config file specified; using built-in defaults." << endl;
  }

  //
  // Build training categories from config or fallback to hardcoded defaults
  vector<unsigned int> TrainingCategories;
  if (!AppConfig.Data.TrainingCategories.empty()) {
    TrainingCategories = AppConfig.Data.TrainingCategories;
  } else {
    TrainingCategories = vector<unsigned int> (mDefaultTrainingCategories, mDefaultTrainingCategories + ARRAY_SIZE(mDefaultTrainingCategories));
  }

  // Build network layout from config or fallback to hardcoded defaults
  NETWORK_LAYOUT Layout;
  if (!AppConfig.Network.Layout.empty()) {
    Layout = NETWORK_LAYOUT (AppConfig.Network.Layout.begin(), AppConfig.Network.Layout.end());
  } else {
    Layout = NETWORK_LAYOUT (mDefaultNetworkLayout, mDefaultNetworkLayout + ARRAY_SIZE(mDefaultNetworkLayout));
  }

  // Check if train categories match network output layer size.
  if (TrainingCategories.size() != (size_t)Layout.back()) {
    cout << "Error: Training categories size does not match network output layer size!" << endl;
    return -1;
  }

  //
  // Get trainning data set.
  // Convert LabelSet to matrix format to match with network output.
  //
  ReadMNIST_and_label (TRAINING_DATA, DataSet, LabelSet, TrainingCategories);
  DataInputs     = ConvertDataToNetworkInput (DataSet);
  DesiredOutputs = ConvertLabelsToNetworkOutput (LabelSet, TrainingCategories);

  //
  // Initialize network, here we use Fully Connected Network(FCN)
  FullyConnectedNetwork  FCN (Layout);

  //
  // Test the trained network
  //
  ReadMNIST_and_label (TEST_DATA, DataSet, TestLabelSet, TrainingCategories);
  TestDataInputs = ConvertDataToNetworkInput (DataSet);

  //
  // Initialize trainning algorithm and parameters, here we use Back Propagation.
  BackPropagator  TrainingAlgoBp (FCN);

  if (AppConfig.IsValid()) {
    TrainingAlgoBp.SetTrainingParameters (AppConfig.Training);
  }

  TrainingAlgoBp.Train (
    DataInputs,      // Input data
    DesiredOutputs   // Desired Output
    );

  unsigned int        Score = 0;
  ComputationContext  Context (Layout);

  for (unsigned int Index = 0; Index < TestDataInputs.size(); Index++) {
    unsigned int  PredictedLabel = FCN.Predict (TestDataInputs[Index], Context);

    Score += (TrainingCategories[PredictedLabel] == TestLabelSet[Index]) ? 1 : 0;

    // cout << "Test Image " << Index << ": Predicted Label = " << TrainingCategories[PredictedLabel] << ", Actual Label = " << LabelSet[Index] << endl;
  }

  // cout << "==================== Training Round " << Rounds + 1 << " Score ====================" << endl;
  cout << "Final Score: " << Score << " / " << TestDataInputs.size() << endl;

  // Use an ostringstream to format accuracy so we don't modify cout's global formatting state.
  {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << (double)Score / TestDataInputs.size() * 100;
    cout << "Accuracy: " << oss.str() << " %" << endl;
  }

  // Filename = "Round_" + to_string (Rounds + 1) + ".dat";

  // FCN.ExportToFile (GetRootPath() + "Test", Filename);
  // }

  return 0;
}