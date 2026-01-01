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

int mTrainingCategories[] = {
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9
};

int mNetworkLayout[] = {
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
  void
  )
{
  DATA_SET        DataSet;
  LABELS          LabelSet;
  LABELS          TestLabelSet;
  vector<matrix>  DesiredOutputs;
  vector<matrix>  DataInputs;
  vector<matrix>  TestDataInputs;
  string          Filename;

  //
  // Initialize random generator.
  //
  srand (time (NULL));

  //
  // Check if train categories match network output layer size.
  //
  if (ARRAY_SIZE (mTrainingCategories) != mNetworkLayout[ARRAY_SIZE (mNetworkLayout) - 1]) {
    cout << "Error: Training categories size does not match network output layer size!" << endl;
    return -1;
  }

  //
  // Init categories to be trained.
  //
  vector<unsigned int> TrainingCategories (mTrainingCategories, mTrainingCategories + ARRAY_SIZE (mTrainingCategories));

  //
  // Get trainning data set.
  // Convert LabelSet to matrix format to match with network output.
  //
  ReadMNIST_and_label (TRAINING_DATA, DataSet, LabelSet, TrainingCategories);
  DataInputs     = ConvertDataToNetworkInput (DataSet);
  DesiredOutputs = ConvertLabelsToNetworkOutput (LabelSet, TrainingCategories);

  //
  // Initialize network, here we use Fully Connected Network(FCN)
  //
  NETWORK_LAYOUT  Layout (mNetworkLayout, mNetworkLayout + ARRAY_SIZE (mNetworkLayout));
  FullyConnectedNetwork  FCN (Layout);

  //
  // Test the trained network
  //
  ReadMNIST_and_label (TEST_DATA, DataSet, TestLabelSet, TrainingCategories);
  TestDataInputs = ConvertDataToNetworkInput (DataSet);

  //
  // Initialize trainning algorithm and parameters, here we use Back Propagation.
  //
  BackPropagator  TrainingAlgoBp (FCN);

  for (unsigned int Rounds = 0; Rounds < 5; Rounds++) {
    TrainingAlgoBp.SetLearningRate (0.1);
    TrainingAlgoBp.SetEpochs (2);
    TrainingAlgoBp.SetTargetLoss (0.05);
    TrainingAlgoBp.SetTrainingMode (BATCH_MODE);
    TrainingAlgoBp.SetBatchSize (300);

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

    cout << "==================== Training Round " << Rounds + 1 << " Score ====================" << endl;
    cout << "Final Score: " << Score << " / " << TestDataInputs.size() << endl;

    // Use an ostringstream to format accuracy so we don't modify cout's global formatting state.
    {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(2) << (double)Score / TestDataInputs.size() * 100;
      cout << "Accuracy: " << oss.str() << " %" << endl;
    }
  
    Filename = "Round_" + to_string (Rounds + 1) + ".dat";

    FCN.ExportToFile (GetRootPath() + "Test", Filename);
  }

  return 0;
}