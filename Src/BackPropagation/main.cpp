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

#define ARRAY_SIZE(Array) \
  (sizeof(Array) / sizeof(Array[0]))

#define PAUSE printf("Press Enter key to continue..."); fgetc(stdin);
#define train_start 0
//#define train_images 60
#define test_images 0
#define last_save 15

using namespace std;

int mTrainingCategories[] = {
  //0, 1, 2, 3, 4, 5, 6, 7, 8, 9
  1, 2, 3
};

int mNetworkLayout[] = {
  784,  // Input layer
  15,   // Hidden layer
  3     // Output layer
};

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
  vector<matrix>  DesiredOutputs;
  vector<matrix>  DataInputs;

  //
  // Initialize random generator.
  //
  srand (time (NULL));

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
  // Initialize trainning algorithm and parameters, here we use Back Propagation.
  //
  BackPropagator  TrainingAlgoBp (FCN);

  TrainingAlgoBp.SetLearningRate (0.1);
  TrainingAlgoBp.SetEpochs (10);
  TrainingAlgoBp.SetTargetLoss (0.05);
  TrainingAlgoBp.SetTrainingMode (BATCH_MODE, 300);

  TrainingAlgoBp.Train (
    DataInputs,      // Input data
    DesiredOutputs   // Desired Output
    );

  //
  // Test the trained network
  //
  ReadMNIST_and_label (TEST_DATA, DataSet, LabelSet, TrainingCategories);
  DataInputs     = ConvertDataToNetworkInput (DataSet);

  unsigned int        Score = 0;
  ComputationContext  Context (Layout);

  for (unsigned int Index = 0; Index < DataInputs.size(); Index++) {
    unsigned int  PredictedLabel = FCN.Predict (DataInputs[Index], Context);

    Score += (TrainingCategories[PredictedLabel] == LabelSet[Index]) ? 1 : 0;

    cout << "Test Image " << Index << ": Predicted Label = " << TrainingCategories[PredictedLabel] << ", Actual Label = " << LabelSet[Index] << endl;
  }

  cout << "Final Score: " << Score << " / " << DataInputs.size() << endl;
  cout << "Accuracy: " << fixed << setprecision(2) << (double)Score / DataInputs.size() * 100 << " %" << endl;

  return 0;
}