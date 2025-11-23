#ifndef _BACK_PROPAGATOR_H_
#define _BACK_PROPAGATOR_H_

#include "matrix.h"
#include "FullyConnectedNetwork.h"

#include <vector>
#include <functional>

typedef enum {
  BATCH_MODE = 0,
  PATTERN_MODE,
  TRAINING_MODE_MAX
} TRAINING_MODE;

class BackPropagator
{
  public:
    BackPropagator (FullyConnectedNetwork &FCN);
  
    void Train (
      const std::vector<matrix>  &InputData,
      const std::vector<matrix>  &DesiredOutput,
      const double               LearningRate,
      const unsigned int         Epochs,
      const double               TargetLoss,
      const TRAINING_MODE        TrainingMode
      );

  private:
    void InitNodeDelta ();
    void InitDeltaWeights ();
    void SetNodeDelta (
      unsigned int  Layer,
      unsigned int  Number,
      double        Delta
      );
    void PrintNodeDelta (); // Only internal debug use.
    void PrintDeltaWeights (); // Only internal debug use.

    void NodeDeltaCalculation (
      const matrix &DesiredOutput
     );
    matrix  CalculateLastLayerDelta (
      matrix  DesiredOutput
      );
    matrix  CalculateMidLayerDelta (
      unsigned int  Layer
      );

    void  DeltaWeightsCalculation (
      const double  LearningRate
      );
    void BackwardPass (
      const matrix &DesiredOutput,
      const double LearningRate
      );

    void  UpdateWeights (
      std::vector<matrix>  DeltaWeights
      );

    void  InitTrainingMode (
      void
      );

    void  UpdateBatchModeDeltaWeights (
      void
      );

    void
    AverageBatchModeDeltaWeights (
      unsigned int  TotalTrainDataSetCount
    );

    double  LossMeanSquareError (
      const matrix &DesiredOutput
      );

    double  TrainOneData (
      const matrix  &InputData,
      const matrix  &DesiredOutput,
      const double  LearningRate
      );

    double  TrainOneEpoch (
      const std::vector<matrix>  &InputData,
      const std::vector<matrix>  &DesiredOutput,
      const double               LearningRate
      );

    FullyConnectedNetwork          &Network;
    std::vector<matrix>            NodeDelta;
    std::vector<matrix>            DeltaWeights;
    std::vector<matrix>            BatchModeDeltaWeights;

    TRAINING_MODE                  TrainingMode;
};

typedef matrix                 IMAGE;
typedef std::vector< IMAGE >   DATA_SET;
typedef std::vector< int >     LABELS;

//
// Dataset I/O functions
//
void ReadMNIST_and_label(
  DATA_SET  &DataSet,
  LABELS    &LabelSet,
  LABELS    &LabelsToRead
  );

//void read_Mnist_Label(std::vector< std::vector<double> > &, std::vector<int> &);

//other global functions
double predict(std::vector< std::vector<double> > &, std::vector< std::vector<double> > &, int, int, FullyConnectedNetwork &, std::vector<int> &);

#endif
