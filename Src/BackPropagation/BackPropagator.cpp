#include "BackPropagator.h"
#include "DebugLib.h"

using namespace std;

/**
  Constructor for BackPropagator class.

  @param  FCN   A reference to a FullyConnectedNetwork object to be associated with this BackPropagator.

**/
BackPropagator::BackPropagator (
  FullyConnectedNetwork &FCN
  ) : Network (FCN)
{
  InitNodeDelta ();
  // InitDeltaWeights ();
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

  for(int Index = 0; Index < (int)Layout.size() - 1; Index++) {
    matrix LayerDeltaWeights (Layout[Index + 1], Layout[Index]);
    DeltaWeights.push_back (LayerDeltaWeights);
  }
}

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

double
BackPropagator::TrainOneEpoch (
  const vector<matrix> &InputDataSet,
  const vector<matrix> &DesiredOutputSet,
  const double         LearningRate
  )
{
  double  EpochLoss = 0.0;

  for (unsigned int DataIndex = 0; DataIndex < (unsigned int)InputDataSet.size(); DataIndex++) {
    EpochLoss += TrainOneData (
                   InputDataSet[DataIndex],
                   DesiredOutputSet[DataIndex],
                   LearningRate
                   );
  }

  return (double)(EpochLoss / InputDataSet.size());
}

void
BackPropagator::Train (
  const vector<matrix>  &InputDataSet,
  const vector<matrix>  &DesiredOutputSet,
  const double          LearningRate,
  const unsigned int    Epochs,
  const double          TargetLoss
  )
{
  if (InputDataSet.size() != DesiredOutputSet.size()) {
    DEBUG_LOG (__FUNCTION__ << ": InputData count = " << InputDataSet.size() << ", DesiredOutput count = " << DesiredOutputSet.size());
    throw runtime_error ("Amount of InputData and DesiredOutput isn't match.");
  }
  if (Epochs == 0) {
    throw runtime_error ("Epochs should at least be 1.");
  }

  double        EpochLoss;
  clock_t       StartTime;
  clock_t       EndTime;

  for (unsigned int Epoch = 1; Epoch <= Epochs; Epoch++) {
    StartTime = clock ();
  
    cout << "Training Epoch #" << Epoch << endl;

    EpochLoss = TrainOneEpoch (
                  InputDataSet,
                  DesiredOutputSet,
                  LearningRate
                  );

    EndTime = clock ();

    cout << "Epoch #" << Epoch << ": " << endl;
    cout << "  Loss = " << EpochLoss << endl;
    cout << "  Consume time = " << (double)(EndTime - StartTime) / CLOCKS_PER_SEC << " seconds" << endl;

    if (EpochLoss < TargetLoss) {
      DEBUG_LOG ("Loss of this epoch is lower than target loss(" << TargetLoss << ")");
      break;
    }

    //
    // TODO: Preturb(shake) weights if loss between each rounds doesn't have much difference.
    //
  }
}