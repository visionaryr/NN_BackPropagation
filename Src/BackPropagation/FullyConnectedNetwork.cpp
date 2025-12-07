#include "matrix.h"
#include "FullyConnectedNetwork.h"
#include "DebugLib.h"

#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

/**
  Constructor to initialize a fully connected network with given network frame.
  The detail information of the network will be shown after initialization.

  @param  NetworkFrame  A vector of integers representing the number of nodes in each layer.

**/
FullyConnectedNetwork::FullyConnectedNetwork (
  NETWORK_LAYOUT  &NetworkFrame
  )
{
  Layout = NetworkFrame;
  ActivationType = SIGMOLD;

  WeightsMatrixInit(true);

  ShowInfo (false);
}

/**
  Constructor to initialize a fully connected network with given filename which contains all weights of each layer.
  The detail information of the network will be shown after initialization.

  @param  filename  A string representing the name of the file containing a FCN.

**/
FullyConnectedNetwork::FullyConnectedNetwork(string filename)
{
  ImportFromFile (filename);
  ActivationType = SIGMOLD;

  ShowInfo (false);
}

/**
  Show detailed information of the fully connected network.
  Information includes number of layers, number of nodes in input and output layers, layout of the network,
  and detailed weights of each layer (optional).

  @param  ShowWeightsDetail  A boolean indicating whether to show detailed weights of each layer.

**/
void FullyConnectedNetwork::ShowInfo (
  bool  ShowWeightsDetail
  )
{
  int WeightRow, WeightColumn;
  int MiddleLayers = Weights.size();

  cout << endl << "===== Fully Connected Network Info =====" << endl;
  cout << "Number of layers: " << Layout.size() << endl;
  cout << "Nodes in input layer: " << Layout[0] << endl;
  cout << "Nodes in output layer: " << Layout.back() << endl;

  cout << "Layout of the network: ";
  for (int Index = 0; Index < (int)Layout.size(); Index++) {
    cout << Layout[Index];
    if (Index != (int)Layout.size() - 1) {
      cout << " -> ";
    }
  }
  cout << endl;

  cout << "Detail of weights:" << endl;
  for(int Index = 0; Index < MiddleLayers; Index++) {
    //
    // Weight matrix between layer Index + 1 and layer Index + 2 will be a row * column matrix,
    // where row = number of nodes in layer Index + 2, column = number of nodes in layer Index + 1.
    // The weight matrix format will be shown as,
    //   Weight (L{Index + 1} <-> L{Index + 2}) = row * column
    //
    WeightRow    = Weights[Index].getrow();
    WeightColumn = Weights[Index].getcolumn();

    cout<<"  Weight (L" << Index + 1 << " <-> L" << Index + 2 << ") = " << WeightRow << " * " << WeightColumn << endl;
    if (ShowWeightsDetail) {
      Weights[Index].show();
    }
  }

  cout << "========================================" << endl;
}

/**
  Initialize weight matrix between each layers based on Layout.

**/
void FullyConnectedNetwork::WeightsMatrixInit (
  bool  RandomizeWeights
  )
{
  //
  // Create weight matrix between each layers according to Layout.
  // Each weight matrix is Row(next layer's node number) * Column(current layer's node number).
  //
  DEBUG_LOG ("Initialize weight matrix between each layers...");
  for(int Index = 0; Index < (int)Layout.size() - 1; Index++) {
    DEBUG_LOG (Layout[Index + 1] << ' ' << Layout[Index]);

    matrix  Weight(Layout[Index + 1], Layout[Index]);
    Weights.push_back(Weight);
  }

  if (RandomizeWeights) {
    FullyConnectedNetwork::WeightsRandomize();
  }
}

/**
  Initialize weights of the network randomly with values between -1.0 and 1.0.

**/
void FullyConnectedNetwork::WeightsRandomize ()
{
  int MiddleLayers = Weights.size();
  int Row, Column;
  double RandNum;

  for(int Index = 0; Index < MiddleLayers; Index++) {
    Row    = Weights[Index].getrow();
    Column = Weights[Index].getcolumn();

    for(int RowIdx = 0; RowIdx < Row; RowIdx++) {
      for(int ColumnIdx = 0; ColumnIdx < Column; ColumnIdx++) {
        RandNum = RandValue();
        Weights[Index].SetValue(RowIdx, ColumnIdx, RandNum);
      }
    }
  }
}

/**
  Generate a random double value between -1.00 and 1.00.

  @return A random double value between -1.00 and 1.00.

**/
double FullyConnectedNetwork::RandValue()
{
  double Value;
  bool Sign;

  //
  // Randomly generate a value be 0.00 to 1.00, then randomly assign a sign.
  //
  Value = (double)(rand() % 101) / 100.0;
  Sign  = (bool)(rand() % 2);

  return (Sign ? (-1.0) * Value : Value);
}

/**
  Perturb weights of the network by adding 0.2 to each weight.

**/
void FullyConnectedNetwork::PerturbWeight()
{
  int Row;
  int Column;

  DEBUG_LOG ("*SHAKE!!! (Perturb Weights)*");

  for(int Index = 0; Index < (int)Weights.size(); Index++) {
    Row    = Weights[Index].getrow();
    Column = Weights[Index].getcolumn();
    matrix AddIn (Row, Column, 0.2);
    Weights[Index] = add (Weights[Index], AddIn);
  }
}

/**
  Get the weight matrix of a specific layer.

  @param  Layer  An unsigned integer representing the layer index.
                 Layer index corresponds to the weight matrix between
                 layer Layer and layer Layer + 1.

  @return A matrix representing the weight values of the specified layer.

  @throw std::runtime_error if the Layer index is out of range.
**/
matrix
FullyConnectedNetwork::GetWeightByLayer (
  unsigned int  Layer
  ) const
{
  if (Layer >= (unsigned int)Weights.size()) {
    DEBUG_LOG ("Layer: " << Layer << ", Weights size: " << Weights.size());
    throw std::runtime_error("Error: Layer index out of range in GetWeightByLayer().");
  }

  return Weights[Layer];
}

/**
  Update the weight matrix of a specific layer.

  @param  Layer        An unsigned integer representing the layer index.
                       Layer index corresponds to the weight matrix between
                       layer Layer and layer Layer + 1.
  @param  DeltaWeight  A matrix representing the delta weight values to be added to the specified layer.

  @throw std::runtime_error if the Layer index is out of range.
**/
void
FullyConnectedNetwork::UpdateWeight (
  unsigned int  Layer,
  const matrix  &DeltaWeight
  )
{
  if (Layer >= (unsigned int)Weights.size()) {
    DEBUG_LOG ("Layer: " << Layer << ", Weights size: " << Weights.size());
    throw std::runtime_error("Error: Layer index out of range in UpdateWeightByLayer().");
  }

  Weights[Layer] = add (Weights[Layer], DeltaWeight);
}

/**
  Update the weight matrix of a all layers.

  @param  DeltaWeights  A vector of matrices representing the delta weight values to be added to each layer.

  @throw std::runtime_error  If the number of layers in DeltaWeights and Weights are different.

**/
void
FullyConnectedNetwork::UpdateWeight (
  const vector<matrix>  &DeltaWeights
  )
{
  if (DeltaWeights.size() != Weights.size()) {
    DEBUG_LOG ("Layer count of DeltaWeights = " << DeltaWeights.size() << " , Weights = " << Weights.size());
    throw runtime_error ("Layer count of DeltaWeights and Weights are different. Failed to update weight");
  }

  for (unsigned int LayerIdx = 0; LayerIdx < (unsigned int)Weights.size(); LayerIdx++) {
    Weights[LayerIdx] = add (Weights[LayerIdx], DeltaWeights[LayerIdx]);
  }
}

/**
  Get the layout of the fully connected network.

  @return A vector of unsigned integers representing the number of nodes in each layer.

**/
vector<unsigned int>
FullyConnectedNetwork::GetLayout () const
{
  return Layout;
}

/**
  Get the activation type of the fully connected network.

  @return An ACTIVATION_TYPE enum representing the activation function type used in the network.

**/
ACTIVATION_TYPE
FullyConnectedNetwork::GetActivationType (
  void
  ) const
{
  return ActivationType;
}

/**
  Perform the forward pass of the fully connected network.

  @param  InputData  A matrix representing the input data to the network.
  @param  Context    A ComputationContext object to store the activations during the forward pass.

**/
void
FullyConnectedNetwork::Forward (
  const  matrix              &InputData,
         ComputationContext  &Context
  )
{
  if (InputData.getrow() != Layout[0] || InputData.getcolumn() != 1) {
    DEBUG_LOG ("InputData size: " << InputData.getrow() << " * " << InputData.getcolumn()
               << ", Expected size: " << Layout[0] << " * 1");
    throw runtime_error ("Input data size does not match input layer size.");
  }

  unsigned int  LayerCount = Layout.size();

  //
  // Set input layer activation
  //
  Context.SetActivationByLayer (0, InputData);

  //
  // Forward pass through each layer
  //
  for (unsigned int LayerIdx = 0; LayerIdx < LayerCount - 1; LayerIdx++) {
    matrix  CurrentLayerActivation = Context.GetActivationByLayer (LayerIdx);
    matrix  CurrentWeights         = GetWeightByLayer (LayerIdx);

    matrix  Z = multiply (CurrentWeights, CurrentLayerActivation);

    ACTIVATION_FUNC  ActivationFunction = GetActivationFunction (ActivationType);

    Context.SetActivationByLayer (
              LayerIdx + 1,
              Z.ApplyElementWise (ActivationFunction)
              );
  }
}

unsigned int
FullyConnectedNetwork::Predict (
  const matrix        &InputData,
  ComputationContext  &Context
  )
{

  Forward (InputData, Context);

  matrix  OutputActivation = Context.GetActivationByLayer (Layout.size() - 1);

  unsigned int  MaxIndex = 0;
  double        MaxValue = OutputActivation.GetValue (0, 0);

  for (unsigned int Index = 1; Index < Layout.back(); Index++) {
    double  CurrentValue = OutputActivation.GetValue (Index, 0);
    if (CurrentValue > MaxValue) {
      MaxValue = CurrentValue;
      MaxIndex = Index;
    }
  }

  return MaxIndex;
}