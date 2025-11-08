#include "bp.h"
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
  vector<int> &NetworkFrame
  )
{
  Layout = NetworkFrame;

  WeightsMatrixInit(true);
  InitNodeActivation ();
  InitNodeDelta ();
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

  InitNodeActivation ();
  InitNodeDelta ();

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

  cout << "===== Fully Connected Network Info =====" << endl;
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
  Initialize node values of each layer to zero.

**/
void
FullyConnectedNetwork::InitNodeActivation ()
{
  for(int Index = 0; Index < (int)Layout.size(); Index++) {
    matrix LayerNodes(Layout[Index],1);
    NodeActivation.push_back(LayerNodes);
  }
}

/**
  Initialize delta values of each layer to zero.

**/
void
FullyConnectedNetwork::InitNodeDelta ()
{
  for(int Index = 0; Index < (int)Layout.size(); Index++) {
    matrix LayerDelta(Layout[Index],1);
    NodeActivation.push_back(LayerDelta);
  }
}

/**
  Set the value of a specific node in a specific layer.

  @param  Layer   An unsigned integer representing the layer index.
  @param  Number  An unsigned integer representing the node index within the layer.
  @param  Value   A double representing the value to be set for the specified node.

  @throw std::runtime_error if the Layer or Number index is out of range.

**/
void
FullyConnectedNetwork::SetNodeActivation (
  unsigned int  Layer,
  unsigned int  Number,
  double        Value
  )
{
  if (Layer >= (unsigned int)Layout.size()) {
    DEBUG_LOG ("Layer: " << Layer << ", Layout size: " << Layout.size());
    throw std::runtime_error("Error: Layer index out of range in SetNodeValue().");
  }
  if (Number >= (unsigned int)Layout[Layer]) {
    DEBUG_LOG ("Number: " << Number << ", Nodes in layer: " << Layout[Layer]);
    throw std::runtime_error("Error: Node number index out of range in SetNodeValue().");
  }

  NodeActivation[Layer].SetValue (Number, 0, Value);
}

/**
  Get the activation matrix of a specific layer.

  @param  Layer  An unsigned integer representing the layer index.

  @return A matrix representing the activation values of the specified layer.

  @throw std::runtime_error if the Layer index is out of range.

**/
matrix
FullyConnectedNetwork::GetActivationByLayer (
  unsigned int  Layer
  ) const
{
  if (Layer >= (unsigned int)Layout.size()) {
    DEBUG_LOG ("Layer: " << Layer << ", Layout size: " << Layout.size());
    throw std::runtime_error("Error: Layer index out of range in GetNodeActivation().");
  }

  return NodeActivation[Layer];
}

/**
  Set the delta value of a specific node in a specific layer.

  @param  Layer   An unsigned integer representing the layer index.
  @param  Number  An unsigned integer representing the node index within the layer.
  @param  Delta   A double representing the delta value to be set for the specified node.

  @throw std::runtime_error if the Layer or Number index is out of range.

**/
void FullyConnectedNetwork::SetNodeDelta (
  unsigned int Layer,
  unsigned int Number,
  double       Delta
  )
{
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

void FullyConnectedNetwork::PrintActivationInLayer (
  unsigned int  Layer
  )
{
  if (Layer >= (unsigned int)Layout.size()) {
    DEBUG_LOG ("Layer: " << Layer << ", Layout size: " << Layout.size());
    throw std::runtime_error("Error: Layer index out of range in PrintNodesInLayer().");
  }

  cout << "Nodes in layer" << Layer << ":" << endl;
  NodeActivation[Layer].show();
}

void FullyConnectedNetwork::print_delta()
{
  for(int i=0;i<(int)NodeDelta.size();i++)
  {
    cout<<"Delta in layer "<<i<<":"<<endl;
    NodeDelta[i].show();
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

  @param  Layer      An unsigned integer representing the layer index.
                     Layer index corresponds to the weight matrix between
                     layer Layer and layer Layer + 1.
  @param  NewWeight  A matrix representing the new weight values to be set for the specified layer.

  @throw std::runtime_error if the Layer index is out of range.
**/
void
FullyConnectedNetwork::UpdateWeightByLayer (
  unsigned int  Layer,
  const matrix  &NewWeight
  )
{
  if (Layer >= (unsigned int)Weights.size()) {
    DEBUG_LOG ("Layer: " << Layer << ", Weights size: " << Weights.size());
    throw std::runtime_error("Error: Layer index out of range in UpdateWeightByLayer().");
  }

  Weights[Layer] = NewWeight;
}
