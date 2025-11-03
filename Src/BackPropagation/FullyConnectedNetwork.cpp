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
  InitNodeValue ();
  Init_para();
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

  InitNodeValue ();

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

void
FullyConnectedNetwork::InitNodeValue()
{
  for(int Index = 0; Index < (int)Layout.size(); Index++) {
    matrix LayerNodes(Layout[Index],1);
    NodeValue.push_back(LayerNodes);
  }
}

void FullyConnectedNetwork::Init_para()
{
  vector<double> temp;
  for(int i=0;i<(int)Layout.size();i++)
  {
    temp.assign(Layout[i],0);
    delta.push_back(temp);
  }
}


void
FullyConnectedNetwork::SetNodeValue (
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

  NodeValue[Layer].SetValue (Number, 0, Value);
}

void FullyConnectedNetwork::set_delta(int layer, int num, double value)
{
  delta[layer][num]=value;
}

void FullyConnectedNetwork::shake()
{
  cout<<"*shake!*"<<endl;
  int row,column;
  for(int i=0;i<(int)Weights.size();i++)
  {
    row=Weights[i].getrow();
    column=Weights[i].getcolumn();
    vector<double> add_in_num(row*column,0.2);
    matrix add_in(row, column, add_in_num);
    (Weights[i])=add(Weights[i], add_in);
  }
  //show_info();
}

// matrix FullyConnectedNetwork::test(matrix &input)
// {
//   Learning_FP(*this, input);
//   matrix output(a.back().size(),1,a.back());
//   return output;
// }

void FullyConnectedNetwork::PrintNodesInLayer (
  unsigned int  Layer
  )
{
  if (Layer >= (unsigned int)Layout.size()) {
    DEBUG_LOG ("Layer: " << Layer << ", Layout size: " << Layout.size());
    throw std::runtime_error("Error: Layer index out of range in PrintNodesInLayer().");
  }

  cout << "Nodes in layer" << Layer << ":" << endl;
  NodeValue[Layer].show();
}

void FullyConnectedNetwork::print_delta()
{
  for(int i=0;i<(int)delta.size();i++)
  {
    for(int j=0;j<(int)delta[i].size();j++)
    {
      cout<<delta[i][j]<<' ';
    }
    cout<<endl;
  }
}
