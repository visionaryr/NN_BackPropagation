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
  Init_para();
  ShowInfo (false);
}

/**
  Constructor to initialize a fully connected network with given filename which contains all weights of each layer.
  The detail information of the network will be shown after initialization.

  @param  filename  A character pointer representing the name of the file containing network weights.

**/
FullyConnectedNetwork::FullyConnectedNetwork(char *filename)
{
  fstream inFile(filename, ios::in);
  if(!inFile) {
    cerr<<"File opening error!"<<endl;
    exit(1);
  }

  string line, token;  
  getline(inFile, line);
  istringstream delim(line);
  while (getline(delim, token, ','))        
    {
        Layout.push_back(stoi(token));                                          
    }
    
  size_t offset;//offset
  double ww;
  for(int i=0;i<(int)Layout.size()-1;i++)
  {
    getline(inFile, line);
    
    istringstream delim(line);
    vector<double> w;
    while(getline(delim, token, ','))
    {
      ww = stod(token, &offset);
      w.push_back(ww);
    }
    matrix *new_weight = new matrix(Layout[i+1], Layout[i], w);
    Weights.push_back(new_weight);
  }
  Init_para();
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
    WeightRow    = Weights[Index]->getrow();
    WeightColumn = Weights[Index]->getcolumn();

    cout<<"  Weight (L" << Index + 1 << " <-> L" << Index + 2 << ") = " << WeightRow << " * " << WeightColumn << endl;
    if (ShowWeightsDetail) {
      Weights[Index]->show();
    }
  }
}

void FullyConnectedNetwork::test_Init()
{
  fstream fs("weight.txt",ios::in);
  if(!fs) { cerr<<"File error"<<endl; exit(1);}
  int No, row, column;
  double x;
  matrix *w;
  fs>>No;
  for(int i=0;i<No;i++)
  {
    w=Weights[i];
    row=w->getrow();
    column=w->getcolumn();
    for(int j=0;j<row;j++)
    {
      for(int k=0;k<column;k++)
      {
        fs>>x;
        w->SetValue(j, k, x);
      }
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
  matrix *Weight;

  //
  // Create weight matrix between each layers according to Layout.
  // Each weight matrix is Row(next layer's node number) * Column(current layer's node number).
  //
  for(int Index = 0; Index < (int)Layout.size() - 1; Index++) {
    DEBUG_LOG (Layout[Index + 1] << ' ' << Layout[Index]);

    Weight = new matrix(Layout[Index + 1], Layout[Index]);
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
  matrix *Weight;
  int MiddleLayers = Weights.size();
  int Row, Column;
  double RandNum;

  for(int Index = 0; Index < MiddleLayers; Index++) {
    Weight = Weights[Index];
    Row    = Weight->getrow();
    Column = Weight->getcolumn();

    for(int RowIdx = 0; RowIdx < Row; RowIdx++) {
      for(int ColumnIdx = 0; ColumnIdx < Column; ColumnIdx++) {
        RandNum = RandValue();
        Weight->SetValue(RowIdx, ColumnIdx, RandNum);
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

void FullyConnectedNetwork::Init_para()
{
  vector<double> temp;
  for(int i=0;i<(int)Layout.size();i++)
  {
    temp.assign(Layout[i],0);
    a.push_back(temp);
    delta.push_back(temp);
  }
}


void FullyConnectedNetwork::set_a(int layer, int num, double value)
{
  a[layer][num]=value;
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
    row=Weights[i]->getrow();
    column=Weights[i]->getcolumn();
    vector<double> add_in_num(row*column,0.2);
    matrix add_in(row, column, add_in_num);
    (*Weights[i])=add(*Weights[i], add_in);
  }
  //show_info();
}

matrix FullyConnectedNetwork::test(matrix &input)
{
  Learning_FP(*this, input);
  matrix output(a.back().size(),1,a.back());
  return output;
}

void FullyConnectedNetwork::save_network()
{
  string filename("");
  for(int i=0;i<(int)Layout.size();i++)
  {
    filename+=to_string(Layout[i]);
    if(i!=(int)Layout.size()-1) filename+="_";
    else filename+=".txt";
  }
  cout<<filename<<endl;
  fstream fs(filename, ios::out);
  if(!fs)
  {
    cerr<<"file opening error!"<<endl;
    exit(1);
  }
  //write Layout
  for(int i=0;i<(int)Layout.size();i++)
  {
    fs<<Layout[i]<<',';
  }
  fs<<endl;
  
  //write weight
  for(int i=0;i<(int)Weights.size();i++)
  {
    WriteWeightMatrixToFile(fs, *Weights[i]);
  }
  fs.close();
}

void FullyConnectedNetwork::print_a()
{
  for(int i=0;i<(int)a.size();i++)
  {
    for(int j=0;j<(int)a[i].size();j++)
    {
      cout<<a[i][j]<<' ';
    }
    cout<<endl;
  }
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
