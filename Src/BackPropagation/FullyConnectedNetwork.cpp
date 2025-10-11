#include "bp.h"
#include "matrix.h"
#include "FullyConnectedNetwork.h"

#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

FullyConnectedNetwork::FullyConnectedNetwork(vector<int> &network_frame)
{
  Layout=network_frame;
  weight_Init();
  //network_rand_Init();
  Init_para();
  //test_Init();
  show_info();
}

FullyConnectedNetwork::FullyConnectedNetwork(char *filename)
{
  
  fstream inFile(filename, ios::in);
  if(!inFile)
  {
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
    weight.push_back(new_weight);
  }
  Init_para();
  show_info();
}

void FullyConnectedNetwork::show_info()
{
  int foreNodes, backNodes;
  int midlayers=weight.size();
  for(int i=0;i<midlayers;i++)
  {
    //foreNodes: weight matrix's column; backNodes: weight matrix's row
    backNodes=weight[i]->getrow();
    foreNodes=weight[i]->getcolumn();
    cout<<"layer "<<i+1<<": "<<foreNodes<<"x"<<backNodes<<endl;
    weight[i]->show();
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
    w=weight[i];
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

void FullyConnectedNetwork::network_rand_Init()
{
  matrix *w;
  int midlayers=weight.size();
  int row, column;
  double x;
  for(int i=0;i<midlayers;i++)
  {
    w=weight[i];
    row=w->getrow();
    column=w->getcolumn();
    for(int j=0;j<row;j++)
    {
      for(int k=0;k<column;k++)
      {
        x=rand_value();
        w->SetValue(j, k, x);
      }
    }
  }
}

double FullyConnectedNetwork::rand_value()
{
  double value, sign;
  value = (double)(rand()%10 + 1)/10;
  sign = rand()%2;
  if(sign==1) value*=(-1);
  return value;
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
  for(int i=0;i<(int)weight.size();i++)
  {
    row=weight[i]->getrow();
    column=weight[i]->getcolumn();
    vector<double> add_in_num(row*column,0.2);
    matrix add_in(row, column, add_in_num);
    (*weight[i])=add(*weight[i], add_in);
  }
  //show_info();
}

matrix FullyConnectedNetwork::test(matrix &input)
{
  Learning_FP(*this, input);
  matrix output(a.back().size(),1,a.back());
  return output;
}

void FullyConnectedNetwork::weight_Init()
{
  for(int i=0;i<(int)Layout.size()-1;i++)
  {
    matrix *new_weight = new matrix(Layout[i+1], Layout[i]);
    cout<<Layout[i]<<' '<<Layout[i+1]<<endl;
    weight.push_back(new_weight);
  }
  //network_rand_Init();
  test_Init();
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
  for(int i=0;i<(int)weight.size();i++)
  {
    save_weight_to_file(fs, *weight[i]);
  }
  fs.close();
}

void save_weight_to_file(fstream &fs, matrix &weight)
{
  int row = weight.getrow();
  int column = weight.getcolumn();
  for(int i=0;i<row;i++)
  {
    for(int j=0;j<column;j++)
    {
      fs<<weight.GetValue(i,j)<<',';
    }
  }
  fs<<endl;
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
