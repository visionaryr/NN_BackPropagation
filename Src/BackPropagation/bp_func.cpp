#include "BackPropagator.h"
#include "matrix.h"
#include "FullyConnectedNetwork.h"
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

/**
  Apply the activation function to each element of the input matrix.

  @param  Input   Input matrix.

  @return  A matrix with the same size as input, contains the output values
           after applying the activation function to each element of the input matrix.

**/
// matrix
// Activation (
//   matrix Input
//   )
// {
//   function<double(double)> Func = ActivationFunction;

//   return Input.ApplyElementWise (Func);
// }

void Learning_FP(FullyConnectedNetwork &bp, matrix input)
{
  int input_row=input.getrow();

  //put input value into a
  for(int i=0;i<input_row;i++)
  {
    bp.SetNodeValue(0,i,input.GetValue(i,0));
  }

  matrix sum,f;
  for(int i=0;i<(int)bp.Weights.size();i++)
  {
    matrix node_values = bp.NodeValue[i];
    sum=multiply(bp.Weights[i], node_values);//sum
    f=Activation(sum);//activation rule
    for(int j=0;j<f.getrow();j++)
    {
      bp.SetNodeValue(i+1,j,f.GetValue(j,0));
    }
  }
}

double loss_func(FullyConnectedNetwork &bp, matrix &desired_output)
{
  double loss, temp;
  int n=bp.Layout.size()-1;
  for(int i=0;i<desired_output.getrow();i++)
  {
    temp=(desired_output.GetValue(i,0) - bp.NodeValue[n].GetValue (i, 0));
    loss+=pow(temp,2.0);
    //cout<<"temp:"<<temp<<" loss:"<<loss<<endl;

  }

  loss/=desired_output.getrow();
  return loss;
}

vector<matrix> BatchMode_sum(vector<matrix> &to_add, vector<matrix> &sum)
{
  vector<matrix> new_sum;
  for(int i=0;i<(int)to_add.size();i++)
  {
    new_sum.push_back( add(to_add[i], sum[i]) );
  }
  return new_sum;
}

vector<matrix> BatchMode_Init(vector<int> &Layout)
{
  vector<matrix> Init;
  for(int i=0;i<(int)Layout.size()-1;i++)
  {
    vector<double> AA(Layout[i+1]*Layout[i], 0.0);
    matrix A(Layout[i+1], Layout[i], AA);
    Init.push_back(A);
  }
  return Init;
}

vector<matrix> BatchMode_average(vector<matrix> &sum, double train_images)
{
  vector<matrix> average;
  for(int i=0;i<(int)sum.size();i++)
  {
    average.push_back( multiplyBy(sum[i], 1/train_images) );
  }
  return average;
}
