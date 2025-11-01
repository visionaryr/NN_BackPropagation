#include "bp.h"
#include "matrix.h"
#include "FullyConnectedNetwork.h"
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

/**
  The activation function f(x) = 1 / (1 + e^(-x)).

  @param  x   input value.

  @return  The output value after applying the activation function.

**/
double 
ActivationFunction (
  double x
  )
{
  return 1 / (1 + exp((-1) * x));
}

/**
  Apply the activation function to each element of the input matrix.

  @param  Input   Input matrix.

  @return  A matrix with the same size as input, contains the output values
           after applying the activation function to each element of the input matrix.

**/
matrix
Activation (
  matrix Input
  )
{
  function<double(double)> Func = ActivationFunction;

  return Input.ApplyElementWise (Func);
}

void Learning_FP(FullyConnectedNetwork &bp, matrix input)
{
  int input_row=input.getrow();

  //put input value into a
  for(int i=0;i<input_row;i++)
  {
    bp.set_a(0,i,input.GetValue(i,0));
  }

  matrix sum,f;
  for(int i=0;i<(int)bp.Weights.size();i++)
  {
    matrix node_values(bp.Layout[i],1,bp.a[i]);
    sum=multiply( (*bp.Weights[i]) , node_values);//sum
    f=Activation(sum);//activation rule
    for(int j=0;j<f.getrow();j++)
    {
      bp.set_a(i+1,j,f.GetValue(j,0));
    }
  }
}


void delta_calc(FullyConnectedNetwork &bp, matrix desired_output)
{
  int last_layer_num = (int)bp.Layout.size()-1;
  int last_layer_size = bp.Layout.back();
  double d_output, delta_temp, a_i;
  for(int i=0;i<last_layer_size;i++)
  {
    a_i=bp.a[last_layer_num][i];
    d_output=desired_output.GetValue(i,0);
    delta_temp=(d_output - a_i)*a_i*(1-a_i);
    bp.set_delta(last_layer_num,i,delta_temp);
  }
  
  for(int i=bp.Weights.size()-1;i>0;i--)
  {
    matrix weight_T=transpose((*bp.Weights[i]));
    matrix delta_matrix(bp.Layout[i+1], 1, bp.delta[i+1]);
    matrix x = multiply(weight_T, delta_matrix);
    vector<double> x_value = x.ConvertToVector();//initialize with x
    double f_derivative, delta_value;
    for(int j=0;j<(int)x_value.size();j++)
    {
      //cout<<"x: "<<x_value[j]<<endl;
      f_derivative=bp.a[i][j]*(1-bp.a[i][j]);
      //cout<<"f\': "<<f_derivative<<endl;
      delta_value=x_value[j]*f_derivative;
      //cout<<"delta: "<<delta_value<<endl;
      bp.set_delta(i,j,delta_value);
    }
  }
  /*  
  cout<<"delta"<<endl;
  for(int i=0;i<3;i++)
  {
    for(int j=0;j<(int)bp.delta[i].size();j++)
    {
      cout<<bp.delta[i][j]<<' ';
    }
    cout<<endl;
  }
  */
  
}

vector<matrix> delta_w_calc(FullyConnectedNetwork &bp, double learning_rate)
{

  vector<matrix> delta_w;
  double delta_w_temp;
  for(int i=0;i<(int)bp.Weights.size();i++)
  {
    vector<double> delta_w_value;
    for(int j=0;j<bp.Layout[i+1];j++)
    {
      for(int k=0;k<bp.Layout[i];k++)
      {
        delta_w_temp=learning_rate*bp.delta[i+1][j]*bp.a[i][k];
        //cout<<i<<' '<<j<<' '<<k<<endl;
        //cout<<bp.delta[i+1][j]<<' '<<bp.a[i][k]<<endl;
        delta_w_value.push_back(delta_w_temp);
      }
      //cout<<endl;
    }
    matrix delta_w_layer(bp.Layout[i+1], bp.Layout[i], delta_w_value);
    delta_w.push_back(delta_w_layer);
  }
  return delta_w;
}

void upgrade_weight(FullyConnectedNetwork &bp, vector<matrix> &delta_w)
{
  vector<matrix> new_weight;
  for(int i=0;i<(int)bp.Weights.size();i++)
  {
    *bp.Weights[i] = add((*bp.Weights[i]),delta_w[i]);
  }
  //bp.Weights=new_weight;
}

double loss_func(FullyConnectedNetwork &bp, matrix &desired_output)
{
  double loss, temp;
  int n=bp.Layout.size()-1;
  for(int i=0;i<desired_output.getrow();i++)
  {
    temp=(desired_output.GetValue(i,0)-bp.a[n][i]);
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
