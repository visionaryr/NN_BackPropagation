#include "BackPropagator.h"
#include "matrix.h"
#include "FullyConnectedNetwork.h"
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

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
