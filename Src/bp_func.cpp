#include "bp.h"
#include "matrix.h"
#include "network.h"
#include <vector>
#include <iostream>
#include <cmath>

#define PAUSE printf("Press Enter key to continue..."); fgetc(stdin);

using namespace std;

matrix activation_f(matrix sum)
{
	int row=sum.getrow();
	int column=sum.getcolumn();
	double sum_value, f_value;
	matrix f(row,column);
	for(int i=0;i<row;i++)
	{
		sum_value=sum.getMatrix(i,0);
		f_value=1/(1+exp((-1)*sum_value));
		//cout<<exp((-1)*sum_value)<<endl;
		f.SetValue(i,0,f_value);
	}
	return f;
}

void Learning_FP(network &bp, matrix input)
{
	int input_row=input.getrow();

	//put input value into a
	for(int i=0;i<input_row;i++)
	{
		bp.set_a(0,i,input.getMatrix(i,0));
	}

	matrix sum,f;
	for(int i=0;i<(int)bp.weight.size();i++)
	{
		matrix node_values(bp.nodes[i],1,bp.a[i]);
		sum=multiply( (*bp.weight[i]) , node_values);//sum
		f=activation_f(sum);//activation rule
		for(int j=0;j<f.getrow();j++)
		{
			bp.set_a(i+1,j,f.getMatrix(j,0));
		}
	}
}


void delta_calc(network &bp, matrix desired_output)
{
	int last_layer_num = (int)bp.nodes.size()-1;
	int last_layer_size = bp.nodes.back();
	double d_output, delta_temp, a_i;
	for(int i=0;i<last_layer_size;i++)
	{
		a_i=bp.a[last_layer_num][i];
		d_output=desired_output.getMatrix(i,0);
		delta_temp=(d_output - a_i)*a_i*(1-a_i);
		bp.set_delta(last_layer_num,i,delta_temp);
	}
	
	for(int i=bp.weight.size()-1;i>0;i--)
	{
		matrix weight_T=transpose((*bp.weight[i]));
		matrix delta_matrix(bp.nodes[i+1], 1, bp.delta[i+1]);
		matrix x = multiply(weight_T, delta_matrix);
		vector<double> x_value = x.convert_to_vector();//initialize with x
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

vector<matrix> delta_w_calc(network &bp, double learning_rate)
{

	vector<matrix> delta_w;
	double delta_w_temp;
	for(int i=0;i<(int)bp.weight.size();i++)
	{
		vector<double> delta_w_value;
		for(int j=0;j<bp.nodes[i+1];j++)
		{
			for(int k=0;k<bp.nodes[i];k++)
			{
				delta_w_temp=learning_rate*bp.delta[i+1][j]*bp.a[i][k];
				//cout<<i<<' '<<j<<' '<<k<<endl;
				//cout<<bp.delta[i+1][j]<<' '<<bp.a[i][k]<<endl;
				delta_w_value.push_back(delta_w_temp);
			}
			//cout<<endl;
		}
		matrix delta_w_layer(bp.nodes[i+1], bp.nodes[i], delta_w_value);
		delta_w.push_back(delta_w_layer);
	}
	return delta_w;
}

void upgrade_weight(network &bp, vector<matrix> &delta_w)
{
	vector<matrix> new_weight;
	for(int i=0;i<(int)bp.weight.size();i++)
	{
		*bp.weight[i] = add((*bp.weight[i]),delta_w[i]);
	}
	//bp.weight=new_weight;
}

double loss_func(network &bp, matrix &desired_output)
{
	double loss, temp;
	int n=bp.nodes.size()-1;
	for(int i=0;i<desired_output.getrow();i++)
	{
		temp=(desired_output.getMatrix(i,0)-bp.a[n][i]);
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

vector<matrix> BatchMode_Init(vector<int> &nodes)
{
	vector<matrix> Init;
	for(int i=0;i<(int)nodes.size()-1;i++)
	{
		vector<double> AA(nodes[i+1]*nodes[i], 0.0);
		matrix A(nodes[i+1], nodes[i], AA);
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
