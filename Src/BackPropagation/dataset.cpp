#include <iostream>
#include <vector>
#include <fstream>
#include <stdio.h>
#include "bp.h"
 
#define PAUSE printf("Press Enter key to continue..."); fgetc(stdin); 
 
using namespace std;
int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

void ReadMNIST_and_label(int NumberOfImages, int DataOfAnImage, vector< vector<double> > &arr, vector< vector<double> > &vec, vector<int> &train_cat)
{
    //ifstream file("train-images.idx3-ubyte",ios::binary);
    ifstream In_file("train-images.idx3-ubyte", ios::binary);
    ifstream Out_file("train-labels.idx1-ubyte", ios::binary);
    if (In_file.is_open() && Out_file.is_open())
    {
        vector<double> vec_1d(1,0);
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        In_file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        In_file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        In_file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        In_file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        
        Out_file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        Out_file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        
        cout<<n_rows<<endl;
        for(int i=0;i<number_of_images;++i)
        {
        unsigned char temp = 0;
        Out_file.read((char*) &temp, sizeof(temp));
        vec_1d[0]=(double)temp;
        if( !in(train_cat,vec_1d[0]) )
        {
        unsigned char temp=0;
                for(int kk=0;kk<784;kk++) In_file.read((char*)&temp,sizeof(temp));
        continue;  
        }
        vec.push_back(vec_1d);
        vector<double> arr_temp;
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    In_file.read((char*)&temp,sizeof(temp));
                    arr_temp.push_back( (double)temp );
                }
                //cout<<arr_temp.size()<<' '<<arr.size()<<endl;
              
            }
            arr.push_back(arr_temp);
        }
        cout<<'*'<<arr.size()<<endl;
    }
    binarization(arr);
    In_file.close();
    Out_file.close();
}

/*
void read_Mnist_Label(vector< vector<double> > &vec, vector<int> &train_cat)
{
    ifstream file("train-labels.idx1-ubyte", ios::binary);
    if (file.is_open())
    {
        vector<double> vec_1d(1,0);
        cout<<"*"<<endl;
        int magic_number = 0;
        int number_of_images = 0;
        Out_file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        Out_file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        cout<<number_of_images<<endl;
        for(int i = 0; i < number_of_images; ++i)
        {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            vec_1d[0]=(double)temp;
            vec.push_back(vec_1d);
        }
        cout<<number_of_images<<endl;
    for(int i=0;i<20;i++)
    {
      cout<<vec[i][0]<<endl;
    }
    }
   
}

*/


vector<double> output_convert(int O_dataset, vector<int> &train_cat)
{
  vector<double> d_output;
  for(int i=0;i<(int)train_cat.size();i++)
  {
    if(train_cat[i]==O_dataset)
    {
      d_output.push_back(1.0);
    }
    else
      d_output.push_back(0.0);
  }
  return d_output;
}

void binarization(vector< vector<double> > &input)
{
  for(int i=0;i<(int)input.size();i++)
  {
    for(int j=0;j<784;j++)
    {
      input[i][j] = (input[i][j] > 128) ? 1 : 0;
    }
  }
}

void show_as_image(matrix &A)
{
  for(int i=0;i<28;i++)
  {
    for(int j=0;j<28;j++)
    {
      cout<<A.GetValue(28*i+j,0)<<' ';
    }
  }
}

int to_number(matrix &A, vector<int> &train_cat)
{
  int ans=0;
  for(int i=0;i<(int)train_cat.size();i++)
  {
    if(A.GetValue(i,0)>.7) ans+=train_cat[i];
  }
  if(ans>=10) return ans*(-1);
  else return ans;
}

//load simple data(0,1) for training
void load_input_output(vector< vector<double> > &I, vector< vector<double> > &O)
{
  double a[2]={1,1};
  double b[2]={0,1};
  vector<double> in(a,a+sizeof(a)/sizeof(double));
  vector<double> out(b,b+sizeof(b)/sizeof(double));

  I.push_back(in);
  O.push_back(out);
  /*
  //In:0,1; out:1,1,0
  a[0]=0; a[1]=1; b[0]=1; b[1]=1; b[2]=0;
  in.assign(a,a+sizeof(a)/sizeof(double));
  out.assign(b,b+sizeof(b)/sizeof(double));
  I.push_back(in);
  O.push_back(out);
  
  //In:1,0; out:1,0,1
  a[0]=1; a[1]=0; b[0]=1; b[1]=0; b[2]=1;
  in.assign(a,a+sizeof(a)/sizeof(double));
  out.assign(b,b+sizeof(b)/sizeof(double));
  I.push_back(in);
  O.push_back(out);
  
  //In:1,1; out:0,1,1
  a[0]=1; a[1]=1; b[0]=0; b[1]=1; b[2]=1;
  in.assign(a,a+sizeof(a)/sizeof(double));
  out.assign(b,b+sizeof(b)/sizeof(double));
  I.push_back(in);
  O.push_back(out);
*/
}

bool in(vector<int> &train_cat, double j)
{
  for(int i=0;i<(int)train_cat.size();i++)
  {
    if((int)j==train_cat[i]) return true;
  }
  return false;
}
