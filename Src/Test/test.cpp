#include "bp.h"
#include "network.h"
#include "matrix.h"
#include "PreProcess.h"
#include "MnistDataSet.h"

#include <string>
#include <cmath>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <fstream>

#define test_images 7001

using namespace std;

double predict(vector< vector<double> > &In, vector< vector<double> > &Out, int start, int amount, network &N1, vector<int> &training_cat)
{
  int correct=0;
  int i;
  /*
  for(int i=start+1;i<start+amount;i++)
  {
    matrix test_input((int)In[i].size(),1,In[i]);
    DumpMNISTImage (test_input);
    matrix ANS = N1.test(test_input);
    vector<double> ANS_vec = ANS.ConvertToVector ();
    int ans = ConvertOutputVectorToValue (ANS_vec, training_cat);
    cout<<"ANS: "<<ans<<endl;
    if(ans==Out[i][0]) correct++;
  }
  */
  for(int ri=1;ri<amount+1;ri++)
  {
    i=(rand()%5)*4000+ri;
    matrix test_input((int)In[i-1].size(),1,In[i-1]);
    
    DumpMNISTImage (test_input);
    matrix ANS = N1.test(test_input);
    vector<double> ANS_vec = ANS.ConvertToVector ();
    int ans = ConvertOutputVectorToValue (ANS_vec, training_cat);
    //cout<<"ANS: "<<ans<<endl;
    //fs<<counter<<' '<<ans<<endl;
    if(ans==Out[i-1][0]) correct++;
  }
  return (double)correct/amount;
}

void predict_to_txt(vector< vector<double> > &In, int amount, network &N1, vector<int> &training_cat)
{
  ofstream fs("410685036.txt", ios::out);
  int zeros;
  for(int i=1;i<amount+1;i++)
  {
    zeros = (int)(3-floor(log10(i)));
    string counter("");
    for(int j=0;j<zeros;j++) { counter+="0"; }
    counter+=to_string(i);
    matrix test_input((int)In[i-1].size(),1,In[i-1]);
    DumpMNISTImage (test_input);
    matrix ANS = N1.test(test_input);
    vector<double> ANS_vec = ANS.ConvertToVector ();
    int ans = ConvertOutputVectorToValue (ANS_vec, training_cat);
    cout<<"ANS: "<<ans<<endl;
    fs<<counter<<' '<<ans<<endl;
  }
  fs.close();
}

/*
int main()
{
  int cat[3]={0,3,5};//training categories
  vector<int> training_cat(cat, cat+sizeof(cat)/sizeof(int));
  
  vector< vector<double> > In;
  vector< vector<double> > Out;
  ReadMNIST_and_label(60000,784,In,Out,training_cat);
  network N1("784_15_3.txt");
  cout<<"*Accuracy: "<<predict(In, Out, test_images, 1000, N1, training_cat)<<endl;
  return 0;
}
*/

void get_test_images(vector< vector<double> > & arr)
{
  int zeros;
  for(int i=1;i<5001;i++)
  {
    vector<double> arr_temp;
    string filename_img("Testing data/");
    string counter("");
    zeros = (int)(3-floor(log10(i)));
    for(int j=0;j<zeros;j++) { counter+="0"; }
    counter+=to_string(i);
    cout<<counter<<endl;
    filename_img = filename_img + "/" + counter + ".png";
    cout<<i<<": "<<filename_img<<endl;
    char *filename = new char[filename_img.length()+1];
    strcpy(filename, filename_img.c_str());
    if(!read_png_file(filename, arr_temp)) {cerr<<"No more file"<<endl; continue;}
    
    //get_png_file(arr_temp);
    arr.push_back(arr_temp);
  }
  Binarization(arr);
  
}

/*
int main()
{
  int cat[10]={1,2,3,4,5,6,7,8,9,0};//training categories
  vector<int> training_cat(cat, cat+sizeof(cat)/sizeof(int));
  
  vector< vector<double> > In;
  vector< vector<double> > Out;
  ReadMNIST_and_label(60000,784,In,Out,training_cat);
  network N1("784_15_10.txt");
  cout<<"*Accuracy: "<<predict(In, Out, test_images, 2000, N1, training_cat)<<endl;
  
  return 0;
}
*/

int main()
{
  int cat[5]={2,4,5,6,9};
  vector<int> training_cat(cat, cat+sizeof(cat)/sizeof(int));
  
  vector< vector<double> > In;
  vector< vector<double> > Out;
  get_test_images(In);
  network N1("784_15_5.txt");
  predict_to_txt(In, 5000, N1, training_cat);
  //cout<<"predict: "<<predict(In, Out, 1000, 5000, N1, training_cat)<<endl;
}


