#include "matrix.h"
#include "BackPropagator.h"
#include "FullyConnectedNetwork.h"
#include "PreProcess.h"
#include "PngIo.h"

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <stdio.h>
#include <queue>
#include <cmath>
#include <iomanip>
#include <set>
#include <cstring>

#define ARRAY_SIZE(Array) \
  (sizeof(Array) / sizeof(Array[0]))

#define PAUSE printf("Press Enter key to continue..."); fgetc(stdin);
#define train_start 0
//#define train_images 60
#define test_images 0
#define last_save 15

using namespace std;

int mTrainingCategories[] = {
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9
};

int mNetworkLayout[] = {
  784, 15, 10
};

// void get_train_images(vector< vector<double> > & arr, vector< vector<double> > & vec, vector<int> &training_cat)
// {
//   int zeros;
//   for(int cat=0;cat<(int)training_cat.size();cat++)
//   {
//     for(int i=1;i<4001;i++)
//     {
//       vector<double> arr_temp;
//       vector<double> vec_temp(1,0);
//       string filename_img("Training data/");
//       string counter("");
//       zeros = (int)(3-floor(log10(i)));
//       for(int j=0;j<zeros;j++) { counter+="0"; }
//       counter+=to_string(i);
//       cout<<counter<<endl;
//       filename_img = filename_img + to_string(training_cat[cat]) + "/" + counter + ".png";
//       cout<<i<<": "<<filename_img<<endl;
//       char *filename = new char[filename_img.length()+1];
//       strcpy(filename, filename_img.c_str());
//       if(!read_png_file(filename, arr_temp)) {cerr<<"No more file"<<endl; continue;}
    
//       //get_png_file(arr_temp);
//       arr.push_back(arr_temp);
//       vec_temp[0]=training_cat[cat];
//       vec.push_back(vec_temp);
//     }
//   }
//   DataBinarization(arr);
  
// }

vector<matrix>
ConvertLabelsToNetworkOutput (
  LABELS       &LabelSet,
  vector<int>  &TrainingLabels
  )
{
  vector<matrix>  DesiredOutputs;

  for (unsigned int Index = 0; Index < (unsigned int)LabelSet.size(); Index ++) {
    matrix  DesiredOutput = ConvertOutputValueToMatrix (
                              LabelSet[Index],
                              TrainingLabels
                              );
    DesiredOutputs.push_back (DesiredOutput);
  }

  return DesiredOutputs;
}

/**

**/
int
main (
  void
  )
{
  DATA_SET        DataSet;
  LABELS          LabelSet;
  vector<matrix>  DesiredOutputs;

  //
  // Initialize random generator.
  //
  srand (time (NULL));

  //
  // Init categories to be trained.
  //
  vector<int> TrainingCategories (mTrainingCategories, mTrainingCategories + ARRAY_SIZE (mTrainingCategories));

  //
  // Get trainning data set.
  // Convert LabelSet to matrix format to match with network output.
  //
  ReadMNIST_and_label (DataSet, LabelSet, TrainingCategories);
  DesiredOutputs = ConvertLabelsToNetworkOutput (LabelSet, TrainingCategories);

  //
  // Initialize network, here we use Fully Connected Network(FCN)
  //
  NETWORK_LAYOUT  Layout (mNetworkLayout, mNetworkLayout + ARRAY_SIZE (mNetworkLayout));
  FullyConnectedNetwork  FCN (Layout);

  //
  // Initialize trainning algorithm, here we use Back Propagation.
  //
  BackPropagator  TrainingAlgoBp (FCN);

  TrainingAlgoBp.Train (
    DataSet,         // Input data
    DesiredOutputs,  // Desired Output
    0.1,             // Learning rate
    15,              // Epochs
    0.05             // Target Loss
  );
}

// int main()
// {
//   srand( time(NULL) );
//   int cat[5]={2,4,5,6,9};
//   //int cat[10]={1,2,3,4,5,6,7,8,9,0};//training categories
//   vector<int> training_cat(cat, cat+sizeof(cat)/sizeof(int));

//   vector< vector<double> > In;
//   vector< vector<double> > Out;
//   //ReadMNIST_and_label(60000,784,In,Out,training_cat);
//   get_train_images(In, Out, training_cat);
//   //read_Mnist_Label(Out);
//   cout<<In.size()<<' '<<Out.size()<<endl;
//   int train_images = In.size();
  
//   double loss=1;
//   int epoch=1;
//   double sum, var, mean, sd, learning_rate;
//   //int aa[3]={2,2,3};
  
//   int aa[3]={784,15,5};
//   /*
//   int bb[3]={784,30,10};
//   int cc[3]={784,6,10};
//   int dd[3]={784,3,10};
//   int ee[3]={784,256,10};
//   int ff[4]={784,100,15,10};
//   */
//   //int gg[5]={784,20,20,20,10};

//   vector<int> network_frame_1(aa,aa+sizeof(aa)/sizeof(int));
//   /*
//   vector<int> network_frame_2(bb,bb+sizeof(bb)/sizeof(int));
//   vector<int> network_frame_3(cc,cc+sizeof(cc)/sizeof(int));
//   vector<int> network_frame_4(dd,dd+sizeof(dd)/sizeof(int));
//   vector<int> network_frame_5(ee,ee+sizeof(ee)/sizeof(int));
//   */
//   //vector<int> network_frame_6(ff,ff+sizeof(ff)/sizeof(int));
//   //vector<int> network_frame_7(gg,gg+sizeof(gg)/sizeof(int));
//   vector< vector<int> > network_frame;
  
//   network_frame.push_back(network_frame_1);
//   /*
//   network_frame.push_back(network_frame_2);
//   network_frame.push_back(network_frame_3);
//   network_frame.push_back(network_frame_4);
//   network_frame.push_back(network_frame_5);
//   */
//   //network_frame.push_back(network_frame_6);
//   //network_frame.push_back(network_frame_7);
  
//   for(int times=0;times<(int)network_frame.size();times++)
//   {
  
//     loss=1; epoch=1;
    
//     FullyConnectedNetwork N1(network_frame[times]);//network init.
//     //N1.show_info();
//     //FullyConnectedNetwork N1("1.txt");
//     //PAUSE;
//     vector<double> last_15;
//     clock_t t_start, t_end;
//     vector<matrix> delta_w;
//     while(loss>=0.01 && epoch<=15)
//     {

//       set<int> flag;
//       int rand_i;
//       t_start=clock();
//       vector<double> loss_s;
//       //vector<matrix> delta_w_sum = BatchMode_Init(network_frame[times]);
//       vector<matrix> delta_w;
//       for(int i=train_start;i<train_images;i++)
//       //for(int i=0;i<4;i++)
//       {
//         //rand_i=i;
//         rand_i=(rand()%train_images)+train_start;
//         if(flag.find(rand_i)!=flag.end())
//         {
//           i--;
//           continue;
//         }
//         flag.insert(rand_i);
        
//         matrix input((int)In[rand_i].size(),1,In[rand_i]);
//         vector<double> output = ConvertOutputValueToVector(Out[rand_i][0], training_cat);
//         matrix d_output((int)training_cat.size(),1, output );
        
        
//         //matrix d_output(10,1, Out[rand_i] );
//         Learning_FP(N1,input);
//         loss=loss_func(N1,d_output);
//         delta_calc(N1,d_output);
//         //learning_rate=1/(1+((double)epoch-1)/160);
//         //vector<matrix> delta_w_toadd = delta_w_calc(N1, 1);
//         delta_w=delta_w_calc(N1,1);
//         //delta_w_sum = BatchMode_sum( delta_w_toadd , delta_w_sum);
//         upgrade_weight(N1, delta_w);
//         //N1.show_info();
//         loss_s.push_back(loss);
//       }
//       t_end=clock();
      
//       //loss average
//       loss=0;
//       for(int j=0;j<(int)loss_s.size();j++)
//       {
//         loss+=loss_s[j];
//       }
//       loss/=(double)loss_s.size();
//       //cout<<"average:"<<loss<<endl;
//       cout<<"Epoch "<<epoch++<<": "<<"loss="<<loss<<"  "<<(double)(t_end-t_start)/CLOCKS_PER_SEC<<"sec";
//       //cout<<" accuracy: "<<predict(In, Out, test_images, train_images, N1, training_cat)<<endl;
//       //store loss of last 15 epoch
//       if(last_15.size()<last_save) {last_15.push_back(loss); continue;}
//       else
//       {
//         last_15.erase(last_15.begin());
//         last_15.push_back(loss);
//         //N1.save_network();
//       }
//       //cout<<endl;
//       //for(int i=0;i<(int)last_15.size();i++) cout<<setw(4)<<last_15[i]<<' ';
//       //cout<<endl;
    
//       //calculate standard deviation of last 15 epoch's loss value
//       sum=0;
//       for(int n = 0; n < last_save; n++ )
//       {
//         sum += last_15[n];
//       }
//       mean=sum/last_save;
//       //cout<<"mean:"<<mean<<endl;
//       var = 0;
//       for(int n = 0; n < last_save; n++ )
//       {
//         var += (last_15[n] - mean) * (last_15[n] - mean);
//       }
//       var /= last_save;
//       sd = sqrt(var);
    
//       //If stdeva<=.008 =>loss value didn't change a lot, then shake.
//       if(sd<=.001) N1.PerturbWeight();
//       /*
//       if(epoch)
//       {
//         cout<<"Want to shake?"<<endl;
//         cin>>c;
//         if(c=="yes") N1.shake();
//       }
//       */
//       //PAUSE
    
//     }
//     N1.ExportToFile(".", "");
//   }
//   return 0;
// }

