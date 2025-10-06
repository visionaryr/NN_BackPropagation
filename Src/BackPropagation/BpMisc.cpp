#include <iostream>
#include <vector>
#include <fstream>
#include <stdio.h>

#include "BpMisc.h"
 
#define PAUSE printf("Press Enter key to continue..."); fgetc(stdin); 
 
using namespace std;

/**
  Check if a value is present in a vector.

  @param[in]  VectorToSearch  The vector to be searched.
  @param[in]  Value           The value to search for.

  @retval  true   The value is found in VectorToSearch.
  @retval  false  The value is not found in VectorToSearch.

**/
bool
ValueInVector (
  vector<int>  &VectorToSearch,
  int          Value
  )
{
  for (int Index = 0; Index < (int)VectorToSearch.size(); Index++) {
    if (VectorToSearch[Index] == Value) {
      return true;
    }
  }

  return false;
}

/**
  Get the index of the maximum value in a vector.

  @param[in]  VectorToSearch  The vector to be searched.

  @return     The index of the maximum value in VectorToSearch.
              If there are multiple maximum values, the index of the first one is returned.

**/
unsigned int
GetMaxIndex (
  vector<double>  &VectorToSearch
  )
{
  unsigned int MaxIndex = 0;
  double       MaxValue = VectorToSearch[0];

  for (unsigned int Index = 1; Index < VectorToSearch.size(); Index++) {
    if (VectorToSearch[Index] > MaxValue) {
      MaxValue = VectorToSearch[Index];
      MaxIndex = Index;
    }
  }

  return MaxIndex;
}

/**
  Convert a desired output label to a binary vector representation.

  This function converts a desired output label to a binary vector representation.
  The length of the output vector is equal to the number of training labels.
  The element corresponding to the desired label is set to 1, and all other elements are set to 0.

  @param[in]   DesireOutputLabel  The desired output label to be converted.
  @param[in]   TrainLabels        The vector of training labels.

  @return      The binary vector representation of the desired output label.

  @throw       runtime_error      One of the following conditions is met:
                                    * No training labels are specified in TrainLabels.
                                    * Too many training labels are specified in TrainLabels.
                                    * The desired output label is not in TrainLabels.
**/
std::vector<double>
ConvertOutputValueToVector (
  int     DesireOutputLabel,
  LABELS  &TrainLabels
  )
{
  if (TrainLabels.size() == 0 || TrainLabels.size() > 10) {
    throw runtime_error ("Error: Training labels are invalid.");
  }

  if (!ValueInVector (TrainLabels, DesireOutputLabel)) {
    throw runtime_error ("Error: Desire output label is not in training labels.");
  }

  vector<double> DesireOutput;
  for(int Index = 0; Index < (int)TrainLabels.size(); Index++) {
    DesireOutput.push_back ((TrainLabels[Index] == DesireOutputLabel) ? 1.0 : 0.0);
  }

  return DesireOutput;
}

/**
  Convert an output vector to the corresponding label value.

  This function converts an output vector to the corresponding label value.
  The index of the maximum value in the output vector is used to determine the label.
  The label corresponding to that index in TrainLabels is returned.

  @param[in]   OutputVector  The output vector to be converted.
  @param[in]   TrainLabels   The vector of training labels.

  @return      The label value corresponding to the maximum value in OutputVector.

  @throw       runtime_error  One of the following conditions is met:
                                * No training labels are specified in TrainLabels.
                                * Too many training labels are specified in TrainLabels.
                                * The size of OutputVector does not match the size of TrainLabels.

**/
int
ConvertOutputVectorToValue (
  vector<double>  &OutputVector,
  LABELS          &TrainLabels
  )
{
  if (OutputVector.size() != TrainLabels.size()) {
    throw runtime_error ("Error: Output vector size does not match training labels size.");
  }

  return TrainLabels[GetMaxIndex (OutputVector)];
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
