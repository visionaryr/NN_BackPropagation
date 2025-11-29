#include <iostream>
#include <vector>
#include <fstream>
#include <stdio.h>

#include "BpMisc.h"
 
using namespace std;

/**
  Check if a value is present in a vector.

  @param[in]  VectorToSearch  The vector to be searched.
  @param[in]  Value           The value to search for.
  @param[out] IndexInVector   The Index of the Value in VectorToSearch.

  @retval  true   The value is found in VectorToSearch.
  @retval  false  The value is not found in VectorToSearch.

**/
bool
ValueInVector (
  vector<unsigned int>   &VectorToSearch,
  unsigned int           Value,
  unsigned int           *IndexInVector
  )
{
  for (unsigned int Index = 0; Index < (unsigned int)VectorToSearch.size(); Index++) {
    if (VectorToSearch[Index] == Value) {
      if (IndexInVector != NULL) {
        *IndexInVector = Index;
      }
      return true;
    }
  }

  return false;
}

/**
  Check if a value is present in a vector.

  @param[in]  VectorToSearch  The vector to be searched.
  @param[in]  Value           The value to search for.

  @retval  true   The value is found in VectorToSearch.
  @retval  false  The value is not found in VectorToSearch.

**/
bool
ValueInVector (
  vector<unsigned int>   &VectorToSearch,
  unsigned int           Value
  )
{
  return ValueInVector (VectorToSearch, Value, NULL);
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

  @return      The binary matrix(Row, Column) = (TrainLables Count, 1) representation of the desired output label.

  @throw       runtime_error      One of the following conditions is met:
                                    * No training labels are specified in TrainLabels.
                                    * Too many training labels are specified in TrainLabels.
                                    * The desired output label is not in TrainLabels.
**/
matrix
ConvertOutputValueToMatrix (
  int                   DesireOutputLabel,
  vector<unsigned int>  &TrainLabels
  )
{
  unsigned int  Index;

  if (TrainLabels.size() == 0 || TrainLabels.size() > 10) {
    throw runtime_error ("Error: Training labels are invalid.");
  }

  if (!ValueInVector (TrainLabels, DesireOutputLabel, &Index)) {
    throw runtime_error ("Error: Desire output label is not in training labels.");
  }

  matrix DesireOutputMatrix (TrainLabels.size(), 1);

  //
  // Set value to the field where (Row, Column) = (Index, 0)
  // *Note: The first one is index 0.
  //
  DesireOutputMatrix.SetValue (Index, 0, 1.0);

  return DesireOutputMatrix;
}