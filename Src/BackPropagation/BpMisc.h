/**
  Miscellaneous helper functions for BackPropagator.

  Copyright (c) 2025, visionaryr
  Licensed under the MIT License. See the accompanying 'LICENSE' file for details.
**/

#ifndef _BACK_PROPAGATION_MISC_H_
#define _BACK_PROPAGATION_MISC_H_

#include "BackPropagator.h"

#include <vector>

using namespace std;

/**
  Convert a desired output label to a binary vector representation.

  This function converts a desired output label to a binary vector representation.
  The length of the output vector is equal to the number of training labels.
  The element corresponding to the desired label is set to 1, and all other elements are set to 0.

  @param[in]   DesireOutputLabel  The desired output label to be converted.
  @param[in]   TrainLabels        The vector of training labels.

  @return      The binary matrix(Row, Column) = (10, 1) representation of the desired output label.

  @throw       runtime_error      One of the following conditions is met:
                                    * No training labels are specified in TrainLabels.
                                    * Too many training labels are specified in TrainLabels.
                                    * The desired output label is not in TrainLabels.
**/
matrix
ConvertOutputValueToMatrix (
  int                   DesireOutputLabel,
  vector<unsigned int>  &TrainLabels
  );

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
  );

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
  );

#endif