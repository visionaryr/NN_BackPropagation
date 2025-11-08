#ifndef _BACK_PROPAGATION_MISC_H_
#define _BACK_PROPAGATION_MISC_H_

#include "BackPropagator.h"

#include <vector>

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
  );

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
  std::vector<double>  &OutputVector,
  LABELS               &TrainLabels
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
  std::vector<int>  &VectorToSearch,
  int               Value
  );

#endif