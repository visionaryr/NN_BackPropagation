#ifndef _PRE_PROCESS_H_
#define _PRE_PROCESS_H_

#include "BpMisc.h"

#include <vector>

/**
  Binarize the pixel values in the dataset.

  This function converts pixel values in the dataset to binary values.
  Pixel values greater than 128 are set to 1, and others are set to 0.

  @param[in,out]  DataSet  The dataset containing images to be binarized.
                           Each image is represented as a vector of doubles, and pixels are in row-major order.

**/
void
DataBinarization (
  std::vector<matrix>  &DataSet
  );

#endif