/**
  Data preprocessing functions implementation.

  Copyright (c) 2025, visionaryr
  Licensed under the MIT License. See the accompanying 'LICENSE' file for details.
**/

#include "BpMisc.h"

#include <vector>

using namespace std;

double
PixelBinarization (
  double x
  )
{
  return (x > 128) ? 1.0 : 0.0;
}

/**
  Binarize the pixel values in the dataset.

  This function converts pixel values in the dataset to binary values.
  Pixel values greater than 128 are set to 1, and others are set to 0.

  @param[in,out]  DataSet  The dataset containing images to be binarized.
                           Each image is represented as a vector of doubles, and pixels are in row-major order.

**/
void
DataBinarization (
  vector<matrix>  &DataSet
  )
{
  function<double(double)> Binarization = PixelBinarization;

  for(int Index = 0; Index < (int)DataSet.size(); Index++) {
    DataSet[Index].ApplyElementWise (Binarization);
  }
}