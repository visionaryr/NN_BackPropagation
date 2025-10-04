#include "dataset.h"

#include <vector>

using namespace std;

/**
  Binarize the pixel values in the dataset.

  This function converts pixel values in the dataset to binary values.
  Pixel values greater than 128 are set to 1, and others are set to 0.

  @param[in,out]  DataSet  The dataset containing images to be binarized.
                           Each image is represented as a vector of doubles, and pixels are in row-major order.

**/
void
Binarization (
  DATA_SET  &DataSet
  )
{
  vector<double> Image;

  for(int Index = 0; Index < (int)DataSet.size(); Index++) {
    Image = DataSet[Index];

    for(int PixelIndex = 0; PixelIndex < (int)Image.size(); PixelIndex++) {
      Image[PixelIndex] = (Image[PixelIndex] > 128) ? 1 : 0;
    }
  }
}