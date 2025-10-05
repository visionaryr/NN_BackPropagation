#ifndef _DATA_SET_H_
#define _DATA_SET_H_

#include <vector>

typedef std::vector< unsigned int >           IDX_HEADER;

typedef std::vector< double >  IMAGE;
typedef std::vector< IMAGE >   DATA_SET;
typedef std::vector< int >     LABEL_SET;

/**
  Read images and labels from the MNIST dataset files.

  This function reads images and their corresponding labels from the MNIST data set files.
  Only images with labels specified in LabelsToRead are kept.

  @param[out]  DataSet       The vector to store the read images.
                             Each image is represented as a vector of doubles, and pixels are in row-major order.
  @param[out]  LabelSet      The vector to store the corresponding labels for the images in DataSet.
  @param[in]   LabelsToRead  The vector of labels to be read. Only images with these labels will be kept.

  @throw  runtime_error  One of the following conditions is met:
                          * No labels are specified in LabelsToRead.
                          * Too many labels are specified in LabelsToRead.
                          * The image file cannot be opened or has an invalid header.
                          * The label file cannot be opened or has an invalid header.
                          * The total number of images does not match the amount number of all labels.

**/
void
ReadMNIST_and_label (
  DATA_SET          &DataSet,
  LABEL_SET         &LabelSet,
  std::vector<int>  &LabelsToRead
  );

#endif