#include "BpMisc.h"
#include "MnistDataSet.h"

#include <iostream>
#include <cstring>

using namespace std;

#define  ROOT_PATH "/home/rexchen/Desktop/1092_NN/Back_Propagation/"

#define  TRAIN_IMAGES_IDX_FILE  (string)"train-images.idx3-ubyte"
#define  TRAIN_LABELS_IDX_FILE  (string)"train-labels.idx1-ubyte"

/**
  Convert a 32-bit unsigned integer from big-endian to little-endian format.

  @param[in]  BigEndian  The big-endian unsigned integer to be converted.

  @return     The converted little-endian unsigned integer.

**/
static
unsigned int
BigEndianToLittleEndianU32 (
  unsigned int BigEndian
  )
{
  return ((BigEndian & 0x000000FF) << 24) |
         ((BigEndian & 0x0000FF00) << 8)  |
         ((BigEndian & 0x00FF0000) >> 8)  |
         ((BigEndian & 0xFF000000) >> 24);
}

/**
  Open an IDX file and read its header.

  @param[in]   filename             The name of the IDX file to be opened.
  @param[in]   ExpectedMagicNumber  The expected magic number for the IDX file.
  @param[out]  Header               The header read from the IDX file.

  @return      The opened file stream.

  @throw       runtime_error        One of the following conditions is met:
                                      * File cannot be opened.
                                      * The magic number in the file is invalid.

**/
ifstream
OpenIdxFile (
  const string        &filename,
  const unsigned int  ExpectedMagicNumber,
  IDX_HEADER          &Header
  )
{
  ifstream      File;
  unsigned int  MagicNumber = 0;
  int           Dimension = 0;
  unsigned int  Value = 0;

  File.open (filename, ios::binary);
  if (!File.is_open ()) {
    throw runtime_error ("Error: Cannot open file " + filename);
  }

  File.read ((char*)&MagicNumber, sizeof(MagicNumber));
  MagicNumber = BigEndianToLittleEndianU32 (MagicNumber);

  if (MagicNumber != ExpectedMagicNumber) {
    throw runtime_error ("Error: Invalid magic number in file " + filename +
                         ". Expected " + to_string(ExpectedMagicNumber) +
                         ", got " + to_string(MagicNumber)
                        );
  }

  Dimension = (int)(MagicNumber & 0xFF);

  Header.resize (Dimension);
  for (int Index = 0; Index < Dimension; Index++) {
    File.read ((char *)&Value, sizeof(Value));
    Header[Index] = BigEndianToLittleEndianU32 (Value);
  }

  return File;
}

/**
  Read a single image from an IDX file.

  @param[in]   File            The file stream of the opened IDX file.
                               The position of the file pointer should be at the beginning of the image data.
  @param[in]   NumberOfRows    The number of rows in the image.
  @param[in]   NumberOfColumns The number of columns in the image.

  @return      The image read from the file, represented as a vector of doubles.
               Pixels are stored in row-major order.

  @throw       runtime_error   One of the following conditions is met:
                                * The file is not open or has reached the end of file.  

**/
static
vector<double>
ReadImageFromIdxToVector (
  ifstream      &File,
  unsigned int  NumberOfRows,
  unsigned int  NumberOfColumns
  )
{
  vector<double> ImageVector;
  unsigned char  Pixel = 0;

  if (!File.is_open () || File.eof ()) {
    throw runtime_error ("Error: Cannot read image from file");
  }

  ImageVector.resize (NumberOfRows * NumberOfColumns);

  for (unsigned int Row = 0; Row < NumberOfRows; Row++) {
    for (unsigned int Column = 0; Column < NumberOfColumns; Column++) {
      File.read ((char*)&Pixel, sizeof(Pixel));
      ImageVector[Row * NumberOfColumns + Column] = (double)Pixel;
    }
  }

  return ImageVector;
}

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
  DATA_SET  &DataSet,
  LABELS    &LabelSet,
  LABELS    &LabelsToRead
  )
{
  ifstream      ImagesFile;
  ifstream      LabelsFile;
  IDX_HEADER    ImagesFileHeader;
  IDX_HEADER    LabelsFileHeader;
  unsigned int  NumberOfImages = 0;
  unsigned int  NumberOfRows = 0;
  unsigned int  NumberOfColumns = 0;
  unsigned int  NumberOfLabels = 0;

  if (LabelsToRead.size() == 0) {
    throw runtime_error ("Error: No labels to read");
  }
  if (LabelsToRead.size() > 10) {
    throw runtime_error ("Error: Too many labels to read");
  }

  //
  // Clear DataSet and LabelSet.
  //
  DataSet.clear ();
  LabelSet.clear ();

  //
  // Open the image file.
  //
  ImagesFile = OpenIdxFile (
                 ROOT_PATH + TRAIN_IMAGES_IDX_FILE,
                 0x00000803,
                 ImagesFileHeader
                 );
  if (ImagesFileHeader.size() != 3) {
    throw runtime_error ("Error: Invalid image file header");
  }

  NumberOfImages  = ImagesFileHeader[0];
  NumberOfRows    = ImagesFileHeader[1];
  NumberOfColumns = ImagesFileHeader[2];

  //
  // Open the label file.
  //
  LabelsFile = OpenIdxFile (
                 ROOT_PATH + TRAIN_LABELS_IDX_FILE,
                 0x00000801,
                 LabelsFileHeader
                 );
  if (LabelsFileHeader.size() != 1) {
    throw runtime_error ("Error: Invalid image file header");
  }

  NumberOfLabels  = LabelsFileHeader[0];

  if (NumberOfImages != NumberOfLabels) {
    throw runtime_error ("Error: The number of images does not match the number of labels");
  }

  cout << "Number of images: " << NumberOfImages << endl;

  //
  // Read all images and labels, but only keep those with labels in LabelsToRead.
  //
  for (int Index = 0; Index < (int)NumberOfImages; Index++) {
    vector<double> ImageVector;
    unsigned char  Pixel = 0;
    unsigned char  LabelValue = 0;

    LabelsFile.read ((char*)&LabelValue, sizeof(LabelValue));

    if (!ValueInVector (LabelsToRead, (int)LabelValue)) {
      //
      // This is not the label we want, skip this image.
      //
      ImagesFile.seekg (NumberOfRows * NumberOfColumns * sizeof(Pixel), ios::cur);
      continue;
    }
  
    ImageVector = ReadImageFromIdxToVector (ImagesFile, NumberOfRows, NumberOfColumns);

    matrix  Image (NumberOfRows, NumberOfColumns, ImageVector);
  
    DataSet.push_back (Image);
    LabelSet.push_back ((int)LabelValue);
  }

  cout << "Number of images read: " << DataSet.size() << endl;
  ImagesFile.close ();
  LabelsFile.close ();
}

/**
  Dump an MNIST image to the standard output.

  This function prints the pixel values of an MNIST image to the standard output.
  The image is represented as a matrix with a single column.
  The image is assumed to be 28x28 pixels, and pixels are stored in row-major order.

  @param[in]  MnistImage  The MNIST image to be dumped.

**/
void
DumpMNISTImage (
  matrix  &MnistImage
  )
{
  int Index;

  for(int Row = 0; Row < 28; Row++) {
    for(int Column = 0; Column < 28; Column++) {
      Index = Row * 28 + Column;

      cout<<MnistImage.GetValue(Index, 0)<<' ';
    }
  }

  cout<<endl;
}