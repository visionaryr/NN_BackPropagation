#include <iostream>
#include <vector>
#include <fstream>
#include <stdio.h>

#include "dataset.h"
#include "bp.h"
 
#define PAUSE printf("Press Enter key to continue..."); fgetc(stdin); 
 
using namespace std;

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
  Check if a value is present in a vector.

  @param[in]  VectorToSearch  The vector to be searched.
  @param[in]  Value           The value to search for.

  @retval  true   The value is found in VectorToSearch.
  @retval  false  The value is not found in VectorToSearch.

**/
static
bool
ValueInVector (
  vector<int> &VectorToSearch,
  int         Value
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
  DATA_SET    &DataSet,
  LABEL_SET   &LabelSet,
  vector<int> &LabelsToRead
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
  DATA_SET      Images;
  LABEL_SET     Labels;

  if (LabelsToRead.size() == 0) {
    throw runtime_error ("Error: No labels to read");
  }
  if (LabelsToRead.size() > 10) {
    throw runtime_error ("Error: Too many labels to read");
  }

  //
  // Open the image file.
  //
  ImagesFile = OpenIdxFile ("train-images.idx3-ubyte", 0x00000803, ImagesFileHeader);
  if (ImagesFileHeader.size() != 3) {
    throw runtime_error ("Error: Invalid image file header");
  }

  NumberOfImages  = ImagesFileHeader[0];
  NumberOfRows    = ImagesFileHeader[1];
  NumberOfColumns = ImagesFileHeader[2];

  //
  // Open the label file.
  //
  LabelsFile = OpenIdxFile ("train-labels.idx1-ubyte", 0x00000801, LabelsFileHeader);
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
    vector<double>  Image;
    unsigned char   Pixel = 0;
    unsigned char   LabelValue = 0;

    for (int Row = 0; Row < (int)NumberOfRows; Row++) {
      for (int Column = 0; Column < (int)NumberOfColumns; Column++) {
        ImagesFile.read ((char*)&Pixel, sizeof(Pixel));
        Image.push_back ((double)Pixel);
      }
    }

    LabelsFile.read ((char*)&LabelValue, sizeof(LabelValue));

    if (ValueInVector (LabelsToRead, (int)LabelValue)) {
      //
      // This is the label we want. Add image and label to DataSet and LabelSet respectively.
      //
      DataSet.push_back (Image);
      LabelSet.push_back ((int)LabelValue);
    }
  }

  cout << "Number of images read: " << DataSet.size() << endl;
  ImagesFile.close ();
  LabelsFile.close ();
}


vector<double> output_convert(int O_dataset, vector<int> &train_cat)
{
  vector<double> d_output;
  for(int i=0;i<(int)train_cat.size();i++)
  {
    if(train_cat[i]==O_dataset)
    {
      d_output.push_back(1.0);
    }
    else
      d_output.push_back(0.0);
  }
  return d_output;
}

void show_as_image(matrix &A)
{
  for(int i=0;i<28;i++)
  {
    for(int j=0;j<28;j++)
    {
      cout<<A.GetValue(28*i+j,0)<<' ';
    }
  }
}

int to_number(matrix &A, vector<int> &train_cat)
{
  int ans=0;
  for(int i=0;i<(int)train_cat.size();i++)
  {
    if(A.GetValue(i,0)>.7) ans+=train_cat[i];
  }
  if(ans>=10) return ans*(-1);
  else return ans;
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
