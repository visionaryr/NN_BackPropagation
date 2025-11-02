#include "FullyConnectedNetwork.h"
#include "DebugLib.h"

#include <iostream>

using namespace std;

/**
  Write the weight matrix to the given file stream.

  @param  fs       The file stream to write the weight matrix to.
  @param  Weight   The weight matrix to be written.
**/
void
WriteWeightMatrixToFile (
  fstream  &fs,
  matrix   &Weight
  )
{
  double  Value;

  int Row = Weight.getrow();
  int Column = Weight.getcolumn();
  for(int RowIdx = 0; RowIdx < Row; RowIdx++) {
    for(int ColumnIdx = 0; ColumnIdx < Column; ColumnIdx++) {
      Value = Weight.GetValue(RowIdx, ColumnIdx);
      fs.write (reinterpret_cast<const char *>(&Value), sizeof(double));
    }
  }
}

/**
  Export the fully connected network to a file.

  @param  Filename  The name of the file to export the network to.

**/
void
FullyConnectedNetwork::ExportToFile (
  string  Filename
  )
{
  fstream       fs;
  NETWORK_FILE  *FileHeader;
  unsigned int  HdrSize;

  fs.open (Filename, ios::out | ios::binary);
  if (!fs) {
    DEBUG_LOG ("Failed to open file: " << Filename << " in binary write mode");
    throw std::runtime_error("ExportToFile: File opening error");
  }

  //
  // Prepare file header
  //
  HdrSize = sizeof(NETWORK_FILE) + (Layout.size() - 1) * sizeof(u_int32_t); // -1 since Layout[1] already includes one unsigned int.
  FileHeader = (NETWORK_FILE *) new char[HdrSize];

  FileHeader->Signature   = NETWORK_FILE_SIGNATURE;
  FileHeader->NumOfLayers = (unsigned int)Layout.size();
  FileHeader->HdrSize     = HdrSize;
  for (int Index = 0; Index < (int)Layout.size(); Index++) {
    FileHeader->Layout[Index] = (unsigned int)Layout[Index];
  }

  //
  // Write file header
  //
  fs.write (reinterpret_cast<const char *>(&FileHeader), HdrSize);
  delete [] (char *)FileHeader;

  //
  // Write weights of each layers.
  //
  for (int Index = 0; Index < (int)Weights.size(); Index++) {
    WriteWeightMatrixToFile (fs, *Weights[Index]);
  }

  fs.close ();
}

/**
  Validate the network file header.

  @param  FileHeader  Pointer to the network file header to be validated.

  @retval true   The file header is valid
  @retval false  File header is not valid.

**/
bool
IsValidFileHeader (
  NETWORK_FILE  *FileHeader
  )
{
  unsigned int  ExpectedHdrSize;

  if (FileHeader == NULL) {
    return false;
  }

  if (FileHeader->Signature != NETWORK_FILE_SIGNATURE) {
    return false;
  }

  ExpectedHdrSize = sizeof(NETWORK_FILE) + (FileHeader->NumOfLayers - 1) * sizeof(unsigned int);
  if (FileHeader->HdrSize != ExpectedHdrSize) {
    return false;
  }

  return true;
}

/**
  Import the fully connected network from a file.

  @param  Filename  The name of the file to import the network from.

**/
void
FullyConnectedNetwork::ImportFromFile (
  string  Filename
  )
{
  fstream       fs;
  NETWORK_FILE  *FileHeader;
  unsigned int  HdrSize;
  unsigned int  FileSignature;

  fs.open (Filename, ios::in | ios::binary);
  if (!fs) {
    DEBUG_LOG ("Failed to open file: " << Filename << " in binary read mode");
    throw std::runtime_error("ImportFromFile: File opening error");
  }

  //
  // Read file header
  //
  fs.read (reinterpret_cast<char *>(&FileSignature), sizeof(u_int32_t));
  if (FileSignature != NETWORK_FILE_SIGNATURE) {
    DEBUG_LOG ("Invalid file signature: " << std::hex << FileSignature);
    throw std::runtime_error("ImportFromFile: Invalid file signature");
  }

  fs.read (reinterpret_cast<char *>(&HdrSize), sizeof(u_int32_t));
  FileHeader = (NETWORK_FILE *) new char[HdrSize];

  fs.seekg (0, ios::beg);
  fs.read (reinterpret_cast<char *>(&FileHeader), HdrSize);

  if (!IsValidFileHeader (FileHeader)) {
    DEBUG_LOG ("Invalid network file header");
    delete [] (char *)FileHeader;
    throw std::runtime_error("ImportFromFile: Invalid network file header");
  }

  //
  // Initialize network layout
  //
  Layout.clear();
  for (int Index = 0; Index < (int)FileHeader->NumOfLayers; Index++) {
    Layout.push_back((int)FileHeader->Layout[Index]);
  }

  delete [] (char *)FileHeader;

  //
  // Initialize weight matrices
  //
  Weights.clear();
  WeightsMatrixInit(false);

  //
  // Read weights of each layers.
  //
  for (int Index = 0; Index < (int)Weights.size(); Index++) {
    matrix  *Weight = Weights[Index];
    double   Value;
    int      Row    = Weight->getrow();
    int      Column = Weight->getcolumn();

    for (int RowIdx = 0; RowIdx < Row; RowIdx++) {
      for (int ColumnIdx = 0; ColumnIdx < Column; ColumnIdx++) {
        fs.read (reinterpret_cast<char *>(&Value), sizeof(double));
        Weight->SetValue(RowIdx, ColumnIdx, Value);
      }
    }
  }

  fs.close ();
}
