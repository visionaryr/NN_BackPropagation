/**
  Matrix class implementation.

  Copyright (c) 2025, visionaryr
  Licensed under the MIT License. See the accompanying 'LICENSE' file for details.
**/

#include "matrix.h"
#include "DebugLib.h"

#include <iostream>
#include <iomanip>

using namespace std;

/**
  Default constructor does nothing. Consumer should never use this one.

**/
matrix::matrix()
{

}

/**
  Initialize a Rows * Columns matrix, all elements are set to zero.

  @param  row     number of rows
  @param  column  number of columns

**/
matrix::matrix(unsigned int Rows, unsigned int Columns)
{
  InitMatrixWithValue (Rows, Columns, 0.0);
}

matrix::matrix(unsigned int Rows, unsigned int Columns, double InitValue)
{
  InitMatrixWithValue (Rows, Columns, InitValue);
}

/**
  Initialize a Rows * Columns matrix, all elements are set to the values
  in the vector InitValues.

  @param  Rows        number of rows
  @param  Columns     number of columns
  @param  InitValues  a vector of size Rows*Columns, contains the initial
                      values of the matrix elements.
                      The order of the values are in row-major order.

  @throw  std::invalid_argument  Size of InitValues is not equal to (Rows * Columns).

**/
matrix::matrix(unsigned int Rows, unsigned int Columns, vector<double> InitValues)
{
  int ret = SetMatrix (Rows, Columns, InitValues);

  if (ret != 0) {
    DEBUG_LOG ("Matrix with (Row, Column) = (" << Rows << ", " << Columns << ") doesn't have same size with InitValues = " << InitValues.size());
    throw std::invalid_argument("matrix constructor: wrong size of InitValues");
  }
}

/**
  Get the number of rows of the matrix.

  @return  number of rows.

**/
unsigned int
matrix::getrow(
  void
  ) const
{
  return row;
}

/**
  Get the number of columns of the matrix.

  @return  number of columns.

**/
unsigned int
matrix::getcolumn(
  void
  ) const
{
  return column;
}

/**
  Print out the matrix.

**/
void matrix::show() const
{
  // Print with reasonable precision without changing global cout state.
  std::ios::fmtflags oldFlags = cout.flags();
  std::streamsize oldPrec = cout.precision();

  for(unsigned int RowIdx = 0; RowIdx < row; RowIdx++) {
    for(unsigned int ColumnIdx = 0; ColumnIdx < column; ColumnIdx++) {
      cout << std::setprecision(6) << setw(10) << Matrix[RowIdx][ColumnIdx] << ' ';
    }
    cout << endl;
  }

  // Restore previous formatting flags
  cout.flags(oldFlags);
  cout.precision(oldPrec);
}

/**
  Print out the test matrix.
  *Test usage.

**/
void matrix::test_show() const
{
  for(unsigned int RowIdx = 0; RowIdx < row; RowIdx++)
  {
    for(unsigned int ColumnIdx = 0; ColumnIdx < column; ColumnIdx++)
    {
      cout<<setw(2)<<((Matrix[RowIdx][ColumnIdx] > 0.5) ? 1 : 0);
    }
    cout<<endl;
  }
}

/**
  Set the matrix to be a Rows * Columns matrix, all elements are set to the values
  in the vector SetValues.
  The previous values in the matrix will be cleared.

  @param  Rows        number of rows
  @param  Columns     number of columns
  @param  SetValues   a vector of size Rows*Columns, contains the values
                      of the matrix elements.
                      The order of the values are in row-major order.

  @return   0  Matrix is set successfully.
  @return  -1  Size of SetValues vector is not equal to (Rows * Columns).

**/
int
matrix::SetMatrix (
  unsigned int   Rows,
  unsigned int   Columns,
  vector<double> SetValues
  )
{
  if ((unsigned int)SetValues.size() != (Rows * Columns)) {
    return -1;
  }

  InitMatrixWithValue (Rows, Columns, 0.0);

  int counter = 0;
  for(unsigned int RowIdx = 0; RowIdx < row; RowIdx++)
  {
    for(unsigned int ColumnIdx = 0; ColumnIdx < column; ColumnIdx++)
    {
      Matrix[RowIdx][ColumnIdx] = SetValues[counter];
      counter++;
    }
  }

  return 0;
}

/**
  Get the value of the element at (Row, Column).

  @param  Row     Row index of the element.
  @param  Column  Column index of the element.

  @return  Value of the element at (Row, Column)

**/
double
matrix::GetValue (
  unsigned int Row,
  unsigned int Column
  ) const
{
  if (Row >= row || Column >= column) {
    throw std::out_of_range("matrix::GetValue: index out of range");
  }
  if (Matrix.size() == 0) {
    throw std::logic_error("matrix::GetValue: matrix is empty");
  }

  return Matrix[Row][Column];
}

/**
  Set the value to the element at (Row, Column).

  @param  Row     Row index of the element.
  @param  Column  Column index of the element.

**/
void
matrix::SetValue (
  unsigned int Row,
  unsigned int Column,
  double Value
  )
{
  if (Row >= row || Column >= column) {
    throw std::out_of_range("matrix::SetValue: index out of range");
  }
  if (Matrix.size() == 0) {
    throw std::logic_error("matrix::SetValue: matrix is empty");
  }

  Matrix[Row][Column] = Value;
}

/**
  Initialize a Rows * Columns matrix with specified value.
  Previous values in the matrix will be cleared.

  @param  Rows     number of rows
  @param  Columns  number of columns

**/
void
matrix::InitMatrixWithValue (
  unsigned int Rows,
  unsigned int Columns,
  double InitValue
  )
{
  if (Matrix.size () != 0) {
    Matrix.clear ();
  }

  row    = Rows;
  column = Columns;

  for(unsigned int RowIdx = 0; RowIdx < row; RowIdx++)
  {
    vector<double> ARow(column, InitValue);
    Matrix.push_back(ARow);
  }
}

/**
  Calculate the sum of all elements in the matrix.

  @return  The sum of all elements.

**/
double
matrix::Sum (
  void
  ) const
{
  double  Sum = 0.0;

  for(unsigned int RowIdx = 0; RowIdx < row; RowIdx++) {
    for (unsigned int ColumnIdx = 0; ColumnIdx < column; ColumnIdx++) {
      Sum += Matrix[RowIdx][ColumnIdx];
    }
  }

  return Sum;
}

/**
  Convert the matrix to a 1-D vector in row-major order.

  @param  Rows     number of rows
  @param  Columns  number of columns

  @return  A vector of size (Rows * Columns), contains the values
           of the matrix elements in row-major order.

**/
vector<double> matrix::ConvertToVector()
{
  vector<double> RetVector;
  for(unsigned int RowIdx = 0; RowIdx < row; RowIdx++)
  {
    for(unsigned int ColumnIdx = 0; ColumnIdx < column; ColumnIdx++)
    {
      RetVector.push_back(Matrix[RowIdx][ColumnIdx]);
    }
  }
  return RetVector;
}

/**
  Convert a specific row of the matrix to a 1-D vector.

  @param  Row   Row index of the row to be converted.

  @return  A vector of size (Columns), contains the values
           of the elements in the specified row.

  @throw  std::out_of_range  Row index is out of range.

**/
vector<double> matrix::ConvertRowToVector (unsigned int Row) const
{
  if (Row >= row)
  {
    throw std::out_of_range("matrix::ConvertRowToVector: index out of range");
  }

  return Matrix[Row];
}

/**
  Convert a specific column of the matrix to a 1-D vector.

  @param  Column   Column index of the column to be converted.

  @return  A vector of size (Rows), contains the values
           of the elements in the specified column.

  @throw  std::out_of_range  Column index is out of range.

**/
vector<double> matrix::ConvertColumnToVector (unsigned int Column) const
{
  if (Column >= column) {
    throw std::out_of_range("matrix::ConvertColumnToVector: index out of range");
  }

  vector<double> ColumnVector;

  for (unsigned int RowIdx = 0; RowIdx < row; RowIdx++) {
    ColumnVector.push_back (Matrix[RowIdx][Column]);
  }

  return ColumnVector;
}

/**
  Apply a mathematical function to each element of this matrix.
  C = Func(Matrix).

  @param  Func  The function to be applied to each element of this matrix.

  @return  The result matrix.

**/
matrix
matrix::ApplyElementWise (
  std::function<double(double)> &Func
  ) const
{
  matrix C(row, column);

  for(unsigned int RowIdx = 0; RowIdx < row; RowIdx++) {
    for(unsigned int ColumnIdx = 0; ColumnIdx < column; ColumnIdx++) {
      C.SetValue (
          RowIdx,
          ColumnIdx,
          Func(Matrix[RowIdx][ColumnIdx])
          );
    }
  }

  return C;
}