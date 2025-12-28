/**
  Matrix calculation functions implementation.

  Copyright (c) 2025, visionaryr
  Licensed under the MIT License. See the accompanying 'LICENSE' file for details.
**/

#include "matrix.h"
#include "DebugLib.h"

#include <iostream>
#include <cstdlib>
using namespace std;

/**
  Multiply a row vector by a column vector.

  @param  Row     A row vector of size n.
  @param  Column  A column vector of size n.

  @return  The result of the multiplication, which is a scalar.

  @throw  std::cout  Size of Row and Column do not match.

**/
static
double RowMultiplyColumn (const vector<double> &Row, const vector<double> &Column)
{
  if (Row.size() != Column.size()) {
    cout << Row.size() << ", " << Column.size() << endl;
    cout<<"RowMultiplyColumn: size of row and column do not match!"<<endl;
    exit(1);
  }

  double Sum = 0;

  for (unsigned int i = 0; i < Row.size(); i++) {
    Sum += Row[i] * Column[i];
  }

  return Sum;
}

/**
  Multiply 2 matrices by A * B. A's column number should be the same as B's row number.
  Return the result matrix.

  @param  A  The first matrix, which should be m * n.
  @param  B  The second matrix, which should be n * p.

  @return  The result matrix, which is m * p.

**/
matrix multiply(const matrix &A, const matrix &B)
{
  int ARows = A.getrow();
  int BRows = B.getrow();
  int AColumns = A.getcolumn();
  int BColumns = B.getcolumn();

  if(AColumns != BRows) {
    DEBUG_LOG ("Columns of A matrix = " << AColumns << ", Rows of B matrix = " << BRows);
    throw runtime_error ("Number of columns in the first matrix should be the same as the number of rows in the second matrix!");
  }

  matrix C (ARows, BColumns);

  //
  // Standard matrix multiplication algorithm
  // C(i, j) = sum (A (i, k) * B (k, j)) for k = 0 to n-1
  //
  double CSum = 0;
  vector<double> ARow;
  vector<double> BColumn;

  for(int ARowIdx = 0; ARowIdx < ARows; ARowIdx++) {
    for(int BColumnIdx = 0; BColumnIdx < BColumns; BColumnIdx++) {
      ARow    = A.ConvertRowToVector (ARowIdx);
      BColumn = B.ConvertColumnToVector (BColumnIdx);
      
      CSum = RowMultiplyColumn (ARow, BColumn);
      C.SetValue(ARowIdx, BColumnIdx, CSum);

      CSum = 0;
    }
  }

  return C;
}

/**
  Transpose a matrix.

  @param  A  The matrix to be transpose, which is m * n.

  @return  The result matrix, which is n * m.

**/
matrix transpose(const matrix &A)
{
  int ARows = A.getrow();
  int AColumns = A.getcolumn();

  matrix C(AColumns, ARows);

  //
  // Standard matrix transpose algorithm
  // C (j, i) = A (i, j)
  //
  for(int RowIdx = 0; RowIdx < ARows; RowIdx++) {
    for(int ColumnIdx = 0; ColumnIdx < AColumns; ColumnIdx++) {
      C.SetValue(ColumnIdx, RowIdx, A.GetValue(RowIdx, ColumnIdx));
    }
  }

  return C;
}

/**
  Multiply a matrix by a scalar(constant). C = A * M(constant).

  @param  A  The matrix to be multiplied by, which is m * n.

  @return  The result matrix, which is m * n.

**/
matrix multiplyBy(const matrix &A, double M)
{
  int ARows = A.getrow();
  int AColumns = A.getcolumn();

  matrix C(ARows, AColumns);

  for(int RowIdx = 0; RowIdx < ARows; RowIdx++) {
    for(int ColumnIdx = 0; ColumnIdx < AColumns; ColumnIdx++) {
      C.SetValue (
          RowIdx,
          ColumnIdx,
          A.GetValue(RowIdx, ColumnIdx) * M
          );
    }
  }

  return C;
}

/**
  Add 2 matrices by C = A + B.

  @param  A  The matrix to be add, which is m * n.
  @param  B  The matrix to be add, which is m * n.

  @return  The result matrix, which is m * n.

**/
matrix add(const matrix &A, const matrix &B)
{
  if ((A.getrow() != B.getrow()) ||
      (A.getcolumn() != B.getcolumn())) {
    DEBUG_LOG ("A size: " << A.getrow() << " * " << A.getcolumn()
               << ", B size: " << B.getrow() << " * " << B.getcolumn());
    throw invalid_argument ("add(): The size of the two matrices should be the same!");
  }

  int Row    = A.getrow();
  int Column = A.getcolumn();

  matrix C(Row, Column);

  for(int RowIdx = 0; RowIdx < Row; RowIdx++) {
    for(int ColumnIdx = 0; ColumnIdx < Column; ColumnIdx++) {
      C.SetValue(
          RowIdx,
          ColumnIdx,
          A.GetValue(RowIdx, ColumnIdx) + B.GetValue(RowIdx, ColumnIdx)
          );
    }
  }

  return C;
}

/**
  Substract 2 matrices by C = A - B.

  @param  A  The matrix to be substract, which is m * n.
  @param  B  The matrix to substract, which is m * n.

  @return  The result matrix, which is m * n.

**/
matrix Substract(const matrix &A, const matrix &B)
{
  if ((A.getrow() != B.getrow()) ||
      (A.getcolumn() != B.getcolumn())) {
    DEBUG_LOG ("A size: " << A.getrow() << " * " << A.getcolumn()
               << ", B size: " << B.getrow() << " * " << B.getcolumn());
    throw invalid_argument ("Substract(): The size of the two matrices should be the same!");
  }

  int Row    = A.getrow();
  int Column = A.getcolumn();

  matrix C(Row, Column);

  for(int RowIdx = 0; RowIdx < Row; RowIdx++) {
    for(int ColumnIdx = 0; ColumnIdx < Column; ColumnIdx++) {
      C.SetValue(
          RowIdx,
          ColumnIdx,
          A.GetValue(RowIdx, ColumnIdx) - B.GetValue(RowIdx, ColumnIdx)
          );
    }
  }

  return C;
}

matrix  HadamardProduct (
  const matrix  &A,
  const matrix  &B
  )
{
  if ((A.getrow() != B.getrow()) ||
      (A.getcolumn() != B.getcolumn())) {
    DEBUG_LOG ("A size: " << A.getrow() << " * " << A.getcolumn()
               << ", B size: " << B.getrow() << " * " << B.getcolumn());
    throw invalid_argument ("HadamardProduct(): The size of the two matrices should be the same!");
  }

  int Row    = A.getrow();
  int Column = A.getcolumn();

  matrix C(Row, Column);

  for(int RowIdx = 0; RowIdx < Row; RowIdx++) {
    for(int ColumnIdx = 0; ColumnIdx < Column; ColumnIdx++) {
      C.SetValue(
          RowIdx,
          ColumnIdx,
          A.GetValue(RowIdx, ColumnIdx) * B.GetValue(RowIdx, ColumnIdx)
          );
    }
  }

  return C;
}