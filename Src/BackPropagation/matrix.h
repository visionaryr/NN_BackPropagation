/**
  Matrix class definition.

  Copyright (c) 2025, visionaryr
  Licensed under the MIT License. See the accompanying 'LICENSE' file for details.
**/
#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <functional>

class matrix
{
  public:
    matrix();    
    matrix(unsigned int, unsigned int);
    matrix(unsigned int, unsigned int, double);
    matrix(unsigned int, unsigned int, std::vector<double>);
    void show() const;
    void test_show() const;
    unsigned int getrow() const;
    unsigned int getcolumn() const;
    double GetValue(unsigned int, unsigned int) const;
    void SetValue(unsigned int, unsigned int, double);
    double Sum () const;

    std::vector<double> ConvertToVector();
    std::vector<double> ConvertRowToVector (unsigned int) const;
    std::vector<double> ConvertColumnToVector (unsigned int) const;

    matrix ApplyElementWise (std::function<double(double)> &Func) const;

  private:
    unsigned int row;
    unsigned int column;
    std::vector< std::vector<double> > Matrix;
    void InitMatrixWithValue(unsigned int, unsigned int, double);
    int SetMatrix(unsigned int, unsigned int, std::vector<double>);
};

//
// matrix calculating functions(in matrix_calculate.cpp)
//
matrix multiply(const matrix &A, const matrix &B);
matrix transpose(const matrix &);
matrix multiplyBy(const matrix &, double);
matrix add(const matrix &, const matrix &);
matrix Substract (const matrix &, const matrix &);
matrix HadamardProduct (const matrix &, const matrix &);



#endif /* MATRIX_H */
