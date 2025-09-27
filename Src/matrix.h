//matrix.h
#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

class matrix
{
	public:
		matrix();		
		matrix(int, int);
		matrix(int, int, std::vector<double>);
		void show() const;
		void test_show() const;
		void setmatrix(int, int, std::vector<double>);
		int getrow() const;
		int getcolumn() const;
		double getMatrix(int, int) const;
		void SetValue(int, int, double);
		std::vector<double> convert_to_vector();

	private:
		int row,column;
		std::vector< std::vector<double> > A;
		void Init();

};

//matrix calculating functions(in matrix_calculate.cpp)
matrix multiply(matrix &A, matrix &B);
matrix transpose(const matrix &);
matrix multiplyBy(const matrix &, double);
matrix add(const matrix &, const matrix &);



#endif /* MATRIX_H */
