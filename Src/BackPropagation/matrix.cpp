#include "matrix.h"
#include <iostream>
#include <iomanip>

using namespace std;


matrix::matrix()
{

}

matrix::matrix(int r, int c)
{
    row=r;
    column=c;
	Init();
}//end of matrix constructor(all zero)

matrix::matrix(int r, int c, vector<double> put)
{
	row=r;
	column=c;
	Init();
	int counter=0;
    for(int i=0;i<row;i++)
    {
        for(int j=0;j<column;j++)
        {
            A[i][j]=put[counter];
            counter++;
        }
    }
}//end of matrix constructor(with set matrix)

int matrix::getrow() const
{
    return row;
}//end of matrix getrow

int matrix::getcolumn() const
{
    return column;
}//end of matrix getcolumn

void matrix::show() const
{
    for(int i=0;i<row;i++)
    {
        for(int j=0;j<column;j++)
            cout<<setw(5)<<A[i][j]<<' ';
        cout<<endl;
    }
}//end of matrix show

void matrix::test_show() const
{
	for(int i=0;i<row;i++)
    {
        for(int j=0;j<column;j++)
            cout<<setw(2)<<(A[i][j]>0.5?1:0);
        cout<<endl;
    }
}

void matrix::setmatrix(int r, int c, vector<double> put)
{
    row=r;
    column=c;
    int counter=0;
    for(int i=0;i<row;i++)
    {
        for(int j=0;j<column;j++)
        {
            A[i][j]=put[counter];
            counter++;
        }

    }
}//end of matrix setmatrix

double matrix::getMatrix(int r, int c) const
{
    return A[r][c];
}

void matrix::SetValue(int row, int column, double value)
{
	A[row][column]=value;
	return;
}

void matrix::Init()
{
	vector<double> A1;
	for(int i=0;i<row;i++)
	{
		for(int j=0;j<column;j++)
		{
			A1.push_back(0);
		}
		A.push_back(A1);
	}
}

vector<double> matrix::convert_to_vector()
{
	vector<double> re;
	for(int i=0;i<row;i++)
	{
		for(int j=0;j<column;j++)
		{
			re.push_back(A[i][j]);
		}
	}
	return re;
}
