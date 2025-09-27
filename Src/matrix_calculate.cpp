#include "matrix.h"
#include <iostream>
#include <cstdlib>
using namespace std;

matrix multiply(matrix &A, matrix &B)
{
	//cout<<"Before multiply 1:"<<endl;
    //A.show();
    //cout<<"Before multiply 2:"<<endl;
    //B.show();
    int ra=A.getrow();
    int rb=B.getrow();
    int ca=A.getcolumn();
    int cb=B.getcolumn();
    if(ca!=rb)
    {
        cout<<"The number of columns in the first matrix should be the same as the number of rows in the second matrix!"<<endl;
        exit(1);
    }
    matrix Ans(ra,cb);
    double sum=0;
    for(int i=0;i<ra;i++)
    {
        for(int j=0;j<cb;j++)
        {
            for(int k=0;k<rb;k++)
            {
                sum+=A.getMatrix(i,k)*B.getMatrix(k,j);
            }
            Ans.SetValue(i, j, sum);
            sum=0;
        }
    }
	//cout<<"multiply finished:"<<endl;
	//Ans.show();
    return Ans;
}

matrix transpose(const matrix &A)
{
    //cout<<"Before transpose:"<<endl;
    //A.show();
    int ra=A.getrow();
    int ca=A.getcolumn();
    matrix Ans(ca,ra);
    for(int i=0;i<ra;i++)
    {
        for(int j=0;j<ca;j++)
            Ans.SetValue(j, i, A.getMatrix(i,j));
    }
    return Ans;
}//end of calculator transpose

matrix multiplyBy(const matrix &A, double M)
{
    //cout<<"Matrix 1:"<<endl;
    //A.show();
    int ra=A.getrow();
    int ca=A.getcolumn();
    matrix Ans(ra,ca);
    for(int i=0;i<ra;i++)
    {
        for(int j=0;j<ca;j++)
            Ans.SetValue(i, j, A.getMatrix(i,j)*M);
    }
    return Ans;
}//end of calculator multiply By


matrix add(const matrix &A, const matrix &B)
{
    //cout<<"Matrix 1:"<<endl;
    //A.show();
    //cout<<"Matrix 2:"<<endl;
    //B.show();
    int row, column;
    row=A.getrow();
    column=B.getcolumn();
    matrix Ans(row,column);
    double a, b;
    for(int k=0;k<row;k++)
    {
        for(int l=0;l<column;l++)
        {
            a=A.getMatrix(k,l);
            b=B.getMatrix(k,l);
            Ans.SetValue(k,l,a+b);
        }
    }
    //cout<<"ANS:"<<endl;
	//Ans.show();
    return Ans;
}//end of calculator add


