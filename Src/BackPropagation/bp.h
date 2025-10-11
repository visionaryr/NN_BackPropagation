#ifndef _BP_H_
#define _BP_H_

#include "matrix.h"
#include "network.h"
#include <vector>
#include <png.h>

typedef std::vector< double >  IMAGE;
typedef std::vector< IMAGE >   DATA_SET;
typedef std::vector< int >     LABELS;

//learning and weight upgrading functions
matrix Activation (matrix);
void Learning_FP(network &, matrix);
void delta_calc(network &, matrix desired_output);
std::vector<matrix> delta_w_calc(network &, double);
void upgrade_weight(network &, std::vector<matrix> &);
double loss_func(network &, matrix &);

//read png file
bool read_png_file(char *filename, std::vector<double> &);
void get_png_file(std::vector<double> &, int, int, png_bytep *);
void get_test_images(std::vector< std::vector<double> > &);
void get_train_images(std::vector< std::vector<double> > &, std::vector< std::vector<double> > &, std::vector<int> &);

//dataset I/O functions
int ReverseInt(int);
void ReadMNIST_and_label(int, int, std::vector< std::vector<double> > &, std::vector< std::vector<double> > &, std::vector<int> &);
//void read_Mnist_Label(std::vector< std::vector<double> > &, std::vector<int> &);

//other global functions
double predict(std::vector< std::vector<double> > &, std::vector< std::vector<double> > &, int, int, network &, std::vector<int> &);

#endif
