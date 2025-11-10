#ifndef _PNG_IO_H_
#define _PNG_IO_H_

#include <vector>

#include <png.h>


bool read_png_file(char *filename, std::vector<double> &);
void get_png_file(std::vector<double> &, int, int, png_bytep *);
void get_test_images(std::vector< std::vector<double> > &);
void get_train_images(std::vector< std::vector<double> > &, std::vector< std::vector<double> > &, std::vector<int> &);

#endif