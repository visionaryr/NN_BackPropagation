#ifndef NETWORK_H
#define NETWORK_H

#include "matrix.h"
#include "bp.h"

#include <vector>
#include <fstream>

class FullyConnectedNetwork
{
  public:
    FullyConnectedNetwork(std::vector<int> &);
    FullyConnectedNetwork(char*);
    void ShowInfo(bool);
    //void Learning_FP(matrix);
    void set_a(int, int, double);
    void set_delta(int, int, double);
    void shake();
    matrix test(matrix &);
    void save_network();
    void print_a();
    void print_delta();
    friend void Learning_FP(FullyConnectedNetwork &, matrix);
    friend void delta_calc(FullyConnectedNetwork &, matrix);
    friend std::vector<matrix> delta_w_calc(FullyConnectedNetwork &, double);
    friend void upgrade_weight(FullyConnectedNetwork &, std::vector<matrix> &);
    friend double loss_func(FullyConnectedNetwork &, matrix &);

  private:
    void Init_para();
    void WeightsRandomize();
    void test_Init();//Initialize to testing mode
    double RandValue(); // Generate a random double value between -1.0 and 1.0.
    void WeightsMatrixInit(bool);// Initialize weight matrix between each layers based on Layout
    void ImportNetwork (char *); // Import a network from a file.

    std::vector< std::vector<double> > a;
    std::vector< std::vector<double> > delta;
    std::vector<matrix *> Weights;
    std::vector<int> Layout;
};

typedef struct {
  unsigned int Signature;
  unsigned int HdrSize;     // Size of the file header in bytes (From Signature to the end of Layout)
  unsigned int NumOfLayers;
  // unsigned int Layout[NumOfLayers];
  // double Weights[Layout[0] * Layout[1]];
  // ...
  // double Weights[Layout[NumOfLayers-2] * Layout[NumOfLayers-1]];
} NetworkFile;

//
// Internal helper functions
//
void save_weight_to_file(std::fstream &, matrix &);

//Batch Mode
std::vector<matrix> BatchMode_sum(std::vector<matrix> &, std::vector<matrix> &);
std::vector<matrix> BatchMode_Init(std::vector<int> &);
std::vector<matrix> BatchMode_average(std::vector<matrix> &, double);

#endif
