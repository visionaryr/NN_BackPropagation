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
    void show_info();
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
    void network_rand_Init();
    void test_Init();//Initialize to testing mode
    double rand_value();
    void weight_Init();//Initialize weight with network frame
    std::vector< std::vector<double> > a;
    std::vector< std::vector<double> > delta;
    std::vector<matrix *> weight;
    std::vector<int> Layout;
};

void save_weight_to_file(std::fstream &, matrix &);
void import_network(char*);

//Batch Mode
std::vector<matrix> BatchMode_sum(std::vector<matrix> &, std::vector<matrix> &);
std::vector<matrix> BatchMode_Init(std::vector<int> &);
std::vector<matrix> BatchMode_average(std::vector<matrix> &, double);

#endif
