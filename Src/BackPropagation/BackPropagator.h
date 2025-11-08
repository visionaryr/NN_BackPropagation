#ifndef _BACK_PROPAGATOR_H_
#define _BACK_PROPAGATOR_H_

#include "matrix.h"
#include "FullyConnectedNetwork.h"
#include <vector>
#include <functional>

class BackPropagator
{
  public:
    BackPropagator (FullyConnectedNetwork &FCN);
  
    //void Train ();

  private:
    void InitNodeDelta ();
    void InitDeltaWeights ();
    void SetNodeDelta (
      unsigned int  Layer,
      unsigned int  Number,
      double        Delta
      );
    void PrintNodeDelta (); // Only internal debug use.

    void NodeDeltaCalculation (
      matrix DesiredOutput
     );
    void CalculateUpdateDeltaWeights
    (
      const std::vector<matrix> &input_activations,
      const matrix &desired_output,
      double learning_rate
    );

    void  CalculateLastLayerDelta (
      matrix  DesiredOutput
      );
    void  CalculateMidLayerDelta (
      unsigned int  Layer
      );
    FullyConnectedNetwork          Network;
    std::vector<matrix>            NodeDelta;
    std::vector<matrix>            DeltaWeights;
    std::function<double(double)>  ActivationDerivative;
};

typedef std::vector< double >  IMAGE;
typedef std::vector< IMAGE >   DATA_SET;
typedef std::vector< int >     LABELS;

//
// Internal learning and weight updating functions
//
void Learning_FP(FullyConnectedNetwork &, matrix);
void delta_calc(FullyConnectedNetwork &, matrix desired_output);
std::vector<matrix> delta_w_calc(FullyConnectedNetwork &, double);
void upgrade_weight(FullyConnectedNetwork &, std::vector<matrix> &);
double loss_func(FullyConnectedNetwork &, matrix &);

//read png file
// bool read_png_file(char *filename, std::vector<double> &);
// void get_png_file(std::vector<double> &, int, int, png_bytep *);
// void get_test_images(std::vector< std::vector<double> > &);
// void get_train_images(std::vector< std::vector<double> > &, std::vector< std::vector<double> > &, std::vector<int> &);

//dataset I/O functions
int ReverseInt(int);
void ReadMNIST_and_label(int, int, std::vector< std::vector<double> > &, std::vector< std::vector<double> > &, std::vector<int> &);
//void read_Mnist_Label(std::vector< std::vector<double> > &, std::vector<int> &);

//other global functions
double predict(std::vector< std::vector<double> > &, std::vector< std::vector<double> > &, int, int, FullyConnectedNetwork &, std::vector<int> &);

#endif
