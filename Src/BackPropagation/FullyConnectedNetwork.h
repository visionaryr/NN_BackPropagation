#ifndef _FULLY_CONNECTED_NETWORK_H
#define _FULLY_CONNECTED_NETWORK_H

#include "matrix.h"
// #include "bp.h"

#include <string>
#include <vector>
#include <fstream>
#include <cstdint>

class FullyConnectedNetwork
{
  public:
    FullyConnectedNetwork(std::vector<int> &);
    FullyConnectedNetwork(std::string);
    void ExportToFile(std::string, std::string); // Export the network to a file.
    void ImportFromFile(std::string); // Import a network from a file.
    void ShowInfo(bool);
    void SetNodeActivation (unsigned int, unsigned int, double);
    matrix GetActivationByLayer (unsigned int) const;
    void PrintActivationInLayer (unsigned int);
    matrix GetWeightByLayer (unsigned int) const;
    void UpdateWeightByLayer (unsigned int, const matrix &);
    void PerturbWeight ();
    void SetNodeDelta(unsigned int, unsigned int, double);
    void print_delta();

    friend void Learning_FP(FullyConnectedNetwork &, matrix);
    friend void delta_calc(FullyConnectedNetwork &, matrix);
    friend std::vector<matrix> delta_w_calc(FullyConnectedNetwork &, double);
    friend void upgrade_weight(FullyConnectedNetwork &, std::vector<matrix> &);
    friend double loss_func(FullyConnectedNetwork &, matrix &);

  private:
    void InitNodeActivation ();
    void InitNodeDelta ();
    void WeightsRandomize();
    double RandValue(); // Generate a random double value between -1.0 and 1.0.
    void WeightsMatrixInit(bool);// Initialize weight matrix between each layers based on Layout
    
    //
    // Data Members
    //
    std::vector<matrix> NodeActivation;
    std::vector<matrix> NodeDelta;
    std::vector<matrix> Weights;
    std::vector<int> Layout;
};

typedef struct {
  u_int32_t Signature;
  u_int32_t HdrSize;     // Size of the file header in bytes (From Signature to the end of Layout)
  u_int32_t NumOfLayers;
  u_int32_t Layout[1];  // variable length(NumOfLayers) array
  // double Weights[Layout[0] * Layout[1]];
  // ...
  // double Weights[Layout[NumOfLayers-2] * Layout[NumOfLayers-1]];
} NETWORK_FILE;

#define NETWORK_FILE_SIGNATURE 0x46434E57  // "FCNW" in ASCII

//
// Internal helper functions
//
void WriteWeightMatrixToFile (std::fstream &, matrix &);

//Batch Mode
std::vector<matrix> BatchMode_sum(std::vector<matrix> &, std::vector<matrix> &);
std::vector<matrix> BatchMode_Init(std::vector<int> &);
std::vector<matrix> BatchMode_average(std::vector<matrix> &, double);

#endif
