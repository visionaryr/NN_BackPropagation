/**
  ComputationContext class definition.

  Copyright (c) 2025, visionaryr
  Licensed under the MIT License. See the accompanying 'LICENSE' file for details.
**/

#ifndef _COMPUTATION_CONTEXT_H_
#define _COMPUTATION_CONTEXT_H_

#include <vector>
#include "matrix.h"

class ComputationContext {

public:
  ComputationContext (std::vector<unsigned int>  Layout);
  ~ComputationContext ();

  double GetCurrentLoss () const;

  void
  SetActivationByLayer (
    unsigned int   Layer,
    const  matrix  ActivationOfLayer
  );

  matrix
  GetActivationByLayer (
    unsigned int Layer
    ) const;

  void
  PrintActivationInLayer (
    unsigned int  Layer
    );

  matrix
  GetNodeDeltaByLayer (
    unsigned int Layer
    ) const;

  void
  SetNodeDeltaByLayer (
    unsigned int Layer,
    matrix       NodeDeltaOfLayer
    );

  void
  PrintNodeDelta(
    void
    );

private:
  void
  InitActivation (
    const std::vector<unsigned int>  &Layout
    );

  void
  InitNodeDelta (
    std::vector<unsigned int>  &Layout
    );

  std::vector<matrix>    Activation;
  std::vector<matrix>    NodeDelta;
  double                 Loss;
};

#endif