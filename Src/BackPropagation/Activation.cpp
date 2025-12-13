/**
  Activation and derivative activation functions implementation.

  Copyright (c) 2025, visionaryr
  Licensed under the MIT License. See the accompanying 'LICENSE' file for details.
**/

#include "Activation.h"
#include "DebugLib.h"

#include <cmath>
#include <iostream>

using namespace std;

/**
  The activation function f(x) = 1 / (1 + e^(-x)).

  @param  x   input value.

  @return  The output value after applying the activation function.

**/
double 
Sigmold (
  double x
  )
{
  return 1 / (1 + exp((-1) * x));
}

/**
  The derivative function f'(x) = x * (1 - x) of activation function f(x) = 1 / (1 + e^(-x)).

  @param  x   input value.

  @return  The output value after applying the derivative activation function.

**/
double
SigmoldDerivative (
  double x
  )
{
  return x * (1 - x);
}

/**
  Get activation function of specified activation type.

  @param[in]  Type  An activation type.

  @return  Activation function of Type.

**/
ACTIVATION_FUNC
GetActivationFunction (
  ACTIVATION_TYPE  Type
  )
{
  switch (Type) {
    case SIGMOLD:
      return Sigmold;

    default:
      DEBUG_LOG ("Unsupported activation type = " << Type);
      throw runtime_error ("Unsupported activation type.");
  }
}

/**
  Get deriative activation function of specified activation type.

  @param[in]  Type  An activation type.

  @return  Deriative activation function of Type.

**/
ACTIVATION_FUNC
GetDeriativeActivationFunction (
  ACTIVATION_TYPE  Type
  )
{
  switch (Type) {
    case SIGMOLD:
      return SigmoldDerivative;

    default:
      DEBUG_LOG ("Unsupported activation type = " << Type);
      throw runtime_error ("Unsupported activation type.");
  }
}