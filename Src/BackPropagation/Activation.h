/**
  Activation and derivative activation functions implementation.

  Copyright (c) 2025, visionaryr
  Licensed under the MIT License. See the accompanying 'LICENSE' file for details.
**/

#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include <functional>

typedef enum {
  SIGMOLD,
  ACTIVATION_TYPE_MAX
} ACTIVATION_TYPE;

typedef std::function<double(double)>  ACTIVATION_FUNC;

/**
  Get activation function of specified activation type.

  @param[in]  Type  An activation type.

  @return  Activation function of Type.

**/
ACTIVATION_FUNC
GetActivationFunction (
  ACTIVATION_TYPE  Type
  );

/**
  Get deriative activation function of specified activation type.

  @param[in]  Type  An activation type.

  @return  Deriative activation function of Type.

**/
ACTIVATION_FUNC
GetDeriativeActivationFunction (
  ACTIVATION_TYPE  Type
  );

#endif