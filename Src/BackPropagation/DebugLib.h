/**
  Debugging utilities.

  Copyright (c) 2025, visionaryr
  Licensed under the MIT License. See the accompanying 'LICENSE' file for details.
**/

#ifndef _DEBUG_LIB_H_
#define _DEBUG_LIB_H_

#include <iostream>

#ifdef DEBUG_ENABLED

#define DEBUG_LOG(msg) \
  do { \
    std::cout << "DEBUG (" << __FILE__ << ":" << __LINE__ << "): " << msg << std::endl; \
  } while (0)

#define DEBUG_START() \
  do {

#define DEBUG_END() \
  } while (0);

#else

#define DEBUG_LOG(msg) \
  do { \
  } while (0)

#define DEBUG_START() \
  do { \
    if (0) {

#define DEBUG_END() \
    } \
  } while (0);

#endif // #ifdef DEBUG_ENABLED

#endif