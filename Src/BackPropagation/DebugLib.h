#ifndef _DEBUG_LIB_H_
#define _DEBUG_LIB_H_

#define DEBUG_LOG(msg) \
  do { \
    std::cout << "DEBUG (" << __FILE__ << ":" << __LINE__ << "): " << msg << std::endl; \
  } while (0)

#endif