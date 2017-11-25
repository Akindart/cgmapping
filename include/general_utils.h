//
// Created by spades on 29/06/17.
//

#ifndef CGMAPPING_GENERAL_UTILS_H
#define CGMAPPING_GENERAL_UTILS_H

#include <iostream>
#include <cuda_runtime.h>

extern void cudaCheckErrors(cudaError_t stat, const std::string &file, const std::string &function, const int line = 0);


#endif //CGMAPPING_GENERAL_UTILS_H
