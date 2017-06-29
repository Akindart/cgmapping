//
// Created by spades on 29/06/17.
//

#ifndef CGMAPPING_GENERAL_UTILS_H
#define CGMAPPING_GENERAL_UTILS_H

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuLiNA/cuBLAS_wrapper/cublas_wrapper.h>
#include <cuLiNA/cuSOLVER_wrapper/cusolver_wrapper.h>
#include <cuLiNA/culina_data_types.h>
#include <cuLiNA/culina_data_types.h>

extern inline cudaCheckErrors(cudaError_t stat, char *file, char *function);
extern inline cudaCheckErrors(cublasStatus_t stat, char *file, char *function);
extern inline cudaCheckErrors(cusolverStatus_t stat, char *file, char *function);
extern inline cudaCheckErrors(cuLiNA::cuLiNA_error_t stat, char *file, char *function);

#endif //CGMAPPING_GENERAL_UTILS_H
