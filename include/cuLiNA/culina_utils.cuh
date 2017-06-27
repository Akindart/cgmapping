//
// Created by spades on 07/06/17.
//

#ifndef CGMAPPING_CUBLAS_WRAPPER_UTILS_H
#define CGMAPPING_CUBLAS_WRAPPER_UTILS_H

#include <iostream>
#include <cuda_parsing_helper_in_clion/clion_helper.h>
#include <cuLiNA/culina_utils_kernels.cuh>
#include <vector_types.h>
#include <cuda_device_properties/cuda_device_properties.h>
#include "cuLiNA/culina_data_types.h"

namespace cuLiNA {
    
    __host__
    extern cuLiNA::cuLiNA_error_t set_identity_matrix(double *d_matrix,
                                                      int n_rows,
                                                      int n_columns,
                                                      cudaStream_t *strm = NULL);
        
    
}

#endif //CGMAPPING_CUBLAS_WRAPPER_UTILS_H
