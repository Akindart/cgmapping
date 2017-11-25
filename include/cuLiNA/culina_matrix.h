//
// Created by spades on 29/05/17.
//

#ifndef CGMAPPING_CUBLAS_MAT_H
#define CGMAPPING_CUBLAS_MAT_H

#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuLiNA/culina_base_matrix.h>
#include <cublas_v2.h>
#include <cuLiNA/cuBLAS_wrapper/cublas_wrapper.h>

namespace cuLiNA {
    
    template<typename T, int rows = 0, int columns = rows, int leading_dimension = rows, typename Alloc = culina_matrix_allocator<T>>
    class culina_matrix : public culina_base_matrix<T> {
     
     public:
        
        culina_matrix(matrix_advanced_initialization_t mai = NOTHING) : culina_base_matrix<T, Alloc>(rows, columns, leading_dimension, mai) {};
        
        culina_matrix(thrust::device_vector<T> &data) : culina_base_matrix<T>(data,
                                                                              rows,
                                                                              columns,
                                                                              leading_dimension) {};
    
        culina_matrix(thrust::host_vector<T> &data) : culina_base_matrix<T>(data,
                                                                              rows,
                                                                              columns,
                                                                              leading_dimension) {};
        
        
    
    };
    
}

#endif //CGMAPPING_CUBLAS_MAT_H
