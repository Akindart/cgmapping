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
    
    template<typename T, int rows, int columns = rows, int leading_dimension = rows>
    class culina_matrix : public culina_base_matrix<T> {
     
     public:
        
        culina_matrix() : culina_base_matrix<T>(rows, columns, leading_dimension) {};
        
        culina_matrix(thrust::device_vector<T> &data) : culina_base_matrix<T>(data,
                                                                              rows,
                                                                              columns,
                                                                              leading_dimension) {};
        
        inline int _loadData(T *h_data) {
            
            cublasStatus_t stat;
            
            stat = cublasSetMatrix(culina_base_matrix<T>::_getRows(),
                                   culina_base_matrix<T>::_getColumns(),
                                   sizeof(*h_data),
                                   h_data,
                                   culina_base_matrix<T>::_getLeading_dimension(),
                                   culina_base_matrix<T>::_getRawData(),
                                   culina_base_matrix<T>::_getLeading_dimension());
            if (stat != CUBLAS_STATUS_SUCCESS)
                std::cout << "shit happens when loading a matrix" << std::endl;
            
            return 1;
            
        };
        
        /***
         *
         * @param [in] h_data must've been pre-allocated outside this function and also be of the same
         * type of the cuda_matrix it's receiving information from
         *
         * */
        inline int _downloadData(T *h_data) {
            
            cublasStatus_t stat;
            
            stat = cublasGetMatrix(culina_base_matrix<T>::_getRows(),
                                   culina_base_matrix<T>::_getColumns(),
                                   sizeof(*h_data),
                                   culina_base_matrix<T>::_getRawData(),
                                   culina_base_matrix<T>::_getLeading_dimension(),
                                   h_data,
                                   culina_base_matrix<T>::_getLeading_dimension());
            
            if (stat != CUBLAS_STATUS_SUCCESS)
                std::cout << "shit happens when downloading a matrix" << std::endl;
            
            return 1;
            
        }
        
    };
    
}

#endif //CGMAPPING_CUBLAS_MAT_H
