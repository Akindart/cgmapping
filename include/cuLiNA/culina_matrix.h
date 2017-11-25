//
// Created by spades on 29/05/17.
//

#ifndef CGMAPPING_CUBLAS_MAT_H
#define CGMAPPING_CUBLAS_MAT_H

#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <cuLiNA/culina_base_matrix.h>
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
    
        culina_tm<T> &operator=(const culina_tm<T> &rhs) {

            return culina_tm<T>::operator=(rhs);
            
//            if (this->_getRows() == rhs._getRows() &&
//                this->_getColumns() == rhs._getColumns() &&
//                this->_getLeading_dimension() == rhs._getLeading_dimension() &&
//                this->_getNumber_of_elements() == rhs._getNumber_of_elements()) {
//
//                if (this->_getRawData() == const_cast<cuLiNA::culina_tm<T> &>(rhs)._getRawData())
//                    return *this;
//
//                auto stat = cudaMemcpyAsync((void *) this->_getRawData(),
//                                            (const void *) const_cast<cuLiNA::culina_tm<T> *>(&rhs)->_getRawData(),
//                                            sizeof(T) * this->_getNumber_of_elements(),
//                                            cudaMemcpyDeviceToDevice,
//                                            NULL);
//                cudaCheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
//
//            }
//
//            return *this;

        }


    };
    
}

#endif //CGMAPPING_CUBLAS_MAT_H
