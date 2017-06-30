//
// Created by spades on 30/06/17.
//

#ifndef CGMAPPING_CULINA_OPERATION_DATA_TYPES_H
#define CGMAPPING_CULINA_OPERATION_DATA_TYPES_H

#include <cublas_v2.h>

namespace cuLiNA{
    
    typedef enum{
        
        CULINA_INVERSE_OFF, ///<- if the matrix is not required to be inverted during operation
        CULINA_INVERSE_ON,  ///<- if the matrix is required to be inverted during operation
        
    }cuLiNA_operation_t;
    
    template<typename T>
    struct cuLiNA_operation_parameters_t{
        
        cublasOperation_t op_m1 = CUBLAS_OP_N;
        cublasOperation_t op_m2 = CUBLAS_OP_N;
        double alpha = 1;
        double beta = 0;
        cuLiNA_operation_t cuLiNA_op_m1 = CULINA_INVERSE_OFF;
        cuLiNA_operation_t cuLiNA_op_m2 = CULINA_INVERSE_OFF;
        culina_base_matrix<T> *workspace = NULL;
        T *d_TAU = NULL;
        int *dev_info;
        
    } ;
    
};

#endif //CGMAPPING_CULINA_OPERATION_DATA_TYPES_H
