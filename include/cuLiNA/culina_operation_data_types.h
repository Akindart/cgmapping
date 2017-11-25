//
// Created by spades on 30/06/17.
//

#ifndef CGMAPPING_CULINA_OPERATION_DATA_TYPES_H
#define CGMAPPING_CULINA_OPERATION_DATA_TYPES_H

#include <cublas_v2.h>

namespace cuLiNA{
    
    typedef enum{
        
        GEQRF_BUFFER, ///<- buffer to be used when inverting matrices before multiplication
        
    }cuLiNA_buffer_t;
    
    typedef enum{
        
        CULINA_INVERSE_OFF, ///<- if the matrix is not required to be inverted during operation
        CULINA_INVERSE_ON,  ///<- if the matrix is required to be inverted during operation
        
    }cuLiNA_operation_t;
    
    template<typename T>
    struct cuLiNA_operation_parameters_t{
        
        cublasOperation_t op_m1 = CUBLAS_OP_N;
        cublasOperation_t op_m2 = CUBLAS_OP_N;
        T alpha = 1;
        T beta = 0;
        T gamma = 0; ///<-only used for sum
        cuLiNA_operation_t cuLiNA_op_m1 = CULINA_INVERSE_OFF;
        cuLiNA_operation_t cuLiNA_op_m2 = CULINA_INVERSE_OFF;
        culina_base_matrix<T> *workspace = NULL;
        T *d_TAU = NULL;
        int *dev_info = NULL;
        cudaStream_t *strm = NULL;
        
        ~cuLiNA_operation_parameters_t(){
            
            if(d_TAU != NULL)
                cudaFree(d_TAU);
            if(dev_info != NULL)
                cudaFree(d_TAU);
            
        }
        
    } ;
    
};

#endif //CGMAPPING_CULINA_OPERATION_DATA_TYPES_H
