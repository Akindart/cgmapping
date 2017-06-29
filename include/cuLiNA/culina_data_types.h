//
// Created by spades on 14/06/17.
//

/*
 *
 * This file contains only data types that are used by other pieces of this code.
 * Note that here no definition of template is done.
 *
 *
 * */

#ifndef CGMAPPING_CUBLAS_WRAPPER_DATA_TYPES_H
#define CGMAPPING_CUBLAS_WRAPPER_DATA_TYPES_H

#include <cublas_v2.h>

namespace cuLiNA {
    
    typedef enum {
        
        CULINA_SUCCESS, ///<- if everything turns ok
        CULINA_PARAMETERS_MSIMATCH, ///<-if something went wrong with between parameters passed to function
        
    } cuLiNA_error_t;
    
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
        T *workspace = NULL;
        uint workspace_ld = 0;
        T *TAU = NULL;
        int *dev_info;
        
        cudaError_t _allocateWorkspace(uint size_of_workspace){
            
            cudaError_t stat;
            
            if(workspace != NULL) {
                
                stat = cudaFree(this->workspace);
            
            }
            else {
            
                stat = cudaMalloc((void **) &workspace, sizeof(T) * size_of_workspace);
                
                
            
            }
            
            
            return cudaSuccess;
            
        };
        cudaError_t _allocate_dev_info(){
            
            
            return cudaSuccess;
            
        };
    
    } ;
    
    
}

#endif //CGMAPPING_CUBLAS_WRAPPER_DATA_TYPES_H
