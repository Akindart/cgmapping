//
// Created by spades on 16/06/17.
//

#ifndef CGMAPPING_CULINA_OPERATIONS_H
#define CGMAPPING_CULINA_OPERATIONS_H

#include <cuLiNA/cuBLAS_wrapper/cublas_wrapper.h>
#include <cuLiNA/culina_data_types.h>
#include <cuLiNA/cuSOLVER_wrapper/cusolver_wrapper.h>

namespace cuLiNA {
    
    //matrices are now pointers because there will be times when only one matrix is used
    
    /***
     *
     *  culinaOp[cublasOp(A)]
     *
     *
     * */
    
    //TODO: generate function definition, comments, and add option to invert cu_matrix1 or cu_matrix2 before multiplication procedure
    extern cuLiNA::cuLiNA_error_t culina_matrix_Dmultiplication(cuLiNA::culina_base_matrix<double> *cu_matrix1,
                                                                cuLiNA::culina_base_matrix<double> *cu_matrix2,
                                                                cuLiNA::culina_base_matrix<double> *cu_matrix3,
                                                                cublasOperation_t op_m1 = CUBLAS_OP_N,
                                                                cublasOperation_t op_m2 = CUBLAS_OP_N,
                                                                double alpha = 1,
                                                                double beta = 0,
                                                                cuLiNA_operation_t cuLiNA_op_m1 = CULINA_INVERSE_OFF,
                                                                cuLiNA_operation_t cuLiNA_op_m2 = CULINA_INVERSE_OFF,
                                                                culina_base_matrix<double> *workspace = NULL,
                                                                double *TAU = NULL);
    
    //TODO: generate function definition, comments, and add option to invert cu_matrix1 or cu_matrix2 before multiplication procedure
    extern cuLiNA::cuLiNA_error_t culina_matrix_Smultiplication(cuLiNA::culina_base_matrix<float> *cu_matrix1,
                                                                cuLiNA::culina_base_matrix<float> *cu_matrix2,
                                                                cuLiNA::culina_base_matrix<float> *cu_matrix3,
                                                                cublasOperation_t op_m1 = CUBLAS_OP_N,
                                                                cublasOperation_t op_m2 = CUBLAS_OP_N,
                                                                float alpha = 1,
                                                                float beta = 0,
                                                                cuLiNA_operation_t cuLiNA_op_m1 = CULINA_INVERSE_OFF,
                                                                cuLiNA_operation_t cuLiNA_op_m2 = CULINA_INVERSE_OFF,
                                                                cuLiNA::culina_base_matrix<float> *workspace = NULL,
                                                                float *TAU = NULL);
    
}

#endif //CGMAPPING_CULINA_OPERATIONS_H
