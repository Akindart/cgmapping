//
// Created by spades on 23/06/17.
//

#include <cuLiNA/culina_operations.h>
#include <cuLiNA/culina_base_matrix.h>
#include <cublas_v2.h>

using namespace cuLiNA;

cuLiNA_error_t culina_matrix_Dmultiplication(culina_base_matrix<double> *cu_matrix1,
                                             culina_base_matrix<double> *cu_matrix2,
                                             culina_base_matrix<double> *result_matrix,
                                             cublasOperation_t op_m1,
                                             cublasOperation_t op_m2,
                                             double alpha,
                                             double beta,
                                             cuLiNA_operation_t cuLiNA_op_m1,
                                             cuLiNA_operation_t cuLiNA_op_m2,
                                             culina_base_matrix<double> *workspace,
                                             double *TAU) {
    
    int *dev_info = NULL;
    
    /***
     *
     * if the left-most matrix of the right hand-side part of the equation is supposed to be inverted before
     * multiplication
     *
     * */
    if (cuLiNA_op_m1 == CULINA_INVERSE_ON) {
        
        cuSOLVER_wrapper::cusolver_wrapper::_cusolver_Dqr_factorization(*cu_matrix1, *workspace, TAU, dev_info);
        cuSOLVER_wrapper::cusolver_wrapper::_cusolver_Doperation_multiplication_qr(*cu_matrix1,
                                                                                   *cu_matrix2,
                                                                                   *workspace,
                                                                                   TAU,
                                                                                   dev_info,
                                                                                   CUBLAS_OP_T);
        cuBLAS_wrapper::cublas_wrapper::_cublas_Dtriangular_system_solver(*cu_matrix1, *cu_matrix2);
        
    }
    
    return CULINA_SUCCESS;
    
}

cuLiNA_error_t culina_matrix_Smultiplication(culina_base_matrix<float> *cu_matrix1,
                                             culina_base_matrix<float> *cu_matrix2,
                                             culina_base_matrix<float> *result_matrix,
                                             cublasOperation_t op_m1,
                                             cublasOperation_t op_m2,
                                             float alpha,
                                             float beta,
                                             cuLiNA_operation_t cuLiNA_op_m1,
                                             cuLiNA_operation_t cuLiNA_op_m2,
                                             culina_base_matrix<float> *workspace,
                                             float *TAU) {
    
    return CULINA_SUCCESS;
    
}
