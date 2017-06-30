//
// Created by spades on 23/06/17.
//

#include <cuLiNA/culina_operations.h>

using namespace cuLiNA;

cuLiNA::cuLiNA_error_t cuLiNA::culina_matrix_Dmultiplication(cuLiNA::culina_base_matrix<double> *cu_matrix1,
                                                             cuLiNA::culina_base_matrix<double> *cu_matrix2,
                                                             cuLiNA::culina_base_matrix<double> *cu_matrix3,
                                                             cuLiNA::culiopD_t &culiopD) {
    
    culina_base_matrix<double> *workspace = culiopD.workspace;
    
    /***
    *
    * if the left-most matrix of the right hand-side part of the equation is supposed to be inverted before
    * multiplication.
    *
    * */
    if (culiopD.cuLiNA_op_m1 == CULINA_INVERSE_ON) {
        
        cusolverCheckErrors(
            cuSOLVER_wrapper::cusolver_wrapper::_cusolver_Dqr_factorization(*cu_matrix1,
                                                                            *workspace,
                                                                            culiopD.d_TAU,
                                                                            culiopD.dev_info),
            __FILE__,
            __FUNCTION__);
        
        cusolverCheckErrors(
            cuSOLVER_wrapper::cusolver_wrapper::_cusolver_Doperation_multiplication_qr(*cu_matrix1, //Q
                                                                                       *cu_matrix2, //b
                                                                                       *workspace,
                                                                                       culiopD.d_TAU,
                                                                                       culiopD.dev_info,
                                                                                       CUBLAS_OP_T),
            __FILE__,
            __FUNCTION__);
        
        cublasCheckErrors(
            cuBLAS_wrapper::cublas_wrapper::_cublas_Dtriangular_system_solver(*cu_matrix1, *cu_matrix2),
            __FILE__,
            __FUNCTION__ );
    }
    
    return CULINA_SUCCESS;
}
