//
// Created by spades on 23/06/17.
//

#include <cuLiNA/culina_operations.h>

using namespace cuLiNA;
using namespace cuSOLVER_wrapper;
using namespace cuBLAS_wrapper;

cuLiNA::cuLiNA_error_t cuLiNA::culina_matrix_Dmultiplication(cuLiNA::culina_base_matrix<double> *cu_matrix1,
                                                             cuLiNA::culina_base_matrix<double> *cu_matrix2,
                                                             cuLiNA::culina_base_matrix<double> *cu_matrix3,
                                                             cuLiNA::culiopD_t &culiopD) {
    
    /***
    *
    * if the left-most matrix of the right hand-side part of the equation is supposed to be inverted before
    * multiplication.
    *
    * */
    if (culiopD.cuLiNA_op_m1 == CULINA_INVERSE_ON) {
        
        culina_base_matrix<double> *workspace = culiopD.workspace;
        
        cublasStatus_t cublas_stat = CUBLAS_STATUS_SUCCESS;
        cusolverStatus_t cusolver_stat1 = CUSOLVER_STATUS_SUCCESS;
        cusolverStatus_t cusolver_stat2 = CUSOLVER_STATUS_SUCCESS;
        
        cusolver_stat1 = cuSOLVER_wrapper::cusolver_wrapper::_cusolver_Dqr_factorization(*cu_matrix1,
                                                                                         *workspace,
                                                                                         culiopD.d_TAU,
                                                                                         culiopD.dev_info);
        
        cusolver_wrapper::_cusolverCheckErrors(cusolver_stat1, __FILE__, __FUNCTION__);
        assert(cusolver_stat1 == CUSOLVER_STATUS_SUCCESS);
        
        cusolver_stat2 = cuSOLVER_wrapper::cusolver_wrapper::_cusolver_Doperation_multiplication_qr(*cu_matrix1, //Q
                                                                                                    *cu_matrix2, //b
                                                                                                    *workspace,
                                                                                                    culiopD.d_TAU,
                                                                                                    culiopD.dev_info,
                                                                                                    CUBLAS_OP_T);
    
        cusolver_wrapper::_cusolverCheckErrors(cusolver_stat2, __FILE__, __FUNCTION__);
        assert(cusolver_stat2 == CUSOLVER_STATUS_SUCCESS);
        
        cublas_stat = cuBLAS_wrapper::cublas_wrapper::_cublas_Dtriangular_system_solver(*cu_matrix1, *cu_matrix2);
    
        cublas_wrapper::_cublasCheckErrors(cublas_stat, __FILE__, __FUNCTION__);
        assert(cublas_stat == CUBLAS_STATUS_SUCCESS);
        
    }
    
    /**
     *
     * if the left-most matrix of right handside equation does not need to be inverted.
     *
     * */
    
    else {
        
        cublasStatus_t cublas_stat;
        
        cublas_stat = cuBLAS_wrapper::cublas_wrapper::_cublas_Dmultiplication(*cu_matrix1,
                                                                              *cu_matrix2,
                                                                              *cu_matrix3,
                                                                              culiopD.op_m1,
                                                                              culiopD.op_m2,
                                                                              culiopD.alpha,
                                                                              culiopD.beta);
    
        cublas_wrapper::_cublasCheckErrors(cublas_stat, __FILE__, __FUNCTION__);
        assert(cublas_stat == CUBLAS_STATUS_SUCCESS);
        
    }
    
    return CULINA_SUCCESS;
}
