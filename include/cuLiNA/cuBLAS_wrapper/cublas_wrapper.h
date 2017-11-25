//
// Created by spades on 31/05/17.
//

#ifndef CGMAPPING_CUBLAS_HANDLER_WRAPPER_H
#define CGMAPPING_CUBLAS_HANDLER_WRAPPER_H

#include <stdio.h>
#include <stdlib.h>
#include <cuLiNA/culina_base_matrix.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace cuBLAS_wrapper {
    
    typedef enum {
        
        MULTIPLICATION,
        ADDITION,
        SUBTRACTION,
        DIVISION,
        TRANSPOSE
        
    } cublas_wrapper_op_type;
    
    class cublas_wrapper {
        
        static cublasHandle_t cublas_handle_;
        static cublasStatus_t stat_;
     
     public:
        
        cublas_wrapper() {};
        
        static cublasStatus_t _start_cublas_handle_wrapper();
        
        /***
         *
         * This method encapsulates cuBLAS multiplications functions cublasDdot, cublasDgemv and cublasDgemm, that respectevely
         * deal with: dot product of two vectors; multiplication of a matrix by a vector; and finally matrix by matrix multiplication.
         * Note that this method is designed to work with only double-type cublas_matrix.
         *
         * Here the following operations are carried out
         *
         *    - in case both are vectors the dot product is
         *
         *        result_matrix = trans(cu_matrix1)*cu_matrix2
         *
         *    - in case cu_matrix1 is a matrix and cu_matrix2 is a vector
         *
         *        result_matrix = alpha*op(cu_matrix1)*op(cu_matrix2) + beta*result_matrix
         *
         *    - in case cu_matrix1 and cu_matrix2 are both matrices
         *
         *        result_matrix = alpha*op(cu_matrix1)*cu_matrix2 + beta*result_matrix
         *
         * Observe that op() is a operation carried over the matrix previous to multiplication, as described in cuBLAS
         * documentation http://docs.nvidia.com/cuda/cublas/index.html#axzz4jLnuwBI8. This operation can assum the following
         * values:
         *
         *    - CUBLAS_OP_N
         *
         *        do not transpose matrix, basically, let matrix as it is
         *
         *    - CUBLAS__OP_T
         *
         *        transpose matrix
         *
         *    - CUBLAS_OP_C
         *
         *        use conjugate transpose on matrix
         *
         * The parameters alpha and beta are just scalars, and notice that beta is responsible to carry out previous information
         * stored in result_matrix. Also about the result_matrix, it has to be already instatiated to the size desired and
         * one cannot pass as a result_matrix a matrix being used as cu_matrix1 or cu_matrix2, although one can use the same matrix
         * to both cu_matrices.
         *
         *
         * @param [in] cu_matrix1
         * @param [in] cu_matrix2
         * @param [out] result_matrix
         * @param [in] op_m1 = CUBLAS_OP_N
         * @param [in] op_m2 = CUBLAS_OP_N
         * @param [in] alpha = 1
         * @param [in] beta = 0
         *
         */
        static cublasStatus_t _cublas_Dmultiplication(cuLiNA::culina_base_matrix<double> &cu_matrix1,
                                                      cuLiNA::culina_base_matrix<double> &cu_matrix2,
                                                      cuLiNA::culina_base_matrix<double> &result_matrix,
                                                      cublasOperation_t op_m1 = CUBLAS_OP_N,
                                                      cublasOperation_t op_m2 = CUBLAS_OP_N,
                                                      double alpha = 1,
                                                      double beta = 0,
                                                      cudaStream_t *strm = NULL);
        
        /***
         *
         * This method encapsulates cuBLAS multiplications functions cublasSdot, cublasSgemv and cublasSgemm, that respectevely
         * deal with: dot product of two vectors; multiplication of a matrix by a vector; and finally matrix by matrix multiplication.
         * Note that this method is designed to work with only float-type cublas_matrix.
         *
         * Here the following operations are carried out
         *
         *    - in case both are vectors the dot product is
         *
         *        result_matrix = trans(cu_matrix1)*cu_matrix2
         *
         *    - in case cu_matrix1 is a matrix and cu_matrix2 is a vector
         *
         *        result_matrix = alpha*op(cu_matrix1)*op(cu_matrix2) + beta*result_matrix
         *
         *    - in case cu_matrix1 and cu_matrix2 are both matrices
         *
         *        result_matrix = alpha*op(cu_matrix1)*cu_matrix2 + beta*result_matrix
         *
         * Observe that op() is a operation carried over the matrix previous to multiplication, as described in cuBLAS
         * documentation http://docs.nvidia.com/cuda/cublas/index.html#axzz4jLnuwBI8. This operation can assum the following
         * values:
         *
         *    - CUBLAS_OP_N
         *
         *        do not transpose matrix, basically, let matrix as it is
         *
         *    - CUBLAS__OP_T
         *
         *        transpose matrix
         *
         *    - CUBLAS_OP_C
         *
         *        use conjugate transpose on matrix
         *
         * The parameters alpha and beta are just scalars, and notice that beta is responsible to carry out previous information
         * stored in result_matrix. Also about the result_matrix, it has to be already instatiated to the size desired and
         * one cannot pass as a result_matrix a matrix being used as cu_matrix1 or cu_matrix2, although one can use the same matrix
         * to both cu_matrices.
         *
         *
         * @param [in] cu_matrix1
         * @param [in] cu_matrix2
         * @param [out] result_matrix
         * @param [in] op_m1 = CUBLAS_OP_N
         * @param [in] op_m2 = CUBLAS_OP_N
         * @param [in] alpha = 1
         * @param [in] beta = 0
         *
         */
        static cublasStatus_t _cublas_Smultiplication(cuLiNA::culina_base_matrix<float> &cu_matrix1,
                                                      cuLiNA::culina_base_matrix<float> &cu_matrix2,
                                                      cuLiNA::culina_base_matrix<float> &result_matrix,
                                                      cublasOperation_t op_m1 = CUBLAS_OP_N,
                                                      cublasOperation_t op_m2 = CUBLAS_OP_N,
                                                      float alpha = 1,
                                                      float beta = 0,
                                                      cudaStream_t *strm = NULL);
    
        //TODO: create comments explaining the function using doxygen format
        static cublasStatus_t _cublas_Dsum(cuLiNA::culina_base_matrix<double> &cu_matrix1,
                                           cuLiNA::culina_base_matrix<double> &cu_matrix2,
                                           cuLiNA::culina_base_matrix<double> &result_matrix,
                                           cublasOperation_t op_m1,
                                           cublasOperation_t op_m2,
                                           double alpha,
                                           double beta,
                                           cudaStream_t *strm);
    
        //TODO: create comments explaining the function using doxygen format
        static cublasStatus_t _cublas_Ssum(cuLiNA::culina_base_matrix<float> &cu_matrix1,
                                           cuLiNA::culina_base_matrix<float> &cu_matrix2,
                                           cuLiNA::culina_base_matrix<float> &result_matrix,
                                           cublasOperation_t op_m1,
                                           cublasOperation_t op_m2,
                                           float alpha,
                                           float beta,
                                           cudaStream_t *strm);
    
        //TODO: create comments explaining the function using doxygen format
        static cublasStatus_t _cublas_Ddiag_multiplication(cuLiNA::culina_base_matrix<double> &cu_matrix,
                                                           cuLiNA::culina_base_matrix<double> &diag_matrix,
                                                           cuLiNA::culina_base_matrix<double> &result_matrix,
                                                           cublasSideMode_t mode,
                                                           cudaStream_t *strm);
    
        //TODO: create comments explaining the function using doxygen format
        static cublasStatus_t _cublas_Dinverse(cuLiNA::culina_base_matrix<double> &cu_matrix,
                                               cuLiNA::culina_base_matrix<double> &result_matrix,
                                               int *info,
                                               cudaStream_t *strm);
        
        //TODO: create comments explaining the function using doxygen format
        static cublasStatus_t _cublas_Dtriangular_system_solver(cuLiNA::culina_base_matrix<double> &cu_matrix1,
                                                                cuLiNA::culina_base_matrix<double> &result_matrix,
                                                                double alpha,
                                                                cublasSideMode_t side = CUBLAS_SIDE_LEFT,
                                                                cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER,
                                                                cublasOperation_t op_m1 = CUBLAS_OP_N,
                                                                cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT,
                                                                cudaStream_t *strm = NULL);
    
        //TODO: create comments explaining the function using doxygen format
        static cublasStatus_t _cublas_Dnorm(cuLiNA::culina_base_matrix<double> &cu_matrix,
                                            double *result,
                                            cudaStream_t *strm = NULL);
        
        static std::string _cublasGetErrorString(cublasStatus_t stat);
        
        static void _cublasCheckErrors(cublasStatus_t stat,
                                              const std::string &file,
                                              const std::string &function,
                                              const int line);
        
        static cublasHandle_t &_getCublas_handle() {
            return cublas_wrapper::cublas_handle_;
        }
        
        static cublasStatus_t &_getStat() {
            return cublas_wrapper::stat_;
        }
        
         ~cublas_wrapper() {
            
            cublasDestroy_v2(cublas_wrapper::cublas_handle_);
            
        }
        
    };
    
}

#endif //CGMAPPING_CUBLAS_HANDLER_WRAPPER_H
