//
// Created by spades on 15/06/17.
//

#ifndef CGMAPPING_CUSOLVER_WRAPPER_H
#define CGMAPPING_CUSOLVER_WRAPPER_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuLiNA/culina_base_matrix.h>

namespace cuSOLVER_wrapper {
    
    class cusolver_wrapper {
        
        static cusolverDnHandle_t cusolverDn_handle_;
        static cusolverStatus_t stat_;
     
     public:
        
        cusolver_wrapper() {};
        
        static cusolverStatus_t _start_cusolverDn_handle_wrapper();
        
        static const cusolverDnHandle_t &_getCusolverDn_handle() {
            return cusolverDn_handle_;
        }
        static void _setCusolver_handle(const cusolverDnHandle_t &cusolverDn_handle_) {
            cusolver_wrapper::cusolverDn_handle_ = cusolverDn_handle_;
        }
        static cusolverStatus_t &_getStat() {
            return stat_;
        }
        static void setStat_(cusolverStatus_t stat_) {
            cusolver_wrapper::stat_ = stat_;
        }
        
        //TODO: create function comments
        static cusolverStatus_t _cusolver_Dqr_factorization(cuLiNA::culina_tm<double> &result_matrix,
                                                            cuLiNA::culina_tm<double> &workspace,
                                                            double *TAU,
                                                            int *devInfo,
                                                            cudaStream_t *strm = NULL);
    
        //TODO: create function comments
        static cusolverStatus_t _cusolver_Doperation_multiplication_qr(cuLiNA::culina_tm<double> &cu_matrix,
                                                                       cuLiNA::culina_tm<double> &result_matrix,
                                                                       cuLiNA::culina_tm<double> &workspace,
                                                                       double *TAU,
                                                                       int *devInfo,
                                                                       cublasOperation_t op_m1 = CUBLAS_OP_N,
                                                                       cublasSideMode_t side = CUBLAS_SIDE_LEFT,
                                                                       cudaStream_t *strm = NULL);
        
        static cusolverStatus_t _cusolver_Dgeqrf_bufferSize(cuLiNA::culina_tm<double> &cu_matrix,
                                                            int *lwork,
                                                            cudaStream_t *strm = NULL);
        
        static std::string _cusolver_wrapper_get_cusolver_error(cusolverStatus_t stat);
        static void _cusolverCheckErrors(cusolverStatus_t stat, const std::string &file, const std::string &function, const int line = 0);
        
        ~cusolver_wrapper(){
            
            cusolverDnDestroy(cusolverDn_handle_);
            
        }
        
    };
    
}

#endif //CGMAPPING_CUSOLVER_WRAPPER_H
