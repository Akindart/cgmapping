//
// Created by spades on 19/06/17.
//

#include <cuLiNA/cuSOLVER_wrapper/cusolver_wrapper.h>
#include <iostream>

using namespace cuSOLVER_wrapper;

cusolverDnHandle_t cusolver_wrapper::cusolverDn_handle_;
cusolverStatus_t cusolver_wrapper::stat_;

cusolverStatus_t cusolver_wrapper::_start_cusolverDn_handle_wrapper() {
    
    stat_ = cusolverDnCreate(&cusolverDn_handle_);
    
    if (stat_ != CUSOLVER_STATUS_SUCCESS) {

#ifndef DEBUG
        
        std::err << "File: " << __FILE__ << " - ERROR INFO SUPPRESSED - use SET(CMAKE_CXX_FLAGS_DEBUG \"-DDEBUG\") to see it" << std::endl;

#endif
#ifdef DEBUG
        
        std::cerr << std::endl << "###########################################################################"
                  << std::endl << std::endl;
        
        std::cerr << "ERROR HAPPENED FROM WITHIN cuSOLVER_wrapper::cusolver_wrapper::" << __FUNCTION__ << std::endl;
        std::cerr << "File: \"" << __FILE__ << "\"." << std::endl;
        std::cerr << "CUBLAS handle initialization failed." << std::endl;
        
        std::cerr << std::endl << "###########################################################################"
                  << std::endl;

#endif
    
    }
    
    return stat_;
    
}

cusolverStatus_t cusolver_wrapper::_cusolver_Dqr_factorization(cuLiNA::culina_base_matrix<double> &result_matrix,
                                                               cuLiNA::culina_base_matrix<double> &workspace,
                                                               double *TAU,
                                                               int *devInfo) {
    
    int m = result_matrix._getRows();
    int n = result_matrix._getColumns();
    
    return cusolverDnDgeqrf(cusolver_wrapper::cusolverDn_handle_,
                            m,
                            n,
                            result_matrix._getRawData(),
                            result_matrix._getLeading_dimension(),
                            TAU,
                            workspace._getRawData(),
                            workspace._getLeading_dimension(),
                            devInfo);
    
};

cusolverStatus_t cusolver_wrapper::_cusolver_Doperation_multiplication_qr(cuLiNA::culina_base_matrix<double> &cu_matrix,
                                                                          cuLiNA::culina_base_matrix<double> &result_matrix,
                                                                          cuLiNA::culina_base_matrix<double> &workspace,
                                                                          double *TAU,
                                                                          int *devInfo,
                                                                          cublasOperation_t op_m1,
                                                                          cublasSideMode_t side) {
    int m = cu_matrix._getRows();
    int n = cu_matrix._getColumns();
    int k = result_matrix._getColumns();
    
    int ld_cu_matrix = cu_matrix._getLeading_dimension();
    int ld_result_matrix = result_matrix._getLeading_dimension();
    int ld_workspace = workspace._getLeading_dimension();
    
    return cusolverDnDormqr(cusolver_wrapper::cusolverDn_handle_,
                            side,
                            op_m1,
                            m,
                            n,
                            k,
                            cu_matrix._getRawData(),
                            ld_cu_matrix,
                            TAU,
                            result_matrix._getRawData(),
                            ld_result_matrix,
                            workspace._getRawData(),
                            ld_workspace,
                            devInfo);
    
    
};