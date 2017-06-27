//
// Created by spades on 01/06/17.
//

#include <cuLiNA/cuBLAS_wrapper/cublas_wrapper.h>

//need declarations of static members

cublasHandle_t cuBLAS_wrapper::cublas_wrapper::cublas_handle_;
cublasStatus_t cuBLAS_wrapper::cublas_wrapper::stat_;

cublasStatus_t cuBLAS_wrapper::cublas_wrapper::_start_cublas_handle_wrapper() {
    
    stat_ = cublasCreate_v2(&cublas_handle_);
    
    if (stat_ != CUBLAS_STATUS_SUCCESS) {

#ifndef DEBUG
        
        std::cout << "File: " << __FILE__ << " - ERROR INFO SUPPRESSED - use SET(CMAKE_CXX_FLAGS_DEBUG \"-DDEBUG\") to see it" << std::endl;

#endif
#ifdef DEBUG
        
        std::cerr << std::endl << "###########################################################################"
                  << std::endl << std::endl;
        
        std::cerr << "ERROR HAPPENED FROM WITHIN cuBLAS_wrapper::cublas_wrapper::" << __FUNCTION__ << std::endl;
        std::cerr << "File: \"" << __FILE__ << "\"." << std::endl;
        std::cerr << "CUBLAS handle initialization failed." << std::endl;
        
        std::cerr << std::endl << "###########################################################################"
                  << std::endl;

#endif
    
    }
    
    return stat_;
    
};

cublasStatus_t cuBLAS_wrapper::cublas_wrapper::_cublas_Dmultiplication(cuLiNA::culina_base_matrix<double> &cu_matrix1,
                                                                       cuLiNA::culina_base_matrix<double> &cu_matrix2,
                                                                       cuLiNA::culina_base_matrix<double> &result_matrix,
                                                                       cublasOperation_t op_m1,
                                                                       cublasOperation_t op_m2,
                                                                       double alpha,
                                                                       double beta) {
    
    int m = cu_matrix1._getRows();
    int n = cu_matrix1._getColumns();
    int k = cu_matrix2._getColumns();
    
    int ld_cu_m1 = cu_matrix1._getLeading_dimension();
    int ld_cu_m2 = cu_matrix2._getLeading_dimension();
    int ld_result = result_matrix._getLeading_dimension();
    
    if (n > 1 && k > 1)
        return cublasDgemm(cublas_wrapper::_getCublas_handle(),
                           op_m1,
                           op_m2,
                           m,
                           n,
                           k,
                           &alpha,
                           cu_matrix1._getRawData(),
                           ld_cu_m1,
                           cu_matrix2._getRawData(),
                           ld_cu_m2,
                           &beta,
                           result_matrix._getRawData(),
                           ld_result);
    
    else if (n > 1 && k == 1)
        return cublasDgemv(cublas_wrapper::_getCublas_handle(),
                           op_m1,
                           m,
                           n,
                           &alpha,
                           cu_matrix1._getRawData(),
                           ld_cu_m1,
                           cu_matrix2._getRawData(),
                           k,
                           &beta,
                           result_matrix._getRawData(),
                           k);
    
    else if (n == 1 && k == 1)
        return cublasDdot_v2(cublas_wrapper::_getCublas_handle(),
                             m,
                             cu_matrix1._getRawData(),
                             n,
                             cu_matrix2._getRawData(),
                             k,
                             result_matrix._getRawData());
    
};

cublasStatus_t cuBLAS_wrapper::cublas_wrapper::_cublas_Smultiplication(cuLiNA::culina_base_matrix<float> &cu_matrix1,
                                                                       cuLiNA::culina_base_matrix<float> &cu_matrix2,
                                                                       cuLiNA::culina_base_matrix<float> &result_matrix,
                                                                       cublasOperation_t op_m1,
                                                                       cublasOperation_t op_m2,
                                                                       float alpha,
                                                                       float beta) {
    
    int m = cu_matrix1._getRows();
    int n = cu_matrix1._getColumns();
    int k = cu_matrix2._getColumns();
    
    int ld_cu_m1 = cu_matrix1._getLeading_dimension();
    int ld_cu_m2 = cu_matrix2._getLeading_dimension();
    int ld_result = result_matrix._getLeading_dimension();
    
    if (n > 1 && k > 1)
        return cublasSgemm(cublas_wrapper::_getCublas_handle(),
                           op_m1,
                           op_m2,
                           m,
                           n,
                           k,
                           &alpha,
                           cu_matrix1._getRawData(),
                           ld_cu_m1,
                           cu_matrix2._getRawData(),
                           ld_cu_m2,
                           &beta,
                           result_matrix._getRawData(),
                           ld_result);
    
    else if (n > 1 && k == 1)
        return cublasSgemv(cublas_wrapper::_getCublas_handle(),
                           op_m1,
                           m,
                           n,
                           &alpha,
                           cu_matrix1._getRawData(),
                           ld_cu_m1,
                           cu_matrix2._getRawData(),
                           k,
                           &beta,
                           result_matrix._getRawData(),
                           k);
    
    else if (n == 1 && k == 1)
        return cublasSdot_v2(cublas_wrapper::_getCublas_handle(),
                             m,
                             cu_matrix1._getRawData(),
                             n,
                             cu_matrix2._getRawData(),
                             k,
                             result_matrix._getRawData());
    
};

cublasStatus_t cuBLAS_wrapper::cublas_wrapper::_cublas_Dtriangular_system_solver(cuLiNA::culina_base_matrix<double> &cu_matrix1,
                                                                                 cuLiNA::culina_base_matrix<double> &result_matrix,
                                                                                 double alpha,
                                                                                 cublasSideMode_t side,
                                                                                 cublasFillMode_t uplo,
                                                                                 cublasOperation_t op_m1,
                                                                                 cublasDiagType_t diag) {
    
    int m = cu_matrix1._getRows();
    int n = result_matrix._getColumns();
    
    int ld_cu_m1 = cu_matrix1._getLeading_dimension();
    int ld_result = result_matrix._getLeading_dimension();
    
    return cublasDtrsm_v2(cublas_wrapper::_getCublas_handle(),
                          side,
                          uplo,
                          op_m1,
                          diag,
                          m,
                          n,
                          &alpha,
                          cu_matrix1._getRawData(),
                          ld_cu_m1,
                          result_matrix._getRawData(),
                          ld_result);
    
};