//
// Created by spades on 01/06/17.
//

#include <cuLiNA/cuBLAS_wrapper/cublas_wrapper.h>
#include <general_utils.h>

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

cublasStatus_t cuBLAS_wrapper::cublas_wrapper::_cublas_Dmultiplication(cuLiNA::culina_tm<double> &cu_matrix1,
                                                                       cuLiNA::culina_tm<double> &cu_matrix2,
                                                                       cuLiNA::culina_tm<double> &result_matrix,
                                                                       cublasOperation_t op_m1,
                                                                       cublasOperation_t op_m2,
                                                                       double alpha,
                                                                       double beta,
                                                                       cudaStream_t *strm) {
    
    if(strm != NULL)
        cublasSetStream_v2(cublas_wrapper::cublas_handle_, *strm);
   else cublasSetStream_v2(cublas_wrapper::cublas_handle_, 0);
    
    int m = cu_matrix1._getRows();
    int n = cu_matrix2._getColumns();
    int k = cu_matrix1._getColumns();
    int l = cu_matrix2._getRows();
    
    //std::cout << m << " " << n << " " << k << std::endl;
    
    int ld_cu_m1 = cu_matrix1._getLeading_dimension();
    int ld_cu_m2 = cu_matrix2._getLeading_dimension();
    int ld_result = result_matrix._getLeading_dimension();
    
    if (n > 1 && k > 1)
        return cublasDgemm_v2(cublas_wrapper::_getCublas_handle(),
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

    else if (n == 1 && k > 1)
        return cublasDgemv_v2(cublas_wrapper::_getCublas_handle(),
                              op_m1,
                              m,
                              k,
                              &alpha,
                              cu_matrix1._getRawData(),
                              ld_cu_m1,
                              cu_matrix2._getRawData(),
                              n,
                              &beta,
                              result_matrix._getRawData(),
                              n);
    
    else if (n == 1 && k == 1)
        return cublasDdot_v2(cublas_wrapper::_getCublas_handle(),
                             m,
                             cu_matrix1._getRawData(),
                             n,
                             cu_matrix2._getRawData(),
                             k,
                             result_matrix._getRawData());
    
    
};

cublasStatus_t cuBLAS_wrapper::cublas_wrapper::_cublas_Smultiplication(cuLiNA::culina_tm<float> &cu_matrix1,
                                                                       cuLiNA::culina_tm<float> &cu_matrix2,
                                                                       cuLiNA::culina_tm<float> &result_matrix,
                                                                       cublasOperation_t op_m1,
                                                                       cublasOperation_t op_m2,
                                                                       float alpha,
                                                                       float beta,
                                                                       cudaStream_t *strm) {
    
    if(strm != NULL)
        cublasSetStream_v2(cublas_wrapper::cublas_handle_, *strm);
    else cublasSetStream_v2(cublas_wrapper::cublas_handle_, 0);
    
    int m = cu_matrix1._getRows();
    int n = cu_matrix1._getColumns();
    int k = cu_matrix2._getColumns();
    
    int ld_cu_m1 = cu_matrix1._getLeading_dimension();
    int ld_cu_m2 = cu_matrix2._getLeading_dimension();
    int ld_result = result_matrix._getLeading_dimension();
    
    cublasStatus_t stat;
    
    if (n > 1 && k > 1)
        stat = cublasSgemm_v2(cublas_wrapper::_getCublas_handle(),
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
        stat = cublasSgemv_v2(cublas_wrapper::_getCublas_handle(),
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
        stat = cublasSdot_v2(cublas_wrapper::_getCublas_handle(),
                             m,
                             cu_matrix1._getRawData(),
                             n,
                             cu_matrix2._getRawData(),
                             k,
                             result_matrix._getRawData());
    
    _cublasCheckErrors(stat, __FILE__, __FUNCTION__, 0);
    
    return stat;
    
};

cublasStatus_t cuBLAS_wrapper::cublas_wrapper::_cublas_Dsum(cuLiNA::culina_tm<double> &cu_matrix1,
                                                            cuLiNA::culina_tm<double> &cu_matrix2,
                                                            cuLiNA::culina_tm<double> &result_matrix,
                                                            cublasOperation_t op_m1,
                                                            cublasOperation_t op_m2,
                                                            double alpha,
                                                            double beta,
                                                            cudaStream_t *strm) {
    
    if(strm != NULL)
        cublasSetStream_v2(cublas_wrapper::cublas_handle_, *strm);
    else cublasSetStream_v2(cublas_wrapper::cublas_handle_, 0);
    
    return cublasDgeam(cublas_wrapper::_getCublas_handle(),
                       op_m1,
                       op_m2,
                       cu_matrix1._getRows(),
                       cu_matrix2._getColumns(),
                       &alpha,
                       cu_matrix1._getRawData(),
                       cu_matrix1._getLeading_dimension(),
                       &beta,
                       cu_matrix2._getRawData(),
                       cu_matrix2._getLeading_dimension(),
                       result_matrix._getRawData(),
                       result_matrix._getLeading_dimension());
    
};

cublasStatus_t cuBLAS_wrapper::cublas_wrapper::_cublas_Ddiag_multiplication(cuLiNA::culina_tm<double> &cu_matrix,
                                                                            cuLiNA::culina_tm<double> &diag_matrix,
                                                                            cuLiNA::culina_tm<double> &result_matrix,
                                                                            cublasSideMode_t mode,
                                                                            cudaStream_t *strm) {
    
    if(strm != NULL)
        cublasSetStream_v2(cublas_wrapper::cublas_handle_, *strm);
    else cublasSetStream_v2(cublas_wrapper::cublas_handle_, 0);
    
    int diagonal_stride = 1;
          //ensures diagonal matrix is represented by either a vector with stride one or a diagonal matrix
    if(diag_matrix._getColumns() > 1 && diag_matrix._isSquare())
            diagonal_stride  = diag_matrix._getLeading_dimension() + 1;
    else {
    
        cublas_wrapper::_cublasCheckErrors(CUBLAS_STATUS_INVALID_VALUE, __FILE__, __FUNCTION__, __LINE__);
        return CUBLAS_STATUS_INVALID_VALUE;
        
    }
    
    return cublasDdgmm(cublas_wrapper::_getCublas_handle(),
                       mode,
                       result_matrix._getRows(),
                       result_matrix._getColumns(),
                       cu_matrix._getRawData(),
                       cu_matrix._getLeading_dimension(),
                       diag_matrix._getRawData(),
                       diagonal_stride,
                       result_matrix._getRawData(),
                       result_matrix._getLeading_dimension());
    
}

cublasStatus_t cuBLAS_wrapper::cublas_wrapper::_cublas_Dinverse(cuLiNA::culina_tm<double> &cu_matrix,
                                                                cuLiNA::culina_tm<double> &result_matrix,
                                                                int *info,
                                                                cudaStream_t *strm){
    
    if(strm != NULL)
        cublasSetStream_v2(cublas_wrapper::cublas_handle_, *strm);
    else cublasSetStream_v2(cublas_wrapper::cublas_handle_, 0);
    
    if(!cu_matrix._isSquare()){
        
        cublas_wrapper::_cublasCheckErrors(CUBLAS_STATUS_INVALID_VALUE, __FILE__, __FUNCTION__, __LINE__);
        return CUBLAS_STATUS_INVALID_VALUE;
        
    }
    
    const double *raw_data_cu_matrix = cu_matrix._getRawData();
    double *raw_data_result_matrix = result_matrix._getRawData();
    
    return cublasDmatinvBatched(cublas_wrapper::_getCublas_handle(),
                                cu_matrix._getRows(),
                                &raw_data_cu_matrix,
                                cu_matrix._getLeading_dimension(),
                                &raw_data_result_matrix,
                                result_matrix._getLeading_dimension(),
                                info,
                                1);
    
}


cublasStatus_t cuBLAS_wrapper::cublas_wrapper::_cublas_Dtriangular_system_solver(cuLiNA::culina_tm<double> &cu_matrix1,
                                                                                 cuLiNA::culina_tm<double> &result_matrix,
                                                                                 double alpha,
                                                                                 cublasSideMode_t side,
                                                                                 cublasFillMode_t uplo,
                                                                                 cublasOperation_t op_m1,
                                                                                 cublasDiagType_t diag,
                                                                                 cudaStream_t *strm) {
    
    
    
    if(strm != NULL)
        cublasSetStream_v2(cublas_wrapper::cublas_handle_, *strm);
    else cublasSetStream_v2(cublas_wrapper::cublas_handle_, 0);
    
    int m = cu_matrix1._getColumns();
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

void cuBLAS_wrapper::cublas_wrapper::_cublasCheckErrors(cublasStatus_t stat,
                                                       const std::string &file,
                                                       const std::string &function,
                                                       const int line) {
    
    if (stat != CUBLAS_STATUS_SUCCESS) {
        
        std::cerr << std::endl << "###########################################################################"
                  << std::endl << std::endl;
        
        std::cerr << "ERROR HAPPENED FROM WITHIN " << function << std::endl;
        std::cerr << "File: \"" << file << "\"." << std::endl;
        std::cerr << "CUDA ERROR: " << _cublasGetErrorString(stat) << std::endl;
        if(line != 0)
            std::cerr << "Line: \"" << line << "\"." << std::endl;
        
        std::cerr << std::endl << "###########################################################################"
                  << std::endl;
        
    }
    
};

std::string cuBLAS_wrapper::cublas_wrapper::_cublasGetErrorString(cublasStatus_t stat) {
    
    switch (stat) {
        
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        
    };
    
}

cublasStatus_t cuBLAS_wrapper::cublas_wrapper::_cublas_Dnorm(cuLiNA::culina_tm<double> &cu_matrix,
                                                             double *result,
                                                             cudaStream_t *strm) {
    
    if(strm != NULL)
        cublasSetStream_v2(cublas_wrapper::cublas_handle_, *strm);
    else cublasSetStream_v2(cublas_wrapper::cublas_handle_, 0);
    
    cublasStatus_t stat;
    
    stat = cublasDnrm2_v2(cublas_wrapper::_getCublas_handle(),
                          cu_matrix._getRows(),
                          cu_matrix._getRawData(),
                          1,
                          result);
    
    cublas_wrapper::_cublasCheckErrors(stat, __FILE__, __FUNCTION__, 0);
    
    return stat;
}

