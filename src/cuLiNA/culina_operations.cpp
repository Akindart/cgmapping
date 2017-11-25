//
// Created by spades on 23/06/17.
//

#include <cuLiNA/culina_operations.h>

using namespace cuLiNA;
using namespace cuSOLVER_wrapper;
using namespace cuBLAS_wrapper;

cuLiNA::cuLiNA_error_t cuLiNA::culina_matrix_Dmultiplication(culina_tm<double> *cu_matrix1,
                                                             culina_tm<double> *cu_matrix2,
                                                             culina_tm<double> *cu_matrix3,
                                                             cuLiNA::culiopD_t &culiopD) {
    
    /***
    *
    * if the left-most matrix of the right hand-side part of the equation is supposed to be inverted before
    * multiplication.
    *
    * */
    if (culiopD.cuLiNA_op_m1 == CULINA_INVERSE_ON) {
        
        culina_tm<double> *workspace = culiopD.workspace;
        
        cublasStatus_t cublas_stat = CUBLAS_STATUS_SUCCESS;
        cusolverStatus_t cusolver_stat1 = CUSOLVER_STATUS_SUCCESS;
        cusolverStatus_t cusolver_stat2 = CUSOLVER_STATUS_SUCCESS;
    
        cusolver_stat1 = cuSOLVER_wrapper::cusolver_wrapper::_cusolver_Dqr_factorization(*cu_matrix1,
                                                                                         *workspace,
                                                                                         culiopD.d_TAU,
                                                                                         culiopD.dev_info,
                                                                                         culiopD.strm);
        
        cusolver_wrapper::_cusolverCheckErrors(cusolver_stat1, __FILE__, __FUNCTION__);
        assert(cusolver_stat1 == CUSOLVER_STATUS_SUCCESS);
    
        //cu_matrix1->_printMatrix();
        
        cusolver_stat2 = cuSOLVER_wrapper::cusolver_wrapper::_cusolver_Doperation_multiplication_qr(*cu_matrix1,
                                                                                                    *cu_matrix2,
                                                                                                    *workspace,
                                                                                                    culiopD.d_TAU,
                                                                                                    culiopD.dev_info,
                                                                                                    CUBLAS_OP_T,
                                                                                                    CUBLAS_SIDE_LEFT,
                                                                                                    culiopD.strm);
    
//        cudaDeviceSynchronize();
//
//        cusolverDnDorgqr(cuSOLVER_wrapper::cusolver_wrapper::_getCusolverDn_handle(),
//                         cu_matrix1->_getRows(),
//                         cu_matrix1->_getColumns(),
//                         220,
//                         cu_matrix1->_getRawData(),
//                         cu_matrix1->_getLeading_dimension(),
//                         culiopD.d_TAU,
//                         workspace->_getRawData(),
//                         workspace->_getLeading_dimension(),
//                         culiopD.dev_info);
//
//        cu_matrix1->_printMatrix();
        
//        std::cout << sizeof(culiopD.d_TAU) << std::endl;
        
        cusolver_wrapper::_cusolverCheckErrors(cusolver_stat2, __FILE__, __FUNCTION__);
        if(cusolver_stat2 != CUSOLVER_STATUS_SUCCESS){
    
            int devInfo;
            cudaMemcpy(&devInfo, culiopD.dev_info, sizeof(int), cudaMemcpyDeviceToHost);
            std::cerr << "devInfo " << devInfo << std::endl;
            
            std::cout << "cu_matrix1" << std::endl << std::endl;
            cu_matrix1->_printMatrix(false, true);
            std::cout << std::endl;

            std::cout << "cu_matrix2" << std::endl << std::endl;
            cu_matrix2->_printMatrix(false, true);
            std::cout << std::endl;
            
            std::cout << "workspace" << std::endl << std::endl;
            workspace->_printMatrix(false, true);
            std::cout << std::endl;
            
        }
        assert(cusolver_stat2 == CUSOLVER_STATUS_SUCCESS);
        
        cublas_stat = cuBLAS_wrapper::cublas_wrapper::_cublas_Dtriangular_system_solver(*cu_matrix1,
                                                                                        *cu_matrix2,
                                                                                        culiopD.alpha,
                                                                                        CUBLAS_SIDE_LEFT,
                                                                                        CUBLAS_FILL_MODE_UPPER,
                                                                                        CUBLAS_OP_N,
                                                                                        CUBLAS_DIAG_NON_UNIT,
                                                                                        culiopD.strm);
    
        cublas_wrapper::_cublasCheckErrors(cublas_stat, __FILE__, __FUNCTION__, 0);
        assert(cublas_stat == CUBLAS_STATUS_SUCCESS);
        
    }
        
        /**
         *
         * if the left-most matrix of right handside equation does not need to be inverted.
         *
         * */
    
    else {
    
        /**
         *
         *
         *
         * */
        if (cu_matrix2->    _getMatrix_type() == cuLiNA::matrix_advanced_initialization_t::DIAGONAL) {
        
            cuLiNA::cuLiNA_error_t culina_stat;
            
            culina_stat = cuLiNA::culina_Ddiagonal_multiplication(cu_matrix1->_getRawData(),
                                                                  culiopD.op_m1,
                                                                  culiopD.alpha,
                                                                  cu_matrix1->_getRows(),
                                                                  cu_matrix1->_getColumns(),
                                                                  cu_matrix1->_getLeading_dimension(),
                                                                  cu_matrix2->_getRawData(),
                                                                  cu_matrix2->_getRows(),
                                                                  cu_matrix2->_getColumns(),
                                                                  cu_matrix2->_getLeading_dimension(),
                                                                  cu_matrix3->_getRawData(),
                                                                  culiopD.beta,
                                                                  cu_matrix3->_getRows(),
                                                                  cu_matrix3->_getColumns(),
                                                                  cu_matrix3->_getLeading_dimension(),
                                                                  culiopD.strm);
            
            cuLiNACheckErrors(culina_stat, __FILE__, __FUNCTION__, __LINE__);
            assert(culina_stat == cuLiNA::cuLiNA_error_t::CULINA_SUCCESS);
            
            
        } else {
    
            cublasStatus_t cublas_stat;
    
            cublas_stat = cuBLAS_wrapper::cublas_wrapper::_cublas_Dmultiplication(*cu_matrix1,
                                                                                  *cu_matrix2,
                                                                                  *cu_matrix3,
                                                                                  culiopD.op_m1,
                                                                                  culiopD.op_m2,
                                                                                  culiopD.alpha,
                                                                                  culiopD.beta,
                                                                                  culiopD.strm);
    
            cublas_wrapper::_cublasCheckErrors(cublas_stat, __FILE__, __FUNCTION__, 0);
            assert(cublas_stat == CUBLAS_STATUS_SUCCESS);
            if(cublas_stat == CUBLAS_STATUS_ALLOC_FAILED)
                return CULINA_MATRIX_NOT_INSTANTIATED;
            if(cublas_stat == CUBLAS_STATUS_INVALID_VALUE)
                return CULINA_MATRIX_NOT_INSTANTIATED;
        
        }
        
    }
    
    return CULINA_SUCCESS;
    
}

cuLiNA::cuLiNA_error_t cuLiNA::culina_Dnorm(culina_tm<double> *cu_matrix1,
                                            double *result,
                                            cuLiNA::culiopD_t &culiopD) {
    
    cublasStatus_t cublas_stat;
    
    cublas_stat = cuBLAS_wrapper::cublas_wrapper::_cublas_Dnorm(*cu_matrix1, result, culiopD.strm);
    
    cublas_wrapper::_cublasCheckErrors(cublas_stat, __FILE__, __FUNCTION__, 0);
    assert(cublas_stat == CUBLAS_STATUS_SUCCESS);
    
    return CULINA_SUCCESS;
    
}

cuLiNA::cuLiNA_error_t cuLiNA::culina_matrix_Dsum(culina_tm<double> *cu_matrix1,
                                                  culina_tm<double> *cu_matrix2,
                                                  culina_tm<double> *cu_matrix3,
                                                  cuLiNA::culiopD_t &culiopD) {
    
    cublasStatus_t stat;
    
    stat = cuBLAS_wrapper::cublas_wrapper::_cublas_Dsum(*cu_matrix1,
                                                        *cu_matrix2,
                                                        *cu_matrix3,
                                                        culiopD.op_m1,
                                                        culiopD.op_m2,
                                                        culiopD.alpha,
                                                        culiopD.beta,
                                                        culiopD.strm);
    
    cuBLAS_wrapper::cublas_wrapper::_cublasCheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    assert(stat == cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    
    return CULINA_SUCCESS;
    
}

/***
     *
     * This function is a little bit complex, observe that here we are dealing with the following system
     *
     * ((J^T)*W*J)*DELTA_x = -(J^T)*W*data_x
     *
     * an so we can divide the operations as
     *
     * r1 = (J^T)*W
     *
     * r2 = -r1*data_x
     *
     * r3 = r1*J
     *
     * r4 = inv(r3)*r2
     *
     * DELTA_x = r4
     *
     * Where r2 and r3 can run concurrently, which means 3 threads are needed;
     *
     * culiopD_1 serves for r1 and r4, culiopD_2 serves for r2, culiopD_3 serves for r3
     *
     * buffers in each culiopD are used only to store intermediary results, final result is stored
     * directly in delta.
     *
     * Watch out for dimensions of matrices
     *
     * J is nxm, DELTA_x is mx1, data is nx1, and weight is nxn. Therefore r1 should be at least mxn, r2 should be mx1
     * and r3 should be mxm
     *
     *
     * */
cuLiNA::cuLiNA_error_t cuLiNA::culina_Dsolve_gradient_descent_first_order(culina_tm<double> *jacobian,
                                                                          culina_tm<double> *delta,
                                                                          culina_tm<double> *data,
                                                                          culina_tm<double> *weight,
                                                                          cuLiNA::culiopD_t &culiopD_1,
                                                                          cuLiNA::culiopD_t &culiopD_2,
                                                                          cuLiNA::culiopD_t &culiopD_3) {
    
    
    if(jacobian == NULL)
        return cuLiNA_error_t::CULINA_MATRIX_NOT_INSTANTIATED;
    
    if(delta == NULL)
        return cuLiNA_error_t::CULINA_MATRIX_NOT_INSTANTIATED;
    
    if(data == NULL)
        return cuLiNA_error_t::CULINA_MATRIX_NOT_INSTANTIATED;
    
    if(weight == NULL)
        return cuLiNA_error_t::CULINA_MATRIX_NOT_INSTANTIATED;
    
    if(culiopD_1.workspace == NULL)
        return cuLiNA_error_t::CULINA_MATRIX_NOT_INSTANTIATED;
    
    if(culiopD_2.workspace == NULL)
        return cuLiNA_error_t::CULINA_MATRIX_NOT_INSTANTIATED;
    
    if(culiopD_3.workspace == NULL)
        return cuLiNA_error_t::CULINA_MATRIX_NOT_INSTANTIATED;
    
    cuLiNA_error_t stat;
    
    cudaEvent_t evnt_1, evnt_2;
    
    cudaEventCreate(&evnt_1);
    cudaEventCreate(&evnt_2);
    
    //r1 = (J^T)*W
    culiopD_1.op_m1 = cublasOperation_t::CUBLAS_OP_T;
    int old_num_ele_1 = culiopD_1.workspace->_getNumber_of_elements();
    
    if(old_num_ele_1 >= jacobian->_getNumber_of_elements()) {
        culiopD_1.workspace->_setRows(jacobian->_getColumns()); ///<-Because it will receive the value of a transpose jacobian
        culiopD_1.workspace->_setColumns(jacobian->_getRows());
    }
    else{
        std::cout << "lascou-se" << std::endl;
        return cuLiNA_error_t::CULINA_PARAMETERS_MISMATCH;
    }
    
//    jacobian->_printMatrix();
//    weight->_printMatrix();
    
    stat = cuLiNA::culina_matrix_Dmultiplication(jacobian, weight, culiopD_1.workspace, culiopD_1);
    cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    if(culiopD_1.strm != NULL)
        cudaEventRecord(evnt_1, *culiopD_1.strm);
    else cudaEventRecord(evnt_1, NULL);
    
    if(culiopD_2.strm != NULL)
        cudaStreamWaitEvent(*culiopD_2.strm, evnt_1, 0);
    if(culiopD_3.strm != NULL)
        cudaStreamWaitEvent(*culiopD_3.strm, evnt_1, 0);
    if(culiopD_2.strm == NULL && culiopD_3.strm == NULL)
        cudaStreamWaitEvent(NULL, evnt_1, 0);
    
//    culiopD_1.workspace->_printMatrix();
    
    culiopD_2.alpha = -1;
    //r2 = -r1*data_x
    //std::cout << "r2 = -r1*data_x" << std::endl;
    stat = cuLiNA::culina_matrix_Dmultiplication(culiopD_1.workspace, data, culiopD_2.workspace, culiopD_2);
    cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    if(culiopD_2.strm != NULL)
        cudaEventRecord(evnt_1, *culiopD_2.strm);
    else cudaEventRecord(evnt_1, NULL);
    
//    data->_printMatrix(true, true);
//    culiopD_2.workspace->_printMatrix();
    
    //r3 = r1*J
    //std::cout << "r3 = r1*J" << std::endl;
    stat = cuLiNA::culina_matrix_Dmultiplication(culiopD_1.workspace, jacobian, culiopD_3.workspace, culiopD_3);
    cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    if(culiopD_3.strm != NULL)
        cudaEventRecord(evnt_2, *culiopD_3.strm);
    else cudaEventRecord(evnt_2, NULL);
    
    if(culiopD_1.strm != NULL) {
        cudaStreamWaitEvent(*culiopD_1.strm, evnt_1, 0);
        cudaStreamWaitEvent(*culiopD_1.strm, evnt_2, 0);
    }
    else{
    
        cudaStreamWaitEvent(NULL, evnt_1, 0);
        cudaStreamWaitEvent(NULL, evnt_2, 0);
        
    }
    
//    culiopD_3.workspace->_printMatrix();
 
    culiopD_1.alpha = 1;
    culiopD_1.op_m1 = cublasOperation_t::CUBLAS_OP_N;
    culiopD_1.cuLiNA_op_m1 = cuLiNA_operation_t::CULINA_INVERSE_ON;
    culiopD_1.workspace->_setRows(old_num_ele_1);
    culiopD_1.workspace->_setColumns(1);
    //DELTA_x = inv(r3)*r2
    stat = cuLiNA::culina_matrix_Dmultiplication(culiopD_3.workspace, culiopD_2.workspace, NULL, culiopD_1);
    cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    cudaEventDestroy(evnt_1);
    cudaEventDestroy(evnt_2);
    
    return stat;
    
}

cuLiNA::cuLiNA_error_t cuLiNA::culina_Dcreate_buffer(culina_tm<double> &target_matrix,
                                                     culina_tm<double> &buffer,
                                                     cuLiNA_buffer_t buffer_t) {

    int lwork;
    
    switch(buffer_t){
    
        case cuLiNA_buffer_t::GEQRF_BUFFER:
            
            cusolverStatus_t stat;
    
            stat = cuSOLVER_wrapper::cusolver_wrapper::_cusolver_Dgeqrf_bufferSize(target_matrix, &lwork, nullptr);
            cuSOLVER_wrapper::cusolver_wrapper::_cusolverCheckErrors(stat, __FILE__, __FUNCTION__);
    
            if(stat != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS)
                return CULINA_CUSOLVER_WRAPPER_PROBLEM;
    
        //std::cout << lwork << std::endl;
            
            buffer._setRows(lwork);
            buffer._setColumns(1);
            buffer._allocateMatrixDataMemory();
            
            break;
            
        default: break;
    }
    
    return CULINA_SUCCESS;

}

cuLiNA::cuLiNA_error_t cuLiNA::culina_Dskew_matrix3x3_operator(culina_tm<double> *vector,
                                                               culina_tm<double> *result_matrix,
                                                               cuLiNA::culiopD_t &culiopD) {
    
    
    
    cuLiNA::cuLiNA_error_t stat = cuLiNA::cuLiNA_error_t::CULINA_MATRIX_NOT_INSTANTIATED;
    
    if(vector != NULL && result_matrix != NULL)
        stat = culina_Dskew_matrix3x3_operator(vector->_getRawData(),
                                               culiopD.alpha,
                                               vector->_getRows(),
                                               vector->_getLeading_dimension(),
                                               result_matrix->_getRawData(),
                                               result_matrix->_getRows(),
                                               result_matrix->_getColumns(),
                                               result_matrix->_getLeading_dimension(),
                                               culiopD.strm);
    
    
    return stat;
}

cuLiNA::cuLiNA_error_t cuLiNA::culina_Dblock_assignment_operation(culina_tm<double> *cu_matrix,
                                                                  culina_tm<double> *cu_matrix_result,
                                                                  int n_row_m_init,
                                                                  int n_column_m_init,
                                                                  int n_row_result_init,
                                                                  int n_column_result_init,
                                                                  int n_rows,
                                                                  int n_columns,
                                                                  culiopD_t &culiopD) {
    
    if(cu_matrix == NULL) return CULINA_MATRIX_NOT_INSTANTIATED;
    if(cu_matrix_result == NULL) return CULINA_MATRIX_NOT_INSTANTIATED;
    
    return cuLiNA::culina_Dblock_assingment(cu_matrix->_getRawData(),
                                            culiopD.op_m1,
                                            culiopD.alpha,
                                            cu_matrix->_getRows(),
                                            cu_matrix->_getColumns(),
                                            n_row_m_init,
                                            n_column_m_init,
                                            cu_matrix->_getLeading_dimension(),
                                            cu_matrix_result->_getRawData(),
                                            cu_matrix_result->_getRows(),
                                            cu_matrix_result->_getColumns(),
                                            n_row_result_init,
                                            n_column_result_init,
                                            cu_matrix_result->_getLeading_dimension(),
                                            n_rows,
                                            n_columns,
                                            culiopD.strm);

}

cuLiNA::cuLiNA_error_t cuLiNA::culina_Ddiagonal_to_vector_operation(culina_tm<double> *cu_matrix,
                                                                    culina_tm<double> *cu_vector_result,
                                                                    culiopD_t &culiopD) {
    
    if(cu_matrix == NULL) return CULINA_MATRIX_NOT_INSTANTIATED;
    if(cu_vector_result == NULL) return CULINA_MATRIX_NOT_INSTANTIATED;
    
    return culina_Ddiagonal_to_vector(cu_matrix->_getRawData(),
                                      culiopD.alpha,
                                      cu_matrix->_getRows(),
                                      cu_matrix->_getColumns(),
                                      cu_matrix->_getLeading_dimension(),
                                      cu_vector_result->_getRawData(),
                                      cu_vector_result->_getRows(),
                                      cu_vector_result->_getLeading_dimension(),
                                      culiopD.strm);

}


cuLiNA::cuLiNA_error_t cuLiNA::culina_Dtrace_operation(culina_tm<double> *cu_matrix,
                                                       culina_tm<double> *cu_auxiliar_vector,
                                                       double &result,
                                                       culiopD_t &culiopD) {
    
    if(cu_matrix == NULL) return CULINA_MATRIX_NOT_INSTANTIATED;
    if(cu_auxiliar_vector == NULL) return CULINA_MATRIX_NOT_INSTANTIATED;
    
    cuLiNA_error_t stat;
    
    stat = cuLiNA::culina_Ddiagonal_to_vector_operation(cu_matrix, cu_auxiliar_vector, culiopD);
    if(stat != CULINA_SUCCESS)
        return stat;
    
    return cuLiNA::culina_Dreduction(cu_auxiliar_vector->_getRawData(), cu_auxiliar_vector->_getNumber_of_elements(), result, culiopD.strm);

}

