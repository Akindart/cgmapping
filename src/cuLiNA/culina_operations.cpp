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
    
    if(cu_matrix1 != NULL && cu_matrix2 != NULL && cu_matrix3 != NULL) {
    
        if((culiopD.op_m1==CUBLAS_OP_T?cu_matrix1->_getRows():cu_matrix1->_getColumns()) !=
            (culiopD.op_m2==CUBLAS_OP_T?cu_matrix2->_getColumns():cu_matrix2->_getRows()))
            return CULINA_PARAMETERS_MISMATCH;
    
        if((culiopD.op_m1==CUBLAS_OP_T?cu_matrix1->_getColumns():cu_matrix1->_getRows()) != cu_matrix3->_getRows())
            return CULINA_PARAMETERS_MISMATCH;
    
        if((culiopD.op_m2==CUBLAS_OP_T?cu_matrix2->_getRows():cu_matrix2->_getColumns()) != cu_matrix3->_getColumns())
            return CULINA_PARAMETERS_MISMATCH;
        
        if (cu_matrix1->_getMatrix_type() != cuLiNA::matrix_advanced_initialization_t::DIAGONAL &&
            cu_matrix2->_getMatrix_type() == cuLiNA::matrix_advanced_initialization_t::DIAGONAL) {
    
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
            //assert(culina_stat == cuLiNA::cuLiNA_error_t::CULINA_SUCCESS);
    
        }  else {
            
            cublasStatus_t cublas_stat;
        
            cublas_stat = cuBLAS_wrapper::cublas_wrapper::_cublas_Dmultiplication(*cu_matrix1,
                                                                                  *cu_matrix2,
                                                                                  *cu_matrix3,
                                                                                  culiopD.op_m1,
                                                                                  culiopD.op_m2,
                                                                                  culiopD.alpha,
                                                                                  culiopD.beta,
                                                                                  culiopD.strm);
        
            cublas_wrapper::_cublasCheckErrors(cublas_stat, __FILE__, __FUNCTION__, __LINE__);
            //assert(cublas_stat == CUBLAS_STATUS_SUCCESS);
            if (cublas_stat == CUBLAS_STATUS_ALLOC_FAILED)
                return CULINA_MATRIX_NOT_INSTANTIATED;
            if (cublas_stat == CUBLAS_STATUS_INVALID_VALUE)
                return CULINA_MATRIX_NOT_INSTANTIATED;
            if (cublas_stat != CUBLAS_STATUS_SUCCESS)
                return CULINA_INVALID_PARAMETER;
        }
    
    }
    
    else return CULINA_MATRIX_NOT_INSTANTIATED;
    
    
    return CULINA_SUCCESS;
    
}

cuLiNA::cuLiNA_error_t cuLiNA::culina_Dnorm(culina_tm<double> *cu_matrix1,
                                            double *result,
                                            cuLiNA::culiopD_t &culiopD) {
    
    cublasStatus_t cublas_stat;
    
    if(cu_matrix1->_getColumns() != 1) return CULINA_PARAMETERS_MISMATCH;
    
    cublas_stat = cuBLAS_wrapper::cublas_wrapper::_cublas_Dnorm(*cu_matrix1, result, culiopD.strm);
    cublas_wrapper::_cublasCheckErrors(cublas_stat, __FILE__, __FUNCTION__, __LINE__);
    
    assert(cublas_stat == CUBLAS_STATUS_SUCCESS);
   
    return CULINA_SUCCESS;
    
}

cuLiNA::cuLiNA_error_t cuLiNA::culina_matrix_Dsum(culina_tm<double> *cu_matrix1,
                                                  culina_tm<double> *cu_matrix2,
                                                  culina_tm<double> *cu_matrix3,
                                                  cuLiNA::culiopD_t &culiopD) {
    
    auto stat = CULINA_SUCCESS;
    
    if(cu_matrix3 != NULL) {
    
//        cublasStatus_t stat;
        
//        stat = cuBLAS_wrapper::cublas_wrapper::_cublas_Dsum(*cu_matrix1,
//                                                            *cu_matrix2,
//                                                            *cu_matrix3,
//                                                            culiopD.op_m1,
//                                                            culiopD.op_m2,
//                                                            culiopD.alpha,
//                                                            culiopD.beta,
//                                                            culiopD.strm);
    
        stat = cuLiNA::culina_Dsumm(cu_matrix1 == NULL ? NULL : cu_matrix1->_getRawData(),
                                    (culiopD.op_m1 == CUBLAS_OP_T),
                                    culiopD.alpha,
                                    (cu_matrix1 == NULL ? cu_matrix3 : cu_matrix1)->_getRows(),
                                    (cu_matrix1 == NULL ? cu_matrix3 : cu_matrix1)->_getColumns(),
                                    (cu_matrix1 == NULL ? cu_matrix3 : cu_matrix1)->_getLeading_dimension(),
                                    cu_matrix2 == NULL ? NULL : cu_matrix2->_getRawData(),
                                    (culiopD.op_m2 == CUBLAS_OP_T),
                                    culiopD.beta,
                                    (cu_matrix2 == NULL ? cu_matrix3 : cu_matrix2)->_getRows(),
                                    (cu_matrix2 == NULL ? cu_matrix3 : cu_matrix2)->_getColumns(),
                                    (cu_matrix2 == NULL ? cu_matrix3 : cu_matrix2)->_getLeading_dimension(),
                                    cu_matrix3->_getRawData(),
                                    culiopD.gamma,
                                    cu_matrix3->_getRows(),
                                    cu_matrix3->_getColumns(),
                                    cu_matrix3->_getLeading_dimension(),
                                    culiopD.strm);
        
    
//        cuBLAS_wrapper::cublas_wrapper::_cublasCheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
//        assert(stat == cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        
        cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
        
    }
    else stat = CULINA_MATRIX_NOT_INSTANTIATED;
    
    return stat;
    
}

cuLiNA::cuLiNA_error_t cuLiNA::culina_Dinverse_matrix(culina_tm<double> *cu_matrix1,
                                                      culina_tm<double> *cu_matrix2,
                                                      culiopD_t &culiopD) {
    
    if (cu_matrix1 != NULL && cu_matrix2 != NULL) {
        
        if (culiopD.dev_info == NULL) return CULINA_INVALID_PARAMETER;
        
        auto stat = (cu_matrix1->_isSquare()) ?
                    cuBLAS_wrapper::cublas_wrapper::_cublas_Dinverse(*cu_matrix1,
                                                                     *cu_matrix2,
                                                                     culiopD.dev_info,
                                                                     culiopD.strm) :
                    CUBLAS_STATUS_INVALID_VALUE;
        
        cuBLAS_wrapper::cublas_wrapper::_cublasCheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
        //std::cout << "and that's us again friend" << std::endl;
        
    } else return CULINA_MATRIX_NOT_INSTANTIATED;
    
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
                                                                          double k_lambda_scalar,
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
    
    cuLiNA_error_t stat_culina;
    cudaError_t stat_cuda;
    
    cudaEvent_t evnt_1, evnt_2;

    cudaCheckErrors(cudaEventCreate(&evnt_1), __FILE__, __FUNCTION__, __LINE__);
    
    cudaCheckErrors(cudaEventCreate(&evnt_2), __FILE__, __FUNCTION__, __LINE__);
    
    auto old_num_ele_1 = culiopD_1.workspace->_getNumber_of_elements();
    auto old_num_rows_1 = culiopD_1.workspace->_getRows();
    auto old_num_cols_1 = culiopD_1.workspace->_getColumns();
    
    if(old_num_ele_1 >= jacobian->_getNumber_of_elements()) {
        culiopD_1.workspace->_setRows(jacobian->_getColumns()); ///<-Because it will receive the value of a transpose jacobian
        culiopD_1.workspace->_setColumns(jacobian->_getRows());
    }
    else{
        std::cout << "lascou-se" << std::endl;
        return cuLiNA_error_t::CULINA_PARAMETERS_MISMATCH;
    }


//    r1 = (J^T)*W
    
    culiopD_1.op_m1 = cublasOperation_t::CUBLAS_OP_T;
    culiopD_1.alpha = 1;
    culiopD_1.beta = 0;
    culiopD_1.gamma = 0;
    
    stat_culina = cuLiNA::culina_matrix_Dmultiplication(jacobian, weight, culiopD_1.workspace, culiopD_1);
    cuLiNACheckErrors(stat_culina, __FILE__, __FUNCTION__, __LINE__);
    
    stat_cuda = cudaEventRecord(evnt_1, (culiopD_1.strm!=NULL)?*culiopD_1.strm:0);
    cudaCheckErrors(stat_cuda, __FILE__, __FUNCTION__, __LINE__);
    
    stat_cuda = cudaStreamWaitEvent((culiopD_2.strm!=NULL)?*culiopD_2.strm:0, evnt_1, 0);
    cudaCheckErrors(stat_cuda, __FILE__, __FUNCTION__, __LINE__);
    
    stat_cuda = cudaStreamWaitEvent((culiopD_3.strm!=NULL)?*culiopD_3.strm:0, evnt_1, 0);
    cudaCheckErrors(stat_cuda, __FILE__, __FUNCTION__, __LINE__);
    
//    r2 = -r1*data_x
//    std::cout << "r2 = -r1*data_x" << std::endl;
    culiopD_2.alpha = -1;
    culiopD_2.beta = 0;
    culiopD_2.gamma = 0;
    culiopD_2.op_m1 = culiopD_2.op_m2  = CUBLAS_OP_N;
    
//    culiopD_1.workspace->_printMatrix(false, true);
//    data->_printMatrix(false, true);
//    culiopD_2.workspace->_printMatrix(false, true);
    
    stat_culina = cuLiNA::culina_matrix_Dmultiplication(culiopD_1.workspace, data, culiopD_2.workspace, culiopD_2);
    cuLiNACheckErrors(stat_culina, __FILE__, __FUNCTION__, __LINE__);
    
    //    r3 = r1*J
//    std::cout << "r3 = r1*J" << std::endl;
    
    culiopD_3.workspace->_setIdentity(culiopD_3.strm);
    culiopD_3.alpha = 1;
    culiopD_3.beta = ((k_lambda_scalar>0)?k_lambda_scalar:0);
    culiopD_3.gamma = 0;
    culiopD_3.op_m1 = culiopD_3.op_m2  = CUBLAS_OP_N;
    
    stat_culina = cuLiNA::culina_matrix_Dmultiplication(culiopD_1.workspace, jacobian, culiopD_3.workspace, culiopD_3);
    cuLiNACheckErrors(stat_culina, __FILE__, __FUNCTION__, __LINE__);
    
    cudaDeviceSynchronize();

    std::string r1("/home/spades/kinetic_ws/src/cgmapping/r1_matrix.m");
    std::string r2("/home/spades/kinetic_ws/src/cgmapping/r2_matrix.m");
    std::string r3("/home/spades/kinetic_ws/src/cgmapping/r3_matrix.m");

//    cuLiNA::culina_download_matrix_file(*culiopD_1.workspace, r1);
//    cuLiNA::culina_download_matrix_file(*culiopD_2.workspace, r2);
//    cuLiNA::culina_download_matrix_file(*culiopD_3.workspace, r3);
    
    cudaMalloc(&culiopD_3.dev_info, sizeof(int));
    
    stat_culina = cuLiNA::culina_Dinverse_matrix(culiopD_3.workspace, culiopD_3.workspace, culiopD_3);
    cuLiNACheckErrors(stat_culina, __FILE__, __FUNCTION__, __LINE__);
    
    stat_cuda = cudaEventRecord(evnt_1, (culiopD_2.strm!=NULL)?*culiopD_2.strm:0);
    cudaCheckErrors(stat_cuda, __FILE__, __FUNCTION__, __LINE__);
    
    stat_cuda = cudaEventRecord(evnt_2, (culiopD_3.strm!=NULL)?*culiopD_3.strm:0);
    cudaCheckErrors(stat_cuda, __FILE__, __FUNCTION__, __LINE__);
    
    stat_cuda = cudaStreamWaitEvent((culiopD_1.strm!=NULL)?*culiopD_1.strm:0, evnt_1, 0);
    cudaCheckErrors(stat_cuda, __FILE__, __FUNCTION__, __LINE__);
    
    stat_cuda = cudaStreamWaitEvent((culiopD_1.strm!=NULL)?*culiopD_1.strm:0, evnt_2, 0);
    cudaCheckErrors(stat_cuda, __FILE__, __FUNCTION__, __LINE__);
    
//    DELTA_x = inv(r3)*r2
    culiopD_1.alpha = 1;
    culiopD_1.beta = 0;
    culiopD_1.gamma = 0;
    culiopD_1.op_m1 = cublasOperation_t::CUBLAS_OP_N;
    culiopD_1.workspace->_setRows(old_num_ele_1);
    culiopD_1.workspace->_setColumns(1);
    
    stat_culina = cuLiNA::culina_matrix_Dmultiplication(culiopD_3.workspace, culiopD_2.workspace, delta, culiopD_1);
    cuLiNACheckErrors(stat_culina, __FILE__, __FUNCTION__, __LINE__);
    
    cudaStreamSynchronize(*(culiopD_1.strm));
    
//    delta->_printMatrix();
    
    cudaEventDestroy(evnt_1);
    cudaEventDestroy(evnt_2);
    
    cudaFree(culiopD_3.dev_info);
    
    culiopD_1.workspace->_setRows(old_num_rows_1);
    culiopD_1.workspace->_setColumns(old_num_cols_1);
    
    return stat_culina;
    
}

cuLiNA::cuLiNA_error_t cuLiNA::culina_Dsolve_gradient_descent_first_order(culina_tm<double> *jacobian,
                                                                          culina_tm<double> *delta,
                                                                          culina_tm<double> *data,
                                                                          culina_tm<double> *weight,
                                                                          culina_tm<double> *motion_prior,
                                                                          culina_tm<double> *cov_motion_prior,
                                                                          culina_tm<double> *estimation_k,
                                                                          double k_lambda_scalar,
                                                                          culiopD_t &culiopD_1,
                                                                          culiopD_t &culiopD_2,
                                                                          culiopD_t &culiopD_3) {
    
    if (jacobian == NULL)
        return cuLiNA_error_t::CULINA_MATRIX_NOT_INSTANTIATED;
    
    if (delta == NULL)
        return cuLiNA_error_t::CULINA_MATRIX_NOT_INSTANTIATED;
    
    if (data == NULL)
        return cuLiNA_error_t::CULINA_MATRIX_NOT_INSTANTIATED;
    
    if (weight == NULL)
        return cuLiNA_error_t::CULINA_MATRIX_NOT_INSTANTIATED;
    
    if (weight == NULL)
        return cuLiNA_error_t::CULINA_MATRIX_NOT_INSTANTIATED;
    
    if (motion_prior == NULL)
        return cuLiNA_error_t::CULINA_MATRIX_NOT_INSTANTIATED;
    
    if (cov_motion_prior == NULL)
        return cuLiNA_error_t::CULINA_MATRIX_NOT_INSTANTIATED;
    
    if (estimation_k == NULL)
        return cuLiNA_error_t::CULINA_MATRIX_NOT_INSTANTIATED;
    
    if (culiopD_1.workspace == NULL)
        return cuLiNA_error_t::CULINA_MATRIX_NOT_INSTANTIATED;
    
    if (culiopD_2.workspace == NULL)
        return cuLiNA_error_t::CULINA_MATRIX_NOT_INSTANTIATED;
    
    if (culiopD_3.workspace == NULL)
        return cuLiNA_error_t::CULINA_MATRIX_NOT_INSTANTIATED;
    
    cuLiNA_error_t stat_culina;
    cudaError_t stat_cuda;
    
    cudaEvent_t evnt_1, evnt_2;
    
    cudaCheckErrors(cudaEventCreate(&evnt_1), __FILE__, __FUNCTION__, __LINE__);
    
    cudaCheckErrors(cudaEventCreate(&evnt_2), __FILE__, __FUNCTION__, __LINE__);
    
    auto old_num_ele_1 = culiopD_1.workspace->_getNumber_of_elements();
    auto old_num_rows_1 = culiopD_1.workspace->_getRows();
    auto old_num_cols_1 = culiopD_1.workspace->_getColumns();
    
    if (old_num_ele_1 >= jacobian->_getNumber_of_elements()) {
        culiopD_1.workspace
            ->_setRows(jacobian->_getColumns()); ///<-Because it will receive the value of a transpose jacobian
        culiopD_1.workspace->_setColumns(jacobian->_getRows());
    } else {
        std::cout << "lascou-se" << std::endl;
        return cuLiNA_error_t::CULINA_PARAMETERS_MISMATCH;
    }
    
//    r1 = (J^T)*W
    culiopD_1.op_m1 = cublasOperation_t::CUBLAS_OP_T;
    culiopD_1.alpha = 1;
    culiopD_1.beta = 0;
    culiopD_1.gamma = 0;
    
    stat_culina = cuLiNA::culina_matrix_Dmultiplication(jacobian, weight, culiopD_1.workspace, culiopD_1);
    cuLiNACheckErrors(stat_culina, __FILE__, __FUNCTION__, __LINE__);
    
//    r2 = scale_factor*(prior - est_k)
    culiopD_2.alpha = k_lambda_scalar;
    culiopD_2.beta = -k_lambda_scalar;
    culiopD_2.gamma = 0;
    culiopD_2.op_m1 = culiopD_2.op_m2 = CUBLAS_OP_N;
    
    stat_culina = cuLiNA::culina_matrix_Dsum(motion_prior, estimation_k, culiopD_2.workspace, culiopD_2);
    cuLiNACheckErrors(stat_culina, __FILE__, __FUNCTION__, __LINE__);
    
//    r3 <-- cov_motion_prior
    cudaCheckErrors(culiopD_3.workspace->_loadData(*cov_motion_prior, *culiopD_3.strm), __FILE__, __FUNCTION__, __LINE__);
    
//    r3 <-- inv(r3)=info_matrix
    cudaMalloc(&culiopD_3.dev_info, sizeof(int));
    stat_culina = cuLiNA::culina_Dinverse_matrix(culiopD_3.workspace, culiopD_3.workspace, culiopD_3);
    cuLiNACheckErrors(stat_culina, __FILE__, __FUNCTION__, __LINE__);
    
    cudaCheckErrors(cudaEventRecord(evnt_1, (culiopD_1.strm != NULL) ? *culiopD_1.strm : 0),
                    __FILE__, __FUNCTION__, __LINE__);
    
    cudaCheckErrors(cudaStreamWaitEvent((culiopD_2.strm != NULL) ? *culiopD_2.strm : 0, evnt_1, 0),
                    __FILE__, __FUNCTION__, __LINE__);
    
    cudaCheckErrors(cudaStreamWaitEvent((culiopD_3.strm != NULL) ? *culiopD_3.strm : 0, evnt_1, 0),
                    __FILE__, __FUNCTION__, __LINE__);

//    r2 = -r1*data_x + r2
    culiopD_2.alpha = -1;
    culiopD_2.beta = 1;
    culiopD_2.gamma = 0;
    culiopD_2.op_m1 = culiopD_2.op_m2 = CUBLAS_OP_N;
    
    stat_culina = cuLiNA::culina_matrix_Dmultiplication(culiopD_1.workspace, data, culiopD_2.workspace, culiopD_2);
    cuLiNACheckErrors(stat_culina, __FILE__, __FUNCTION__, __LINE__);
    
//    r3 = (r1*J + k_lambda_scalar*cov_prior)
    culiopD_3.alpha = 1;
    culiopD_3.beta = k_lambda_scalar;
    culiopD_3.gamma = 0;
    culiopD_3.op_m1 = culiopD_3.op_m2 = CUBLAS_OP_N;
    
    stat_culina = cuLiNA::culina_matrix_Dmultiplication(culiopD_1.workspace, jacobian, culiopD_3.workspace, culiopD_3);
    cuLiNACheckErrors(stat_culina, __FILE__, __FUNCTION__, __LINE__);
    
//    cudaDeviceSynchronize();
    
//    std::string r1("/home/spades/kinetic_ws/src/cgmapping/r1_matrix.m");
//    std::string r2("/home/spades/kinetic_ws/src/cgmapping/r2_matrix.m");
//    std::string r3("/home/spades/kinetic_ws/src/cgmapping/r3_matrix.m");

//    cuLiNA::culina_download_matrix_file(*culiopD_1.workspace, r1);
//    cuLiNA::culina_download_matrix_file(*culiopD_2.workspace, r2);
//    cuLiNA::culina_download_matrix_file(*culiopD_3.workspace, r3);
    
    stat_culina = cuLiNA::culina_Dinverse_matrix(culiopD_3.workspace, culiopD_3.workspace, culiopD_3);
    cuLiNACheckErrors(stat_culina, __FILE__, __FUNCTION__, __LINE__);
    
    cudaCheckErrors(cudaEventRecord(evnt_1, (culiopD_2.strm != NULL) ? *culiopD_2.strm : 0), __FILE__, __FUNCTION__, __LINE__);
    
    cudaCheckErrors(cudaEventRecord(evnt_2, (culiopD_3.strm != NULL) ? *culiopD_3.strm : 0), __FILE__, __FUNCTION__, __LINE__);
    
    cudaCheckErrors(cudaStreamWaitEvent((culiopD_1.strm != NULL) ? *culiopD_1.strm : 0, evnt_1, 0), __FILE__, __FUNCTION__, __LINE__);
    
    cudaCheckErrors(cudaStreamWaitEvent((culiopD_1.strm != NULL) ? *culiopD_1.strm : 0, evnt_2, 0), __FILE__, __FUNCTION__, __LINE__);

//    DELTA_x = inv(r3)*r2
    culiopD_1.alpha = 1;
    culiopD_1.beta = 0;
    culiopD_1.gamma = 0;
    culiopD_1.op_m1 = cublasOperation_t::CUBLAS_OP_N;
    culiopD_1.workspace->_setRows(old_num_ele_1);
    culiopD_1.workspace->_setColumns(1);
    
    stat_culina = cuLiNA::culina_matrix_Dmultiplication(culiopD_3.workspace, culiopD_2.workspace, delta, culiopD_1);
    cuLiNACheckErrors(stat_culina, __FILE__, __FUNCTION__, __LINE__);
    
//    cudaStreamSynchronize(*(culiopD_1.strm));

//    delta->_printMatrix();
    
    cudaEventDestroy(evnt_1);
    cudaEventDestroy(evnt_2);
    
    cudaFree(culiopD_3.dev_info);
    
    culiopD_1.workspace->_setRows(old_num_rows_1);
    culiopD_1.workspace->_setColumns(old_num_cols_1);
    
    return stat_culina;

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

cuLiNA::cuLiNA_error_t cuLiNA::culina_Dvector_from_skew_matrix3x3_operator(culina_tm<double> *matrix,
                                                                           culina_tm<double> *result_vector,
                                                                           culiopD_t &culiopD) {
    
    cuLiNA::cuLiNA_error_t stat = cuLiNA::cuLiNA_error_t::CULINA_MATRIX_NOT_INSTANTIATED;
    
    if(result_vector != NULL && matrix != NULL)
        stat = culina_Dvector_from_skew_matrix3x3_operator(matrix->_getRawData(),
                                                           culiopD.alpha,
                                                           matrix->_getRows(),
                                                           matrix->_getColumns(),
                                                           matrix->_getLeading_dimension(),
                                                           result_vector->_getRawData(),
                                                           result_vector->_getRows(),
                                                           result_vector->_getLeading_dimension(),
                                                           culiopD.strm);
    return stat;
}


cuLiNA::cuLiNA_error_t cuLiNA::culina_Dblock_assignment_operation(culina_tm<double> *cu_matrix,
                                                                  culina_tm<double> *cu_matrix_result,
                                                                  int n_row_m_init = 0,
                                                                  int n_column_m_init = 0,
                                                                  int n_row_result_init = 0,
                                                                  int n_column_result_init = 0,
                                                                  int n_rows = -1,
                                                                  int n_columns = -1,
                                                                  culiopD_t &culiopD) {
    
    if(cu_matrix == NULL) return CULINA_MATRIX_NOT_INSTANTIATED;
    if(cu_matrix_result == NULL) return CULINA_MATRIX_NOT_INSTANTIATED;
    
    if(n_rows < 0) n_rows = cu_matrix->_getRows();
    if(n_columns < 0) n_columns = cu_matrix->_getColumns();
    
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
    if(!cu_matrix->_isSquare()) return CULINA_PARAMETERS_MISMATCH;
    if(cu_matrix->_getRows() != cu_vector_result->_getRows()) return CULINA_PARAMETERS_MISMATCH;
    
    
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
    if(!cu_matrix->_isSquare()) return CULINA_PARAMETERS_MISMATCH;
    if(cu_matrix->_getRows() != cu_auxiliar_vector->_getRows()) return CULINA_PARAMETERS_MISMATCH;
    
    cuLiNA_error_t stat;
    
    stat = cuLiNA::culina_Ddiagonal_to_vector_operation(cu_matrix, cu_auxiliar_vector, culiopD);
    if(stat != CULINA_SUCCESS)
        return stat;
    
    stat = cuLiNA::culina_Dreduction(cu_auxiliar_vector->_getRawData(), cu_auxiliar_vector->_getNumber_of_elements(), result, culiopD.strm);;
    
    cudaDeviceSynchronize();
    
    return stat;

}
