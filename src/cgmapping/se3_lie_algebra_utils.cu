//
// Created by spades on 21/05/17.
//

#include "cgmapping/se3_lie_algebra_utils.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

cuLiNA_error_t cgmapping::cuda::exponential_Dmap_se3(culina_vector3d &d_linear_velocity,
                                                     culina_vector3d &d_angular_velocity,
                                                     culina_matrix4d &d_homogenic_transformation,
                                                     culiopD_t &culiopD_1,
                                                     culiopD_t &culiopD_2,
                                                     double time_elapsed) {
    
    if(culiopD_1.workspace==NULL) return cuLiNA::CULINA_MATRIX_NOT_INSTANTIATED;
    if(culiopD_2.workspace==NULL) return cuLiNA::CULINA_MATRIX_NOT_INSTANTIATED;
    
    if(!culiopD_1.workspace->_isSquare()) return cuLiNA::CULINA_MATRIX_NOT_INSTANTIATED;
    if(!culiopD_2.workspace->_isSquare()) return cuLiNA::CULINA_MATRIX_NOT_INSTANTIATED;
    
    if(culiopD_1.workspace->_getRows() != 3) return cuLiNA::CULINA_MATRIX_NOT_INSTANTIATED;
    if(culiopD_2.workspace->_getRows() != 3) return cuLiNA::CULINA_MATRIX_NOT_INSTANTIATED;
    
    auto angular_velocity_norm = time_elapsed;
    
    //culina_matrix<double, 1, 1> tmp_angular_velocity_norm;
    
    cuLiNA_error_t stat = cuLiNA::culina_Dnorm(&d_angular_velocity,
                                               &angular_velocity_norm,
                                               culiopD_1);
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
//    angular_velocity_norm = tmp_angular_velocity_norm(0,0);
    
//    std::cout << "angular_velocity_norm = " << angular_velocity_norm << std::endl;
    
    if (angular_velocity_norm > 0) {
    
        angular_velocity_norm *= time_elapsed;
        
        auto angular_velocity_norm_sqrd = angular_velocity_norm * angular_velocity_norm;
        auto sin_ang_vel_norm = sin(angular_velocity_norm);
        auto sin_over_norm = sin_ang_vel_norm / angular_velocity_norm;
    
        auto cos_ang_vel_norm = cos(angular_velocity_norm);
        auto one_minus_cos_over_norm_sqrd = (1 - cos_ang_vel_norm) / (angular_velocity_norm_sqrd);
    
        auto A = sin_over_norm;
        auto B = one_minus_cos_over_norm_sqrd;
        auto C = (1 - sin_over_norm)/(angular_velocity_norm_sqrd);
        
        if(angular_velocity_norm <= NEAR_ZERO_VALUE){
    
            auto theta = angular_velocity_norm;
            auto theta_sqrd = angular_velocity_norm_sqrd;
            
            A = 1 - ((theta_sqrd/6)*(1 - (theta_sqrd/20)*(1 - (theta_sqrd/42))));
            B = (0.5)*(1 - ((theta_sqrd/12)*(1 - (theta_sqrd/30)*(1 - (theta_sqrd/56)))));
            C = (1./24.)*(1 - ((theta_sqrd/30)*(1 - (theta_sqrd/56)*(1 - (theta_sqrd/90)))));
            
        }
        
        culina_matrix3d skew_symmetric_matrix;
        
        culiopD_1.alpha = time_elapsed;
        
        //this matrix is a skew-matrix of the angular velocity;
        stat = cuLiNA::culina_Dskew_matrix3x3_operator(&d_angular_velocity, &skew_symmetric_matrix, culiopD_1);
        cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
        culiopD_1.workspace->_setIdentity(culiopD_1.strm);
        culiopD_2.workspace->_setIdentity(culiopD_2.strm);
        
        d_homogenic_transformation._setIdentity();
        
        //workspace1 = I + A*skew_symmetric_matrix*I
        culiopD_1.alpha = A;
        culiopD_1.beta = 1;
        stat = cuLiNA::culina_matrix_Dmultiplication(&skew_symmetric_matrix, culiopD_1.workspace, culiopD_1.workspace, culiopD_1);
        cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
        //workspace2 = I + B*skew_symmetric_matrix*I
        culiopD_2.alpha = B;
        culiopD_2.beta = 1;
        stat = cuLiNA::culina_matrix_Dmultiplication(&skew_symmetric_matrix, culiopD_2.workspace, culiopD_2.workspace, culiopD_2);
        cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
        
        //workspace1 += B*skew_symmetric_matrix*skew_symmetric_matrix
        culiopD_1.alpha = B;
        culiopD_1.beta = 1;
        stat = cuLiNA::culina_matrix_Dmultiplication(&skew_symmetric_matrix, &skew_symmetric_matrix, culiopD_1.workspace, culiopD_1);
        cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
        //workspace2 += C*skew_symmetric_matrix*skew_symmetric_matrix
        culiopD_2.alpha = C;
        culiopD_2.beta = 1;
        stat = cuLiNA::culina_matrix_Dmultiplication(&skew_symmetric_matrix, &skew_symmetric_matrix, culiopD_2.workspace, culiopD_2);
        cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
        
        stat = cuLiNA::culina_Dblock_assignment_operation(culiopD_1.workspace, &d_homogenic_transformation, 0, 0, 0, 0, 3, 3, culiopD_1);
        cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
        
        //morph skew_symmetric_matrix to a column vector and perform
        //skew_symmetric_matrix = workspace2*linear_vel
        skew_symmetric_matrix._setRows(3);
        skew_symmetric_matrix._setColumns(1);
        culiopD_2.alpha = time_elapsed;
        culiopD_2.beta = 0;
        stat = cuLiNA::culina_matrix_Dmultiplication(culiopD_2.workspace, &d_linear_velocity, &skew_symmetric_matrix, culiopD_2);
        cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
        stat = cuLiNA::culina_Dblock_assignment_operation(&skew_symmetric_matrix, &d_homogenic_transformation, 0, 0, 0, 3, 3, 1, culiopD_2);
        cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
        skew_symmetric_matrix._setColumns(3);
        
    } else {
    
//        std::cout << "what the fuckinside" << std::endl;
        
//        culiopD_1.workspace->_setIdentity(culiopD_1.strm);
        d_homogenic_transformation._setIdentity(culiopD_2.strm);

//        auto cuda_stat = cudaDeviceSynchronize();
//        cudaCheckErrors(cuda_stat, __FILE__, __FUNCTION__, __LINE__);
        
//        std::cout << "what the fuck ////// inside" << std::endl;
    
//        cuda_stat = cudaDeviceSynchronize();
//        cudaCheckErrors(cuda_stat, __FILE__, __FUNCTION__, __LINE__);

//        std::cout << "what the fuck ----- inside" << std::endl;
    
//        auto cuda_stat = cudaDeviceSynchronize();
//        cudaCheckErrors(cuda_stat, __FILE__, __FUNCTION__, __LINE__);
        
//        culina_vector3d tmp;
        
        culiopD_1.workspace->_setColumns(1);
        
        culiopD_2.alpha = time_elapsed;
        culiopD_2.beta = 0;
        stat = cuLiNA::culina_matrix_Dmultiplication(&identity3d, &d_linear_velocity, culiopD_1.workspace, culiopD_2);
        cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
        stat = cuLiNA::culina_Dblock_assignment_operation(culiopD_1.workspace, &d_homogenic_transformation, 0, 0, 0, 3, 3, 1, culiopD_2);
        cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
        culiopD_1.workspace->_setColumns(3);
        
    }
    
    return stat;
    
}

cuLiNA_error_t cgmapping::cuda::logarithmic_Dmap_se3(culina_matrix4d &d_homogenic_transformation,
                                                     culina_vector3d &d_linear_velocity,
                                                     culina_vector3d &d_angular_velocity,
                                                     culiopD_t &culiopD,
                                                     double time_elapsed) {
    
    if(culiopD.workspace==NULL) return cuLiNA::CULINA_MATRIX_NOT_INSTANTIATED;
    
    if(!culiopD.workspace->_isSquare() && culiopD.workspace->_getRows() != 3) return cuLiNA::CULINA_PARAMETERS_MISMATCH;
    
    cuLiNA::cuLiNA_error_t stat;
    
    double R_trace;
    
    culiopD.op_m1 = CUBLAS_OP_N;
    
    stat = cuLiNA::culina_Dblock_assignment_operation(&d_homogenic_transformation, culiopD.workspace, 0, 0, 0, 0, 3, 3, culiopD);
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    stat = cuLiNA::culina_Dblock_assignment_operation(&d_homogenic_transformation, &d_linear_velocity, 0, 3, 0, 0, 3, 1, culiopD);
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    culiopD.alpha  = 1;
    culiopD.beta = 0;
    culiopD.gamma = 0;
    
    stat = cuLiNA::culina_Dtrace_operation(culiopD.workspace, &d_angular_velocity, R_trace, culiopD);
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    double angular_velocity_norm = (R_trace - 1)/2;
    
    angular_velocity_norm = acos(angular_velocity_norm);
    
    double sin_ang_vel_norm = sin(angular_velocity_norm);
    
    if(angular_velocity_norm > 0) {
    
        culiopD.alpha = angular_velocity_norm/(2*sin_ang_vel_norm);
        culiopD.beta = -culiopD.alpha;
        culiopD.gamma = 0;
        culiopD.op_m2 = CUBLAS_OP_T;
        
//        ln(R) <-- (theta/(2*sin(theta))*(R - R^T)
        stat = cuLiNA::culina_matrix_Dsum(culiopD.workspace, culiopD.workspace, culiopD.workspace, culiopD);
        cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
        culiopD.op_m2 = CUBLAS_OP_N;
        
        culiopD.alpha = (1./time_elapsed);
        culiopD.beta = 0;
    
        stat = cuLiNA::culina_Dvector_from_skew_matrix3x3_operator(culiopD.workspace, &d_angular_velocity, culiopD);
        cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
        double A = sin_ang_vel_norm/angular_velocity_norm;
        double B = (1 - cos(angular_velocity_norm))/(angular_velocity_norm*angular_velocity_norm);
    
        if(angular_velocity_norm <= NEAR_ZERO_VALUE){
        
            auto theta = angular_velocity_norm;
            auto theta_sqrd = theta*theta;
    
            culiopD.alpha = (1./12.)*(1 + ((theta_sqrd/60)*(1 + (theta_sqrd/42)*(1 + (theta_sqrd/40)))));
        
        }
        else {
    
            culiopD.alpha = (1 / (angular_velocity_norm * angular_velocity_norm)) * (1 - (A / (2 * B)));
            
        }
        
        culiopD.beta = -0.5;
        culiopD.gamma = 0;
    
        culiopD.op_m1 = CUBLAS_OP_N;
        culiopD.op_m2 = CUBLAS_OP_N;
    
        stat = cuLiNA::culina_matrix_Dmultiplication(culiopD.workspace, culiopD.workspace, culiopD.workspace, culiopD);
        cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    }
    
    else d_angular_velocity._setZero(culiopD.strm);
    
    culiopD.alpha = 1;
    culiopD.beta = 0;
    culiopD.gamma = (angular_velocity_norm?1:0);
    
    stat = cuLiNA::culina_matrix_Dsum(&identity3d, NULL, culiopD.workspace, culiopD);
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    culiopD.alpha = (1/time_elapsed);
    culiopD.beta = 0;
    culiopD.gamma = 0;
    
    stat = cuLiNA::culina_matrix_Dmultiplication(culiopD.workspace, &d_linear_velocity, &d_linear_velocity, culiopD);
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    return stat;
    
}
cuLiNA_error_t cgmapping::cuda::adjoint_Dse3(culina_matrix4d &d_homogenic_transformation,
                                             culina_matrix<double, 6, 6> &d_adjoint_matrix,
                                             culiopD_t &culiopD) {
    
    if(culiopD.workspace == NULL) return CULINA_MATRIX_NOT_INSTANTIATED;
    
    if(!culiopD.workspace->_isSquare()) return CULINA_PARAMETERS_MISMATCH;
    
    if(culiopD.workspace->_getRows() != 3) return CULINA_PARAMETERS_MISMATCH;
    
    culiopD.alpha = 1;
    culiopD.beta = culiopD.gamma = 0;
    culiopD.op_m1 = culiopD.op_m2 = CUBLAS_OP_N;
    culiopD.cuLiNA_op_m1 = culiopD.cuLiNA_op_m2 = CULINA_INVERSE_OFF;
    
    auto stat = CULINA_SUCCESS;
    
    d_adjoint_matrix._setRows(3);
    d_adjoint_matrix._setColumns(1);
    
    
    // t <-- T(0:2, 3)
    stat = culina_Dblock_assignment_operation(&d_homogenic_transformation, &d_adjoint_matrix, 0,3,0,0,3,1, culiopD);
    cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    //t_x <-- S(t)
    stat = culina_Dskew_matrix3x3_operator(&d_adjoint_matrix, culiopD.workspace, culiopD);
    
    d_adjoint_matrix._setRows(3);
    d_adjoint_matrix._setColumns(3);
    
    //adjoint(0:2,0:2) <-- T(R)
    stat = culina_Dblock_assignment_operation(&d_homogenic_transformation, &d_adjoint_matrix, 0,0,0,0,3,3, culiopD);
    cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    //t_x*R
    stat = culina_matrix_Dmultiplication(culiopD.workspace, &d_adjoint_matrix, culiopD.workspace, culiopD);
    cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    d_adjoint_matrix._setRows(6);
    d_adjoint_matrix._setColumns(6);
    
    //adjoint <-- zeros(6,6)
    d_adjoint_matrix._setZero(culiopD.strm);
    
    //adjoint(0:2,0:2) <-- T(R)
    stat = culina_Dblock_assignment_operation(&d_homogenic_transformation, &d_adjoint_matrix, 0,0,0,0,3,3, culiopD);
    cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    //adjoint(3:5,3:5) <-- T(R)
    stat = culina_Dblock_assignment_operation(&d_homogenic_transformation, &d_adjoint_matrix, 0,0,3,3,3,3, culiopD);
    cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    //adjoint(0:2,3:5) <-- t_xR)
    stat = culina_Dblock_assignment_operation(culiopD.workspace, &d_adjoint_matrix, 0,0,0,3,3,3, culiopD);
    cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    return stat;
}
