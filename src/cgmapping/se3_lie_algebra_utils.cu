//
// Created by spades on 21/05/17.
//

#include "cgmapping/se3_lie_algebra_utils.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

Matrix3d cgmapping::exponential_map_so3(Vector3d &angular_velocity) {
    
    Matrix3d rotation_matrix;
    
    rotation_matrix.setIdentity();
    
    if (angular_velocity.norm() > 0) {
        
        double angular_velocity_norm = angular_velocity.norm();
        double sin_ang_vel_norm = sin(angular_velocity_norm);
        double sin_over_norm = sin_ang_vel_norm / angular_velocity_norm;
        
        double cos_ang_vel_norm = cos(angular_velocity_norm);
        double one_minus_cos_over_norm_sqrd = (1 - cos_ang_vel_norm) / (angular_velocity_norm * angular_velocity_norm);
        
        Matrix3d identity, angular_velocity_skew_matrix;
        
        identity.setIdentity();
        
        angular_velocity_skew_matrix << 0, -angular_velocity(2, 0), angular_velocity(1, 0),
            angular_velocity(2, 0), 0, -angular_velocity(0, 0),
            -angular_velocity(1, 0), angular_velocity(0, 0), 0;
        
        rotation_matrix = identity + sin_over_norm * angular_velocity_skew_matrix +
            one_minus_cos_over_norm_sqrd * angular_velocity_skew_matrix * angular_velocity_skew_matrix;
        
    }
    
    return rotation_matrix;
    
}

Matrix3d cgmapping::exponential_map_so3(Vector3d &angular_velocity, double time_elapsed) {
    
    Vector3d angle_elapsed = angular_velocity * time_elapsed;
    
    return cgmapping::exponential_map_so3(angle_elapsed);
    
}

Vector3d cgmapping::logarithmic_map_so3(Matrix3d &rotation_matrix, double time_elapsed) {
    
    double sin_time_elapsed = sin(time_elapsed);
    
    Matrix3d skew_matrix_angular_velocity;
    
    Vector3d angular_velocity;
    
    angular_velocity << rotation_matrix(2, 1) - rotation_matrix(1, 2), rotation_matrix(0, 2) - rotation_matrix(2, 0),
        rotation_matrix(1, 0) - rotation_matrix(0, 1);
    
    angular_velocity *= (1 / (2 * sin_time_elapsed));
    
    return angular_velocity;
    
}

Vector3d cgmapping::logarithmic_map_so3(Matrix3d &rotation_matrix) {
    
    //as long as no time is informed we recur to variable phi
    double phi;
    double traceR_minus_one = rotation_matrix.trace() - 1.0;
    double trR_min_one_over_two = traceR_minus_one / 2.0;
    
    phi = acos(trR_min_one_over_two);
    
    return cgmapping::logarithmic_map_so3(rotation_matrix, phi);
    
}

Matrix4d cgmapping::exponential_map_se3(Vector6d &twist_velocity) {
    
    Matrix4d homogenic_transformation;
    
    homogenic_transformation.setIdentity();
    
    Vector3d angular_velocity;
    Vector3d linear_velocity;
    
    angular_velocity << twist_velocity.tail(3);
    linear_velocity << twist_velocity.head(3);
    
    if (angular_velocity.norm() > 0) {
        
        double angular_velocity_norm = angular_velocity.norm();
        double angular_velocity_norm_sqrt = angular_velocity_norm * angular_velocity_norm;
        double sin_ang_vel_norm = sin(angular_velocity_norm);
        double sin_over_norm = sin_ang_vel_norm / angular_velocity_norm;
        
        double cos_ang_vel_norm = cos(angular_velocity_norm);
        double one_minus_cos_over_norm_sqrd = (1 - cos_ang_vel_norm) / (angular_velocity_norm_sqrt);
        
        Matrix3d identity, angular_velocity_skew_matrix;
        
        identity.setIdentity();
        
        angular_velocity_skew_matrix << 0, -angular_velocity(2, 0), angular_velocity(1, 0),
            angular_velocity(2, 0), 0, -angular_velocity(0, 0),
            -angular_velocity(1, 0), angular_velocity(0, 0), 0;
        
        Matrix3d angular_velocity_skew_matrix_squared;
        
        angular_velocity_skew_matrix_squared = angular_velocity_skew_matrix * angular_velocity_skew_matrix;
        
        homogenic_transformation.block(0, 0, 3, 3) << identity + sin_over_norm * angular_velocity_skew_matrix +
            one_minus_cos_over_norm_sqrd * angular_velocity_skew_matrix_squared;
        
        if (linear_velocity.norm() > 0) {
            
            homogenic_transformation.block(0, 3, 3, 1) << (identity +
                one_minus_cos_over_norm_sqrd * angular_velocity_skew_matrix +
                ((1 - sin_over_norm) / (angular_velocity_norm_sqrt)) * angular_velocity_skew_matrix_squared) *
                linear_velocity;
            
        }
        
    } else homogenic_transformation.block(0, 3, 3, 1) << linear_velocity;
    
    return homogenic_transformation;
    
}

Matrix4d cgmapping::exponential_map_se3(Vector6d &twist_velocity, double time_elapsed) {
    
    Vector6d twist_vel;
    
    twist_vel << twist_velocity * time_elapsed;
    
    return cgmapping::exponential_map_se3(twist_vel);
    
}

cgmapping::Vector6d cgmapping::logarithmic_map_se3(Matrix4d &homogenic_transformation) {
    
    //as long as no time is informed we recur to variable phi
    double phi;
    double traceR_minus_one = homogenic_transformation.block(0, 0, 3, 3).trace() - 1.0;
    double trR_min_one_over_two = traceR_minus_one / 2.0;
    
    phi = acos(trR_min_one_over_two);
    
    return cgmapping::logarithmic_map_se3(homogenic_transformation, phi);
    
}

cgmapping::Vector6d cgmapping::logarithmic_map_se3(Matrix4d &homogenic_transformation, double time_elapsed) {
    
    Vector3d angular_velocity, linear_velocity;
    Matrix3d rotation_matrix, angular_velocity_skew_matrix, angular_velocity_skew_matrix_sqrd;
    
    rotation_matrix << homogenic_transformation.block(0, 0, 3, 3);
    
    angular_velocity = cgmapping::logarithmic_map_so3(rotation_matrix, time_elapsed);
    
    angular_velocity_skew_matrix << 0, -angular_velocity(2, 0), angular_velocity(1, 0),
        angular_velocity(2, 0), 0, -angular_velocity(0, 0),
        -angular_velocity(1, 0), angular_velocity(0, 0), 0;
    
    angular_velocity_skew_matrix = angular_velocity_skew_matrix * time_elapsed;
    
    angular_velocity_skew_matrix_sqrd = angular_velocity_skew_matrix * angular_velocity_skew_matrix;
    
    Matrix3d identity;
    
    identity.setIdentity();
    
    double angular_velocity_norm = angular_velocity.norm();
    double angular_velocity_norm_sqrd = angular_velocity_norm * angular_velocity_norm;
    double sin_ang_vel_norm = sin(angular_velocity_norm);
    double sin_over_norm = sin_ang_vel_norm / angular_velocity_norm;
    
    double cos_ang_vel_norm = cos(angular_velocity_norm);
    double one_minus_cos_over_norm_sqrd = (1 - cos_ang_vel_norm) / (angular_velocity_norm_sqrd);
    
    Matrix3d V_inverse;
    
    V_inverse << identity - 0.5 * angular_velocity_skew_matrix +
        (1 / angular_velocity_norm_sqrd) * (1 - (sin_over_norm / (2 * one_minus_cos_over_norm_sqrd))) *
            angular_velocity_skew_matrix_sqrd;
    
    Vector3d translation;
    
    translation << homogenic_transformation.block(0, 3, 3, 1);
    
    linear_velocity = V_inverse * translation;
    
    linear_velocity = linear_velocity / time_elapsed;
    
    Vector6d twist_velocity;
    
    twist_velocity << linear_velocity, angular_velocity;
    
    return twist_velocity;
    
}

void cgmapping::cuda::exponential_Dmap_se3(culina_vector3d &d_linear_velocity,
                                           culina_vector3d &d_angular_velocity,
                                           culina_matrix4d &d_homogenic_transformation,
                                           culina_matrix3d &d_auxiliar_matrix1,
                                           culina_matrix3d &d_auxiliar_matrix2,
                                           culina_matrix3d &d_auxiliar_matrix3,
                                           cudaStream_t *strm1,
                                           cudaStream_t *strm2,
                                           cudaStream_t *strm3,
                                           double time_elapsed) {
    
    culiopD_t parameters;
    parameters.alpha = time_elapsed;
    parameters.strm = strm1;
    
    parameters.op_m1 = CUBLAS_OP_N;
    parameters.op_m2 = CUBLAS_OP_N;
    
    d_homogenic_transformation._setIdentity();
    
    double angular_velocity_norm = 1;
    
    cuLiNA::culina_Dnorm((culina_base_matrix<double> *) &d_angular_velocity,
                         &angular_velocity_norm,
                         parameters);
    
    if (angular_velocity_norm > 0) {
        
        double angular_velocity_norm_sqrd = angular_velocity_norm * angular_velocity_norm;
        double sin_ang_vel_norm = sin(angular_velocity_norm);
        double sin_over_norm = sin_ang_vel_norm / angular_velocity_norm;
        
        double cos_ang_vel_norm = cos(angular_velocity_norm);
        double one_minus_cos_over_norm_sqrd = (1 - cos_ang_vel_norm) / (angular_velocity_norm_sqrd);
        
        double A = sin_over_norm;
        double B = one_minus_cos_over_norm_sqrd;
        double C = (1 - A)/(angular_velocity_norm_sqrd);
        
        cuLiNA_error_t stat;
    
        //this matrix is a skew-matrix of the angular velocity;
        stat = cuLiNA::culina_Dskew_matrix3x3_operator(&d_angular_velocity, &d_auxiliar_matrix1, parameters);
        cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
        
//        d_auxiliar_matrix1(0, 0) = 0;
//        d_auxiliar_matrix1(0, 1) = -d_angular_velocity(2, 0) * time_elapsed;
//        d_auxiliar_matrix1(0, 2) = d_angular_velocity(1, 0) * time_elapsed;
//        d_auxiliar_matrix1(1, 0) = d_angular_velocity(2, 0) * time_elapsed;
//        d_auxiliar_matrix1(1, 1) = 0;
//        d_auxiliar_matrix1(1, 2) = -d_angular_velocity(0, 0) * time_elapsed;
//        d_auxiliar_matrix1(2, 0) = -d_angular_velocity(1, 0) * time_elapsed;
//        d_auxiliar_matrix1(2, 1) = d_angular_velocity(0, 0) * time_elapsed;
//        d_auxiliar_matrix1(2, 2) = 0;
        
        double alpha1 = sin_over_norm;
        double alpha2 = one_minus_cos_over_norm_sqrd;
        double alpha3 = (1 - alpha1) / angular_velocity_norm_sqrd;
        
        parameters.alpha = 1;
        parameters.beta = 0;
        parameters.gamma = 0;
        
        cuLiNA::culina_matrix_Dmultiplication((culina_base_matrix<double> *) &d_auxiliar_matrix1,
                                              (culina_base_matrix<double> *) &d_auxiliar_matrix1,
                                              (culina_base_matrix<double> *) &d_auxiliar_matrix2,
                                              parameters);
        
        parameters.alpha = 1;
        parameters.beta = alpha1;
        parameters.gamma = alpha2;
        
        cuLiNA::culina_matrix_Dsum((culina_base_matrix<double> *) &identity3d,
                                   (culina_base_matrix<double> *) &d_auxiliar_matrix1,
                                   (culina_base_matrix<double> *) &d_auxiliar_matrix2,
                                   parameters);
        
        for (int j = 0; j < d_auxiliar_matrix2._getRows(); ++j) {
            
            for (int k = 0; k < d_auxiliar_matrix2._getColumns(); ++k) {
                
                d_homogenic_transformation(j, k) = d_auxiliar_matrix2(j, k);
                
            }
            
        }
        
        parameters.alpha = 1;
        parameters.beta = 0;
        parameters.gamma = 0;
        
        cuLiNA::culina_matrix_Dmultiplication((culina_base_matrix<double> *) &d_auxiliar_matrix1,
                                              (culina_base_matrix<double> *) &d_auxiliar_matrix1,
                                              (culina_base_matrix<double> *) &d_auxiliar_matrix2,
                                              parameters);
        
        parameters.alpha = 1;
        parameters.beta = alpha2;
        parameters.gamma = alpha3;
        
        cuLiNA::culina_matrix_Dsum((culina_base_matrix<double> *) &identity3d,
                                   (culina_base_matrix<double> *) &d_auxiliar_matrix1,
                                   (culina_base_matrix<double> *) &d_auxiliar_matrix2,
                                   parameters);
        
        //cudaDeviceSynchronize();
        
        parameters.alpha = time_elapsed;
        parameters.beta = 0;
        
        int prev_columns = d_auxiliar_matrix1._getColumns();
        
        d_auxiliar_matrix1._setColumns(1);
        
        cuLiNA::culina_matrix_Dmultiplication((culina_base_matrix<double> *) &d_auxiliar_matrix2,
                                              (culina_base_matrix<double> *) &d_linear_velocity,
                                              (culina_base_matrix<double> *) &d_auxiliar_matrix1,
                                              parameters);
        
        //cudaDeviceSynchronize();
        
        d_homogenic_transformation(0, 3) = d_auxiliar_matrix1(0, 0);
        d_homogenic_transformation(1, 3) = d_auxiliar_matrix1(1, 0);
        d_homogenic_transformation(2, 3) = d_auxiliar_matrix1(2, 0);
        
        d_auxiliar_matrix1._setColumns(prev_columns);
        
    } else {
        
        d_homogenic_transformation(0, 3) = d_linear_velocity(0, 0) * time_elapsed;
        d_homogenic_transformation(1, 3) = d_linear_velocity(1, 0) * time_elapsed;
        d_homogenic_transformation(2, 3) = d_linear_velocity(2, 0) * time_elapsed;
        
    }
    
}

void cgmapping::cuda::logarithmic_Dmap_se3(culina_vector3d &d_linear_velocity,
                                           culina_vector3d &d_angular_velocity,
                                           culina_matrix4d &d_homogenic_transformation,
                                           culina_matrix3d &d_auxiliar_matrix1,
                                           culina_matrix3d &d_auxiliar_matrix2,
                                           cudaStream_t *strm,
                                           double time_elapsed) {
    
    culiopD_t parameters;
    parameters.strm = strm;
    
    double phi;
    
    //as long as no time is informed we recur to variable phi
    double traceR = 0;
    
    for (int j = 0; j < 3; ++j) {
        
        traceR += d_homogenic_transformation(j, j);
        
    }
    
    double traceR_minus_3_plus_2t = traceR - 3 + 2 * time_elapsed;
    double trR_minus_3p2t_over_2tt = traceR_minus_3_plus_2t / (2.0 * time_elapsed * time_elapsed);
    
    phi = acos(trR_minus_3p2t_over_2tt) / time_elapsed;
    
    double phi_over_2sin_phi = phi / (2 * sin(phi));
    
    parameters.alpha = phi_over_2sin_phi;
    parameters.beta = -phi_over_2sin_phi;
    parameters.op_m2 = CUBLAS_OP_T;
    parameters.gamma = 0;
    
    for (int k = 0; k < 3; ++k)
        for (int j = 0; j < 3; ++j)
            d_auxiliar_matrix1(k, j) = d_homogenic_transformation(k, j);
    
    cuLiNA::cuLiNA_error_t culina_stat;
    
    culina_stat = cuLiNA::culina_matrix_Dsum(&d_auxiliar_matrix1, &d_auxiliar_matrix1, &d_auxiliar_matrix2, parameters);
    cuLiNACheckErrors(culina_stat, __FILE__, __FUNCTION__);
    
    d_angular_velocity(0, 0) = d_auxiliar_matrix2(2, 1) * (1 / time_elapsed);
    d_angular_velocity(1, 0) = d_auxiliar_matrix2(0, 2);
    d_angular_velocity(2, 0) = d_auxiliar_matrix2(1, 0);
    
    double angular_velocity_norm = 1;
    
    culina_stat = cuLiNA::culina_Dnorm((culina_base_matrix<double> *) &d_angular_velocity,
                                       &angular_velocity_norm,
                                       parameters);
    cuLiNACheckErrors(culina_stat, __FILE__, __FUNCTION__);
    
    parameters.alpha = 1;
    parameters.beta = -0.5 * phi;
    parameters.gamma = 0;
    parameters.op_m2 = CUBLAS_OP_N;
    
    culina_stat = cuLiNA::culina_matrix_Dsum(&identity3d, &d_auxiliar_matrix2, &d_auxiliar_matrix1, parameters);
    
    parameters.alpha = 1;
    parameters.beta = 0;
    
    culina_stat = cuLiNA::culina_matrix_Dmultiplication((culina_base_matrix<double> *) &d_auxiliar_matrix1,
                                                        (culina_base_matrix<double> *) &d_auxiliar_matrix1,
                                                        (culina_base_matrix<double> *) &d_auxiliar_matrix1,
                                                        parameters);
    
    cuLiNACheckErrors(culina_stat, __FILE__, __FUNCTION__);
    
    double angular_velocity_norm_sqrd = angular_velocity_norm * angular_velocity_norm;
    double sin_ang_vel_norm = sin(angular_velocity_norm);
    double sin_over_norm = sin_ang_vel_norm / angular_velocity_norm;
    
    double cos_ang_vel_norm = cos(angular_velocity_norm);
    double one_minus_cos_over_norm_sqrd = (1 - cos_ang_vel_norm) / (angular_velocity_norm_sqrd);
    
    parameters.alpha = 1;
    parameters.beta = (1 / angular_velocity_norm_sqrd) * (1 - (sin_over_norm / (2 * one_minus_cos_over_norm_sqrd)));
    parameters.gamma = 0;
    
    culina_stat = cuLiNA::culina_matrix_Dsum(&d_auxiliar_matrix2, &d_auxiliar_matrix1, &d_auxiliar_matrix2, parameters);
    cuLiNACheckErrors(culina_stat, __FILE__, __FUNCTION__);
    
    d_linear_velocity(0, 0) = d_homogenic_transformation(0, 3);
    d_linear_velocity(1, 0) = d_homogenic_transformation(1, 3);
    d_linear_velocity(2, 0) = d_homogenic_transformation(2, 3);
    
    parameters.alpha = (1 / phi);
    parameters.beta = 0;
    parameters.gamma = 0;
    
    culina_stat = cuLiNA::culina_matrix_Dmultiplication((culina_base_matrix<double> *) &d_auxiliar_matrix2,
                                                        (culina_base_matrix<double> *) &d_linear_velocity,
                                                        (culina_base_matrix<double> *) &d_linear_velocity,
                                                        parameters);
    
    cuLiNACheckErrors(culina_stat, __FILE__, __FUNCTION__);
    
    return;
    
}
