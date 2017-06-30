//
// Created by spades on 21/05/17.
//

#include "cgmapping/se3_lie_algebra_utils.h"

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

void cgmapping::cuda::exponential_map_se3(culina_vector6d &d_twist_velocity,
                                          culina_matrix4d &d_homogenic_transformation,
                                          culina_matrix3d &d_auxiliar_matrix) {
    
    return ;
    
}