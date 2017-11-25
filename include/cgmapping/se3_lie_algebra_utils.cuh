//
// Created by spades on 21/05/17.
//

#ifndef CGMAPPING_SE3_LIE_ALGEBRA_UTILS_H
#define CGMAPPING_SE3_LIE_ALGEBRA_UTILS_H

#include <Eigen/Dense>
#include <cuLiNA/culina_matrix.h>
#include <cuLiNA/culina_operations.h>
#include <cuLiNA/culina_definition.h>
#include <math.h>

#define NEAR_ZERO_VALUE 0.000001

using namespace Eigen;
using namespace cuLiNA;

namespace cgmapping {
    
    typedef Matrix<double, 6, 1> Vector6d;

    extern void logarithmic_Dmap_se3(Matrix4d &homogenic_transformation,
                                     Vector3d &linear_vel,
                                     Vector3d &angular_vel,
                                     double time_elapsed = 1.);
    
    extern void adjoint_Dse3(Matrix4d &homogenic_transformation,
                             Matrix<double,6,6> &adjoint_matrix);
    
    namespace cuda {
        
        //all matrices must be already initialized for full performance
        
        cuLiNA_error_t exponential_Dmap_se3(culina_vector3d &d_linear_velocity,
                                            culina_vector3d &d_angular_velocity,
                                            culina_matrix4d &d_homogenic_transformation,
                                            culiopD_t &culiopD_1 = culiopD_default,
                                            culiopD_t &culiopD_2 = culiopD_default,
                                            double time_elapsed = 1.0);
        
        cuLiNA_error_t logarithmic_Dmap_se3(culina_matrix4d &d_homogenic_transformation,
                                            culina_vector3d &d_linear_velocity,
                                            culina_vector3d &d_angular_velocity,
                                            culiopD_t &culiopD = culiopD_default,
                                            double time_elapsed = 1.0);
        
        cuLiNA_error_t adjoint_Dse3(culina_matrix4d &d_homogenic_transformation,
                                    culina_matrix<double,6,6> &d_adjoint_matrix,
                                    culiopD_t &culiopD = culiopD_default);
        
    }
    
}

#endif //CGMAPPING_SE3_LIE_ALGEBRA_UTILS_H
