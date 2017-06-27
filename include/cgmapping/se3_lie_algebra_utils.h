//
// Created by spades on 21/05/17.
//

#ifndef CGMAPPING_SE3_LIE_ALGEBRA_UTILS_H
#define CGMAPPING_SE3_LIE_ALGEBRA_UTILS_H

#include <Eigen/Dense>
#include <math.h>

using namespace Eigen;

namespace cgmapping{

  typedef Matrix<double, 6, 1> Vector6d;

  Matrix3d exponential_map_so3(Vector3d &angular_velocity);
  Matrix3d exponential_map_so3(Vector3d &angular_velocity, double time_elapsed);
  Vector3d logarithmic_map_so3(Matrix3d &rotation_matrix);
  Vector3d logarithmic_map_so3(Matrix3d &rotation_matrix, double time_elapsed);

  Matrix4d exponential_map_se3(Vector6d &twist_velocity);
  Matrix4d exponential_map_se3(Vector6d &twist_velocity, double time_elapsed);
  Vector6d logarithmic_map_se3(Matrix4d &homegenic_transformation);
  Vector6d logarithmic_map_se3(Matrix4d &homogenic_transformation, double time_elapsed);

  namespace cuda {

    typedef double* cuBLASMat;

//    Matrix3d exponential_map_so3(Vector3d &angular_velocity);
//    Matrix3d exponential_map_so3(Vector3d &angular_velocity, double time_elapsed);
//    Vector3d logarithmic_map_so3(Matrix3d &rotation_matrix);
//    Vector3d logarithmic_map_so3(Matrix3d &rotation_matrix, double time_elapsed);
//
//    Matrix4d exponential_map_se3(Vector6d &twist_velocity);
//    Matrix4d exponential_map_se3(Vector6d &twist_velocity, double time_elapsed);
//    Vector6d logarithmic_map_se3(Matrix4d &homegenic_transformation);
//    Vector6d logarithmic_map_se3(Matrix4d &homogenic_transformation, double time_elapsed);

  }

}

#endif //CGMAPPING_SE3_LIE_ALGEBRA_UTILS_H
