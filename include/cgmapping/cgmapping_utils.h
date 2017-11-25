//
// Created by spades on 05/10/17.
//

#ifndef CGMAPPING_CGMAPPING_UTILS_H
#define CGMAPPING_CGMAPPING_UTILS_H

#include <cgmapping/cgmapping_utils.cuh>
#include <fstream>

#include <cuLiNA/culina_base_matrix.h>
#include <cuLiNA/culina_matrix.h>
#include <cuLiNA/culina_operations.h>
#include <cgmapping/se3_lie_algebra_utils.cuh>

#include <geometry_msgs/Pose.h>

#include "rgb_d_camera_model.h"

namespace cgmapping{
    
    extern void calculate_Dlog_map_se3(Eigen::Vector3d &linear_vel,
                                       Eigen::Vector3d &angular_vel,
                                       Eigen::Matrix4d &homogenic_matrix,
                                       double time_elapsed = 1.);
    
    extern void calculate_Dadjoit_se3(Eigen::Matrix4d &homogenic_matrix,
                                      Eigen::Matrix<double,6,6> &adjoint_matrix);
    
    extern void syncronize_fstreams_from_rgbd_data_set(std::fstream &rgb_stream,
                                                       std::fstream &depth_stream,
                                                       std::fstream &groundtruth_stream);
    
    extern void eigen_pose_matrix4d_2_ros_geometry_msg_pose(Eigen::Matrix4d &eigen_pose, geometry_msgs::Pose &geo_msg_pose);
    
    extern void ros_geometry_msg_pose_2_eigen_pose_matrix4d(geometry_msgs::Pose &geo_msg_pose, Eigen::Matrix4d &eigen_pose);
    
    extern void poses_to_file(std::vector<double> &time_stamps,
                              std::vector<geometry_msgs::Pose> &poses,
                              std::string &file_name);
    
    extern void pose_line_2_eigen_pose_matrix4d(std::string &pose_string, Eigen::Matrix4d &eigen_pose);
    
    namespace cuda {
    
        extern int calculate_Dnumber_of_valid_data(cuLiNA::culina_tm<double> &data,
                                                   cv::cuda::Stream &stream = cv::cuda::Stream::Null());
    
        extern double calculate_Dstandard_deviation_t_student(cuLiNA::culina_tm<double> &data,
                                                              double degrees_of_freedom,
                                                              double standard_deviation_initial_guess,
                                                              int number_of_valid_data,
                                                              double acceptance_epsilon,
                                                              cv::cuda::Stream &stream = cv::cuda::Stream::Null());
    
        extern void calculate_Dweight_matrix(cuLiNA::culina_tm<double> &data,
                                             cuLiNA::culina_tm<double> &weight_matrix,
                                             double degrees_of_freedom,
                                             double variance,
                                             cv::cuda::Stream &stream = cv::cuda::Stream::Null());
    
        extern void calculate_Dsquared_weighted_error(cuLiNA::culina_tm<double> &data,
                                                      cuLiNA::culina_tm<double> &weight_matrix,
                                                      cuLiNA::culina_matrix<double, 1, 1> &squared_weight_error,
                                                      cuLiNA::culina_tm<double> &auxiliar_matrix,
                                                      int number_of_valid_data,
                                                      cv::cuda::Stream &stream = cv::cuda::Stream::Null());
        
        extern void calculate_Dimage_warped(cv::cuda::GpuMat &img_original,
                                            cv::cuda::GpuMat &img_warped,
                                            cv::cuda::GpuMat &img_warped_filter,
                                            cv::cuda::GpuMat &depth_img_reference,
                                            cuLiNA::culina_tm<double> &depth_img_point_cloud,
                                            cuLiNA::culina_matrix4d &homogenic_transformation,
                                            cgmapping::rgb_d_camera_model &camera_model,
                                            cv::cuda::Stream &stream = cv::cuda::Stream::Null());
    
        extern void calculate_Dimage_residual(cv::cuda::GpuMat &img1,
                                              cv::cuda::GpuMat &img2,
                                              cv::cuda::GpuMat &filter,
                                              cuLiNA::culina_tm<double> &residuals,
                                              cv::cuda::Stream &stream = cv::cuda::Stream::Null());
        
        extern void calculate_Dwarp_jacobian(cuLiNA::culina_tm<double> &d_warp_jacobian,
                                             cuLiNA::culina_tm<double> &d_img1_point_cloud,
                                             cgmapping::rgb_d_camera_model &h_camera_model,
                                             cv::cuda::Stream &strm = cv::cuda::Stream::Null());
        
        extern void calculate_Dfull_jacobian(cv::cuda::GpuMat &warped_img_x_derivative,
                                             cv::cuda::GpuMat &warped_img_y_derivative,
                                             cuLiNA::culina_tm<double> &d_warp_jacobian,
                                             cuLiNA::culina_tm<double> &d_jacobian,
                                             cv::cuda::Stream &strm = cv::cuda::Stream::Null());
    
        extern void calculate_Dexp_map_se3(culina_vector3d &d_linear_vel,
                                           culina_vector3d &d_angular_vel,
                                           culina_matrix4d &d_homogenic_matrix,
                                           culina_matrix3d &auxiliar_matrix1,
                                           culina_matrix3d &auxiliar_matrix2,
                                           cv::cuda::Stream &strm1 = cv::cuda::Stream::Null(),
                                           cv::cuda::Stream &strm2 = cv::cuda::Stream::Null(),
                                           double time_elapsed = 1.);
        
        extern void calculate_Dlog_map_se3(culina_vector3d &d_linear_vel,
                                           culina_vector3d &d_angular_vel,
                                           culina_matrix4d &d_homogenic_matrix,
                                           culina_matrix3d &auxiliar_matrix,
                                           cv::cuda::Stream &strm = cv::cuda::Stream::Null(),
                                           double time_elapsed = 1.);
    
        extern void calculate_Dinverse(cuLiNA::culina_tm<double> &original_matrix,
                                       cuLiNA::culina_tm<double> &result_matrix,
                                       cv::cuda::Stream &strm = cv::cuda::Stream::Null());
        
        extern void solve_Dlinear_system(culina_tm<double> &jacobian,
                                         culina_tm<double> &delta,
                                         culina_tm<double> &data,
                                         culina_tm<double> &weight,
                                         double k_lambda_scalar,
                                         culina_tm<double> &auxiliar_matrix1,
                                         culina_tm<double> &auxiliar_matrix2,
                                         culina_tm<double> &auxiliar_matrix3,
                                         cv::cuda::Stream &strm_1 = cv::cuda::Stream::Null(),
                                         cv::cuda::Stream &strm_2 = cv::cuda::Stream::Null(),
                                         cv::cuda::Stream &strm_3 = cv::cuda::Stream::Null());
    
        extern void solve_Dlinear_system_with_prior(culina_tm<double> &jacobian,
                                                    culina_tm<double> &delta,
                                                    culina_tm<double> &data,
                                                    culina_tm<double> &weight,
                                                    culina_tm<double> &motion_prior,
                                                    culina_tm<double> &cov_motion_prior,
                                                    culina_tm<double> &estimation_k,
                                                    double k_lambda_scalar,
                                                    culina_tm<double> &auxiliar_matrix1,
                                                    culina_tm<double> &auxiliar_matrix2,
                                                    culina_tm<double> &auxiliar_matrix3,
                                                    cv::cuda::Stream &strm_1 = cv::cuda::Stream::Null(),
                                                    cv::cuda::Stream &strm_2 = cv::cuda::Stream::Null(),
                                                    cv::cuda::Stream &strm_3 = cv::cuda::Stream::Null());
    
        extern void compose_Dpose_3D(culina_matrix4d &d_homogenic_transformation_1,
                                     culina_matrix4d &d_homogenic_transformation_2,
                                     culina_matrix4d &d_resultant_homogenic_transformation,
                                     cv::cuda::Stream &strm = cv::cuda::Stream::Null());
        
        extern void transform_Dspecial_euclian_3D_orientation2quaternions(culina_matrix4d &d_homogenic_matrix,
                                                                          Vector4d &quaternion,
                                                                          cv::cuda::Stream &strm = cv::cuda::Stream::Null());
        
        extern void compose_Dpose_uncertainty_SE3(culina_matrix4d &d_homogenic_transformation,
                                                  culina_matrix4d &d_delta_homogenic_transformation,
                                                  culina_matrix<double, 6, 6> &covariance_matrix,
                                                  culina_matrix<double, 6, 6> &delta_covariance_matrix,
                                                  culina_matrix<double, 6, 6> &d_auxiliar_matrix_1,
                                                  culina_matrix<double, 6, 6> &d_auxiliar_matrix_2,
                                                  cv::cuda::Stream &strm);
        
        template <typename T, typename OutPutStream = std::fstream>
        extern void print_cv_img_to_file(cv::Mat &img, std::string &file_name);
        
    
    }
    
    
};

#endif //CGMAPPING_CGMAPPING_UTILS_H
