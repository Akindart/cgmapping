//
// Created by spades on 05/10/17.
//

#include <cgmapping/cgmapping_utils.h>

#include <vector>

#include <opencv2/core/cuda_stream_accessor.hpp>

#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/lexical_cast.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

int cgmapping::cuda::calculate_Dnumber_of_valid_data(cuLiNA::culina_tm<double> &data, cv::cuda::Stream &stream) {
    
    auto strm = cv::cuda::StreamAccessor::getStream(stream);
    
    return (int) cgmapping::cuda::count_valid_data(data, &strm);
    
}

double cgmapping::cuda::calculate_Dstandard_deviation_t_student(cuLiNA::culina_tm<double> &data,
                                                                double degrees_of_freedom,
                                                                double standard_deviation_initial_guess,
                                                                int number_of_valid_data,
                                                                double acceptance_epsilon,
                                                                cv::cuda::Stream &stream) {
    
    auto strm = cv::cuda::StreamAccessor::getStream(stream);
    
    double variance_k_minus_1;
    double variance_k = (standard_deviation_initial_guess)*(standard_deviation_initial_guess);
    
    do {
    
        variance_k_minus_1 = variance_k;
        
        variance_k = cgmapping::cuda::variance_t_student_step_Dcalculation_operation(data,
                                                                                     degrees_of_freedom,
                                                                                     variance_k_minus_1,
                                                                                     number_of_valid_data,
                                                                                     &strm);
        
    } while (abs(variance_k - variance_k_minus_1) > acceptance_epsilon);
    
    return std::sqrt(variance_k);
    
}

void cgmapping::cuda::calculate_Dweight_matrix(cuLiNA::culina_tm<double> &data,
                                               cuLiNA::culina_tm<double> &weight_matrix,
                                               double degrees_of_freedom,
                                               double variance,
                                               cv::cuda::Stream &stream) {
    
    auto strm = cv::cuda::StreamAccessor::getStream(stream);
    
    cgmapping::cuda::weight_matrix_t_student_Dcalculation_operation(data,
                                                                    weight_matrix,
                                                                    degrees_of_freedom,
                                                                    variance,
                                                                    &strm);
    
}

void cgmapping::cuda::calculate_Dsquared_weighted_error(cuLiNA::culina_tm<double> &data,
                                                        cuLiNA::culina_tm<double> &weight_matrix,
                                                        cuLiNA::culina_matrix<double, 1, 1> &squared_weight_error,
                                                        cuLiNA::culina_tm<double> &auxiliar_matrix,
                                                        int number_of_valid_data,
                                                        cv::cuda::Stream &stream) {
    
    auto strm = cv::cuda::StreamAccessor::getStream(stream);
    
    auto one_over_num_valid_data = 1./number_of_valid_data;
    
    cuLiNA::culiopD_t tmp_culiop;
    
    tmp_culiop.alpha = one_over_num_valid_data;
    tmp_culiop.beta = 0;
    tmp_culiop.op_m1 = CUBLAS_OP_T;
    tmp_culiop.strm = &strm;
    
//    data._printMatrix(false, true);
//    weight_matrix._printMatrix(false, true);
//    auxiliar_matrix._printMatrix(false, true);
    
    auto stat = cuLiNA::culina_matrix_Dmultiplication(&data, &weight_matrix, &auxiliar_matrix, tmp_culiop);
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    tmp_culiop.alpha = 1;
    tmp_culiop.beta = 0;
    tmp_culiop.op_m1 = CUBLAS_OP_N;
    
    stat = cuLiNA::culina_matrix_Dmultiplication(&auxiliar_matrix, &data, &squared_weight_error, tmp_culiop);
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
}

void cgmapping::cuda::calculate_Dimage_warped(cv::cuda::GpuMat &img_original,
                                              cv::cuda::GpuMat &img_warped,
                                              cv::cuda::GpuMat &img_warped_filter,
                                              cv::cuda::GpuMat &depth_img_reference,
                                              cuLiNA::culina_tm<double> &depth_img_point_cloud,
                                              cuLiNA::culina_matrix4d &homogenic_transformation,
                                              cgmapping::rgb_d_camera_model &camera_model,
                                              cv::cuda::Stream &stream) {
    
    auto strm = cv::cuda::StreamAccessor::getStream(stream);
    
    cgmapping::cuda::warped_image_Dcalculation_operation(img_original,
                                                         img_warped,
                                                         img_warped_filter,
                                                         depth_img_reference,
                                                         depth_img_point_cloud,
                                                         homogenic_transformation,
                                                         camera_model._getFocus_x(),
                                                         camera_model._getFocus_y(),
                                                         camera_model._getCentroid_x(),
                                                         camera_model._getCentroid_y(),
                                                         camera_model._getScale_for_depth(),
                                                         &strm);
    
    return ;
    
}

void cgmapping::cuda::calculate_Dimage_residual(cv::cuda::GpuMat &img1,
                                                cv::cuda::GpuMat &img2,
                                                cv::cuda::GpuMat &filter,
                                                cuLiNA::culina_tm<double> &residuals,
                                                cv::cuda::Stream &stream) {
    
    auto strm = cv::cuda::StreamAccessor::getStream(stream);
    
    cgmapping::cuda::pixel_residual_Dcalculation_operation(img1, img2, filter, residuals, &strm);
    
    return ;
    
}

void cgmapping::cuda::calculate_Dwarp_jacobian(cuLiNA::culina_tm<double> &d_warp_jacobian,
                                               cuLiNA::culina_tm<double> &d_img1_point_cloud,
                                               cgmapping::rgb_d_camera_model &h_camera_model,
                                               cv::cuda::Stream &strm) {
    
    cgmapping::cuda::warp_jacobian_Dcalculation_operation(d_warp_jacobian, d_img1_point_cloud, h_camera_model, strm);
    
    return;
    
}

void cgmapping::cuda::calculate_Dfull_jacobian(cv::cuda::GpuMat &warped_img_x_derivative,
                                               cv::cuda::GpuMat &warped_img_y_derivative,
                                               cuLiNA::culina_tm<double> &d_warp_jacobian,
                                               cuLiNA::culina_tm<double> &d_jacobian,
                                               cv::cuda::Stream &strm) {
    
    cgmapping::cuda::full_jacobian_Dcalculation_operation(warped_img_x_derivative,
                                                          warped_img_y_derivative,
                                                          d_warp_jacobian,
                                                          d_jacobian,
                                                          strm);
    
    return;
    
}

void cgmapping::cuda::calculate_Dexp_map_se3(culina_vector3d &d_linear_vel,
                                             culina_vector3d &d_angular_vel,
                                             culina_matrix4d &d_homogenic_matrix,
                                             culina_matrix3d &auxiliar_matrix1,
                                             culina_matrix3d &auxiliar_matrix2,
                                             cv::cuda::Stream &strm1,
                                             cv::cuda::Stream &strm2,
                                             double time_elapsed) {
    
    cuLiNA::culiopD_t culiopD_1, culiopD_2;
    
    static auto raw_strm_1 = cv::cuda::StreamAccessor::getStream(strm1);
    static auto raw_strm_2 = cv::cuda::StreamAccessor::getStream(strm2);
    
    culiopD_1.strm = &raw_strm_1;
    culiopD_2.strm = &raw_strm_2;
    
    culiopD_1.workspace = &auxiliar_matrix1;
    culiopD_2.workspace = &auxiliar_matrix2;
    
    auto stat = cgmapping::cuda::exponential_Dmap_se3(d_linear_vel,
                                                      d_angular_vel,
                                                      d_homogenic_matrix,
                                                      culiopD_1,
                                                      culiopD_2,
                                                      time_elapsed);
    
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    culiopD_1.workspace = NULL;
    culiopD_2.workspace = NULL;

}

void cgmapping::cuda::calculate_Dlog_map_se3(culina_vector3d &d_linear_vel,
                                             culina_vector3d &d_angular_vel,
                                             culina_matrix4d &d_homogenic_matrix,
                                             culina_matrix3d &auxiliar_matrix,
                                             cv::cuda::Stream &strm,
                                             double time_elapsed) {
    
    cuLiNA::culiopD_t culiopD;
    
    auto raw_strm = cv::cuda::StreamAccessor::getStream(strm);
    
    culiopD.strm = &raw_strm;
    
    culiopD.workspace = &auxiliar_matrix;
    
    auto stat = cgmapping::cuda::logarithmic_Dmap_se3(d_homogenic_matrix, d_linear_vel, d_angular_vel, culiopD, time_elapsed);
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    culiopD.workspace = NULL;
    
}

void cgmapping::calculate_Dlog_map_se3(Eigen::Vector3d &linear_vel,
                                       Eigen::Vector3d &angular_vel,
                                       Eigen::Matrix4d &homogenic_matrix,
                                       double time_elapsed) {
    
    cgmapping::logarithmic_Dmap_se3(homogenic_matrix, linear_vel, angular_vel, time_elapsed);
    
    
}

void cgmapping::cuda::calculate_Dinverse(cuLiNA::culina_tm<double> &original_matrix,
                                         cuLiNA::culina_tm<double> &result_matrix,
                                         cv::cuda::Stream &strm) {
    
    culiopD_t tmp_culiopD;
    
    cudaMallocManaged(&(tmp_culiopD.dev_info), sizeof(int));
    
    auto raw_strm = cv::cuda::StreamAccessor::getStream(strm);
    
    tmp_culiopD.strm = &raw_strm;
    
    cuLiNA::culina_Dinverse_matrix(&original_matrix, &result_matrix, tmp_culiopD);
    
}


void cgmapping::calculate_Dadjoit_se3(Eigen::Matrix4d &homogenic_matrix, Eigen::Matrix<double, 6, 6> &adjoint_matrix) {

    cgmapping::adjoint_Dse3(homogenic_matrix, adjoint_matrix);
    
}

void cgmapping::cuda::solve_Dlinear_system(culina_tm<double> &jacobian,
                                           culina_tm<double> &delta,
                                           culina_tm<double> &data,
                                           culina_tm<double> &weight,
                                           double k_lambda_scalar,
                                           culina_tm<double> &auxiliar_matrix1,
                                           culina_tm<double> &auxiliar_matrix2,
                                           culina_tm<double> &auxiliar_matrix3,
                                           cv::cuda::Stream &strm_1,
                                           cv::cuda::Stream &strm_2,
                                           cv::cuda::Stream &strm_3) {
    
    auto raw_strm_1 = cv::cuda::StreamAccessor::getStream(strm_1);
    auto raw_strm_2 = cv::cuda::StreamAccessor::getStream(strm_2);
    auto raw_strm_3 = cv::cuda::StreamAccessor::getStream(strm_3);
    
    culiopD_t culiopD_1;
    culiopD_t culiopD_2;
    culiopD_t culiopD_3;
    
    culiopD_1.strm = &raw_strm_1;
    culiopD_2.strm = &raw_strm_2;
    culiopD_3.strm = &raw_strm_3;
    
    culiopD_1.workspace = &auxiliar_matrix1;
    culiopD_2.workspace = &auxiliar_matrix2;
    culiopD_3.workspace = &auxiliar_matrix3;
    
    auto stat = cuLiNA::culina_Dsolve_gradient_descent_first_order(&jacobian,
                                                                   &delta,
                                                                   &data,
                                                                   &weight,
                                                                   k_lambda_scalar,
                                                                   culiopD_1,
                                                                   culiopD_2,
                                                                   culiopD_3);
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    culiopD_1.workspace = NULL;
    culiopD_2.workspace = NULL;
    culiopD_3.workspace = NULL;
    
}

void cgmapping::cuda::solve_Dlinear_system_with_prior(culina_tm<double> &jacobian,
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
                                                      cv::cuda::Stream &strm_1,
                                                      cv::cuda::Stream &strm_2,
                                                      cv::cuda::Stream &strm_3) {
    
    auto raw_strm_1 = cv::cuda::StreamAccessor::getStream(strm_1);
    auto raw_strm_2 = cv::cuda::StreamAccessor::getStream(strm_2);
    auto raw_strm_3 = cv::cuda::StreamAccessor::getStream(strm_3);
    
    culiopD_t culiopD_1;
    culiopD_t culiopD_2;
    culiopD_t culiopD_3;
    
    culiopD_1.strm = &raw_strm_1;
    culiopD_2.strm = &raw_strm_2;
    culiopD_3.strm = &raw_strm_3;
    
    culiopD_1.workspace = &auxiliar_matrix1;
    culiopD_2.workspace = &auxiliar_matrix2;
    culiopD_3.workspace = &auxiliar_matrix3;
    
    auto stat = cuLiNA::culina_Dsolve_gradient_descent_first_order(&jacobian,
                                                                   &delta,
                                                                   &data,
                                                                   &weight,
                                                                   &motion_prior,
                                                                   &cov_motion_prior,
                                                                   &estimation_k,
                                                                   k_lambda_scalar,
                                                                   culiopD_1,
                                                                   culiopD_2,
                                                                   culiopD_3);
    
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    culiopD_1.workspace = NULL;
    culiopD_2.workspace = NULL;
    culiopD_3.workspace = NULL;
    
}

void cgmapping::cuda::compose_Dpose_3D(culina_matrix4d &d_homogenic_transformation_1,
                                       culina_matrix4d &d_homogenic_transformation_2,
                                       culina_matrix4d &d_resultant_homogenic_transformation,
                                       cv::cuda::Stream &strm) {
    
    auto raw_strm = cv::cuda::StreamAccessor::getStream(strm);
    
    cuLiNA::culiopD_t culiopD;
    culiopD.strm = &raw_strm;
    culiopD.alpha = 1;
    culiopD.beta = 0;
    culiopD.gamma = 0;
    
    auto stat = cuLiNA::culina_matrix_Dmultiplication(&d_homogenic_transformation_1,
                                                      &d_homogenic_transformation_2,
                                                      &d_resultant_homogenic_transformation,
                                                      culiopD);
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
}

void cgmapping::cuda::transform_Dspecial_euclian_3D_orientation2quaternions(culina_matrix4d &d_homogenic_matrix,
                                                                            Vector4d &quaternion,
                                                                            cv::cuda::Stream &strm) {
    
    auto raw_strm = cv::cuda::StreamAccessor::getStream(strm);
    
    double tmp_homogenic_matrix[16];
    
    d_homogenic_matrix._downloadData(tmp_homogenic_matrix, 16, &raw_strm);
    
    strm.waitForCompletion();
    
    double sqrd_quaternion_terms[4];
    
    int ld = d_homogenic_matrix._getLeading_dimension();
    
    auto r00 = sqrd_quaternion_terms[IDX2C(0,0, ld)];
    auto r11 = sqrd_quaternion_terms[IDX2C(1,1, ld)];
    auto r22 = sqrd_quaternion_terms[IDX2C(2,2, ld)];
    
    sqrd_quaternion_terms[0] = 0.25*(1 + r00 + r11 + r22);
    
    sqrd_quaternion_terms[1] = 0.25*(1 + r00 - r11 - r22);
    
    sqrd_quaternion_terms[2] = 0.25*(1 - r00 + r11 - r22);
    
    sqrd_quaternion_terms[3] = 0.25*(1 - r00 - r11 + r22);
    
    double max_sqrd_quaternion_value = sqrd_quaternion_terms[0];
    char max_sqrd_quaternion_index = 0;
    
    double quaternion_norm = max_sqrd_quaternion_value;
    
    for (char i = 1; i < 4; ++i) {
    
        if(max_sqrd_quaternion_value < sqrd_quaternion_terms[i]){
            
            max_sqrd_quaternion_value = sqrd_quaternion_terms[i];
            max_sqrd_quaternion_index = i;
            
        }
        
        quaternion_norm += sqrd_quaternion_terms[i];
    
    }
    
    auto tmp_value = sqrd_quaternion_terms[max_sqrd_quaternion_index];
    
    tmp_value = sqrt(tmp_value);
    quaternion_norm = sqrt(quaternion_norm);
    
    switch(max_sqrd_quaternion_index){
        
        case 0:
            
            quaternion(0) = tmp_value;
            quaternion(1) = 0.25*(tmp_homogenic_matrix[IDX2C(2,1, ld)] - tmp_homogenic_matrix[IDX2C(1,2, ld)])/tmp_value;
            quaternion(2) = 0.25*(tmp_homogenic_matrix[IDX2C(0,2, ld)] - tmp_homogenic_matrix[IDX2C(2,0, ld)])/tmp_value;
            quaternion(3) = 0.25*(tmp_homogenic_matrix[IDX2C(1,0, ld)] - tmp_homogenic_matrix[IDX2C(0,1, ld)])/tmp_value;
            
            break;
        case 1:
    
            quaternion(0) = 0.25*(tmp_homogenic_matrix[IDX2C(2,1, ld)] - tmp_homogenic_matrix[IDX2C(1,2, ld)])/tmp_value;
            quaternion(1) = tmp_value;
            quaternion(2) = 0.25*(tmp_homogenic_matrix[IDX2C(0,1, ld)] + tmp_homogenic_matrix[IDX2C(1,0, ld)])/tmp_value;
            quaternion(3) = 0.25*(tmp_homogenic_matrix[IDX2C(0,2, ld)] + tmp_homogenic_matrix[IDX2C(2,0, ld)])/tmp_value;
            
            break;
        case 2:
            
            quaternion(0) = 0.25*(tmp_homogenic_matrix[IDX2C(0,2, ld)] - tmp_homogenic_matrix[IDX2C(2,0, ld)])/tmp_value;
            quaternion(1) = 0.25*(tmp_homogenic_matrix[IDX2C(0,1, ld)] + tmp_homogenic_matrix[IDX2C(1,0, ld)])/tmp_value;
            quaternion(2) = tmp_value;
            quaternion(3) = 0.25*(tmp_homogenic_matrix[IDX2C(1,2, ld)] + tmp_homogenic_matrix[IDX2C(2,1, ld)])/tmp_value;
            
            break;
            
        case 3:
    
            quaternion(0) = 0.25*(tmp_homogenic_matrix[IDX2C(1,0, ld)] - tmp_homogenic_matrix[IDX2C(0,1, ld)])/tmp_value;
            quaternion(1) = 0.25*(tmp_homogenic_matrix[IDX2C(0,2, ld)] + tmp_homogenic_matrix[IDX2C(2,0, ld)])/tmp_value;
            quaternion(2) = 0.25*(tmp_homogenic_matrix[IDX2C(1,2, ld)] + tmp_homogenic_matrix[IDX2C(2,1, ld)])/tmp_value;
            quaternion(3) = tmp_value;
            
            break;
        
    }
    
    quaternion = quaternion/quaternion_norm;
    
}
void cgmapping::cuda::compose_Dpose_uncertainty_SE3(culina_matrix4d &d_homogenic_transformation,
                                                    culina_matrix4d &d_delta_homogenic_transformation,
                                                    culina_matrix<double, 6, 6> &covariance_matrix,
                                                    culina_matrix<double, 6, 6> &delta_covariance_matrix,
                                                    culina_matrix<double, 6, 6> &d_auxiliar_matrix_1,
                                                    culina_matrix<double, 6, 6> &d_auxiliar_matrix_2,
                                                    cv::cuda::Stream &strm) {
    
    culiopD_t tmp_culiopD;
    
    auto tmp_raw_strm = cv::cuda::StreamAccessor::getStream(strm);
    
    d_auxiliar_matrix_2._setRows(3);
    d_auxiliar_matrix_2._setColumns(3);
    
    tmp_culiopD.strm = &tmp_raw_strm;
    tmp_culiopD.workspace = &d_auxiliar_matrix_2;
    
    //auxiliar_1 <-- adjoint(homogenic_transformation)
    auto stat = cgmapping::cuda::adjoint_Dse3(d_homogenic_transformation, d_auxiliar_matrix_1, tmp_culiopD);
    cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    tmp_culiopD.workspace = NULL;
    
    d_auxiliar_matrix_2._setRows(6);
    d_auxiliar_matrix_2._setColumns(6);
    
    //auxiliar_2 <-- Ajoint(homogenic_tranformation)*Cov(delta)
    stat = cuLiNA::culina_matrix_Dmultiplication(&d_auxiliar_matrix_1, &delta_covariance_matrix, &d_auxiliar_matrix_2, tmp_culiopD);
    cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    //auxiliar_2 <-- auxiliar_2*Ajoint(homogenic_tranformation)^T;
    tmp_culiopD.op_m2 = CUBLAS_OP_T;
    stat = cuLiNA::culina_matrix_Dmultiplication(&d_auxiliar_matrix_2, &d_auxiliar_matrix_1, &d_auxiliar_matrix_2, tmp_culiopD);
    cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    tmp_culiopD.op_m2 = CUBLAS_OP_N;
    
    //cov_matrix = cov_matrix + auxiliar_2
    tmp_culiopD.beta = 1;
    stat = cuLiNA::culina_matrix_Dsum(&covariance_matrix, &d_auxiliar_matrix_2, &covariance_matrix, tmp_culiopD);
    cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
}


void cgmapping::syncronize_fstreams_from_rgbd_data_set(std::fstream &rgb_stream,
                                                       std::fstream &depth_stream,
                                                       std::fstream &groundtruth_stream) {
    
    using namespace std;
    using namespace boost;
    
    if (rgb_stream.good() && depth_stream.good() && groundtruth_stream.good()) {
        
        std::string rgb_line;
        std::string depth_line;
        std::string groundtruth_line;
        
        vector<string> rgb_field_vector;
        vector<string> depth_field_vector;
        vector<string> groundtruth_field_vector;
        
        auto rgb_stream_read_pos = rgb_stream.tellp();
        auto depth_stream_read_pos = depth_stream.tellp();
        auto groundtruth_stream_read_pos = groundtruth_stream.tellp();
        
        do {
    
            rgb_stream_read_pos = rgb_stream.tellp();
            std::getline(rgb_stream, rgb_line);
            
        }while(rgb_line[0] == '#');
    
        do{
    
            depth_stream_read_pos = depth_stream.tellp();
            std::getline(depth_stream, depth_line);
            
        }while(depth_line[0] == '#');
    
        do{
    
            groundtruth_stream_read_pos = groundtruth_stream.tellp();
            std::getline(groundtruth_stream, groundtruth_line);
        
        }while(groundtruth_line[0] == '#');
        
        split_regex(rgb_field_vector, rgb_line, regex(" "));
        split_regex(depth_field_vector, depth_line, regex(" "));
        split_regex(groundtruth_field_vector, groundtruth_line, regex(" "));
        
        auto rgb_time_stamp = atof(rgb_field_vector[0].c_str());
        auto depth_time_stamp = atof(depth_field_vector[0].c_str());
        auto groundtruth_time_stamp = atof(groundtruth_field_vector[0].c_str());
        
        int bigger_timestamp = -1; //0-rgb 1-depth 2-groundtruth (-1) all the same
        
        if(rgb_time_stamp > depth_time_stamp && rgb_time_stamp > groundtruth_time_stamp)
            bigger_timestamp = 0;
        else if(rgb_time_stamp < depth_time_stamp && depth_time_stamp > groundtruth_time_stamp)
            bigger_timestamp = 1;
        else if(rgb_time_stamp < groundtruth_time_stamp && depth_time_stamp < groundtruth_time_stamp)
            bigger_timestamp = 2;
        
        rgb_stream.seekp(rgb_stream_read_pos);
        depth_stream.seekp(depth_stream_read_pos);
        groundtruth_stream.seekp(groundtruth_stream_read_pos);
    
        auto prev_rgb_pos = rgb_stream_read_pos;
        auto prev_depth_pos = depth_stream_read_pos;
        auto prev_groundtruth_depth_pos = groundtruth_stream_read_pos;
        
        if(bigger_timestamp == 0){
            
            auto prev_time_diff = 10000000.;
            auto cur_time_diff  = prev_time_diff;
            
            do{
    
                depth_field_vector.clear();
    
                prev_depth_pos = depth_stream_read_pos;
                depth_stream_read_pos = depth_stream.tellp();
                
                if(std::getline(depth_stream, depth_line)){
    
                    split_regex(depth_field_vector, depth_line, regex(" "));
                    depth_time_stamp = atof(depth_field_vector[0].c_str());
                
                    prev_time_diff = cur_time_diff;
                    cur_time_diff = abs(rgb_time_stamp - depth_time_stamp);
                    
                }
                
            }while(cur_time_diff < prev_time_diff);
    
            prev_time_diff = 10000000.;
            cur_time_diff  = prev_time_diff;
    
            do{
    
                groundtruth_field_vector.clear();
        
                prev_groundtruth_depth_pos = groundtruth_stream_read_pos;
                groundtruth_stream_read_pos = groundtruth_stream.tellp();
                
                if(std::getline(groundtruth_stream, groundtruth_line)){
            
                    split_regex(groundtruth_field_vector, groundtruth_line, regex(" "));
                    groundtruth_time_stamp = atof(groundtruth_field_vector[0].c_str());
            
                    prev_time_diff = cur_time_diff;
                    cur_time_diff = abs(rgb_time_stamp - groundtruth_time_stamp);
            
                }
        
            }while(cur_time_diff < prev_time_diff);
            
        }
        else if(bigger_timestamp == 1){
    
            auto prev_time_diff = 10000000.;
            auto cur_time_diff  = prev_time_diff;//abs(depth_time_stamp - rgb_time_stamp);
    
            do{
        
                rgb_field_vector.clear();
    
                prev_rgb_pos = rgb_stream_read_pos;
                rgb_stream_read_pos = rgb_stream.tellp();
                
                if(std::getline(rgb_stream, rgb_line)){
            
                    split_regex(rgb_field_vector, rgb_line, regex(" "));
                    rgb_time_stamp = atof(rgb_field_vector[0].c_str());
                    
                    prev_time_diff = cur_time_diff;
                    cur_time_diff = abs(depth_time_stamp - rgb_time_stamp);
                    
                }
        
            }while(cur_time_diff < prev_time_diff);
    
            prev_time_diff = 10000000.;
            cur_time_diff  = prev_time_diff;
    
            do{
        
                groundtruth_field_vector.clear();
    
                prev_groundtruth_depth_pos = groundtruth_stream_read_pos;
                groundtruth_stream_read_pos = groundtruth_stream.tellp();
                
                if(std::getline(groundtruth_stream, groundtruth_line)){
            
                    split_regex(groundtruth_field_vector, groundtruth_line, regex(" "));
                    groundtruth_time_stamp = atof(groundtruth_field_vector[0].c_str());
            
                    prev_time_diff = cur_time_diff;
                    cur_time_diff = abs(depth_time_stamp - groundtruth_time_stamp);
            
                }
        
            }while(cur_time_diff < prev_time_diff);
        
        
        }
        else if(bigger_timestamp == 2){
    
            auto prev_time_diff = 10000000.;
            auto cur_time_diff  = prev_time_diff;
            
            do{
        
                rgb_field_vector.clear();
        
                prev_rgb_pos = rgb_stream_read_pos;
                rgb_stream_read_pos = rgb_stream.tellp();
                
                if(std::getline(rgb_stream, rgb_line)){
            
                    split_regex(rgb_field_vector, rgb_line, regex(" "));
                    rgb_time_stamp = atof(rgb_field_vector[0].c_str());
            
                    prev_time_diff = cur_time_diff;
                    cur_time_diff = abs(groundtruth_time_stamp - rgb_time_stamp);
            
                }
        
            }while(cur_time_diff < prev_time_diff);
    
            prev_time_diff = 10000000.;
            cur_time_diff  = prev_time_diff;
            
            do{
        
                depth_field_vector.clear();
    
                prev_depth_pos = depth_stream_read_pos;
                depth_stream_read_pos = depth_stream.tellp();
                
                if(std::getline(depth_stream, depth_line)){
            
                    split_regex(depth_field_vector, depth_line, regex(" "));
                    depth_time_stamp = atof(depth_field_vector[0].c_str());
            
                    prev_time_diff = cur_time_diff;
                    cur_time_diff = abs(groundtruth_time_stamp - depth_time_stamp);
            
                }
        
            }while(cur_time_diff < prev_time_diff);
        
        
        }
    
        rgb_stream.seekp(prev_rgb_pos);
        depth_stream.seekp(prev_depth_pos);
        groundtruth_stream.seekp(prev_groundtruth_depth_pos);
        
    }
    
}
void cgmapping::eigen_pose_matrix4d_2_ros_geometry_msg_pose(Eigen::Matrix4d &eigen_pose,
                                                            geometry_msgs::Pose &geo_msg_pose) {
    
    geo_msg_pose.position.x = eigen_pose(0,3);
    geo_msg_pose.position.y = eigen_pose(1,3);
    geo_msg_pose.position.z = eigen_pose(2,3);
    
    tf2::Matrix3x3 tmp_matrix;
    
    tmp_matrix.setValue(eigen_pose(0,0), eigen_pose(0,1), eigen_pose(0,2),
                        eigen_pose(1,0), eigen_pose(1,1), eigen_pose(1,2),
                        eigen_pose(2,0), eigen_pose(2,1), eigen_pose(2,2));
    
    tf2::Quaternion tmp_quaternion;
    
    tmp_matrix.getRotation(tmp_quaternion);
    
    geo_msg_pose.orientation.x = tmp_quaternion.getX();
    geo_msg_pose.orientation.y = tmp_quaternion.getY();
    geo_msg_pose.orientation.z = tmp_quaternion.getZ();
    geo_msg_pose.orientation.w = tmp_quaternion.getW();
    
}
void cgmapping::poses_to_file(std::vector<double> &time_stamps,
                              std::vector<geometry_msgs::Pose> &poses,
                              std::string &file_name) {
    
    std::fstream file_stream(file_name, std::ios::in | std::ios::out | std::ios::trunc);
    
    if(file_stream.is_open()){
    
        for (int i = 0; i < time_stamps.size(); ++i) {
            
            file_stream.unsetf(std::ios::floatfield);
            file_stream.precision(16);
            
            file_stream << std::left << time_stamps[i] << " " << poses[i].position.x
                        << " " << poses[i].position.y << " " << poses[i].position.z
                        << " " << poses[i].orientation.x << " " << poses[i].orientation.y
                        << " " << poses[i].orientation.z << " "<< poses[i].orientation.w << std::endl;
            
        }
        
        file_stream.close();
        
    }
    
    else {
        
        std::cerr << "Not able to open file " << file_name << std::endl;
        
    }
    
}
void cgmapping::ros_geometry_msg_pose_2_eigen_pose_matrix4d(geometry_msgs::Pose &geo_msg_pose,
                                                            Eigen::Matrix4d &eigen_pose) {
    
    tf2::Matrix3x3 tmp_matrix(tf2::Quaternion(geo_msg_pose.orientation.x,
                                              geo_msg_pose.orientation.y,
                                              geo_msg_pose.orientation.z,
                                              geo_msg_pose.orientation.w));
    
    eigen_pose << tmp_matrix.getRow(0).getX(), tmp_matrix.getRow(0).getY(), tmp_matrix.getRow(0).getZ(), geo_msg_pose.position.x,
                  tmp_matrix.getRow(1).getX(), tmp_matrix.getRow(1).getY(), tmp_matrix.getRow(1).getZ(), geo_msg_pose.position.y,
                  tmp_matrix.getRow(2).getX(), tmp_matrix.getRow(2).getY(), tmp_matrix.getRow(2).getZ(), geo_msg_pose.position.z,
                  0,                           0,                           0,                           1;
    
}
void cgmapping::pose_line_2_eigen_pose_matrix4d(std::string &pose_string, Eigen::Matrix4d &eigen_pose) {
    
    std::vector<std::string> field_vector;
    
    geometry_msgs::Pose tmp_pose;
    
    boost::split_regex(field_vector, pose_string, boost::regex(" "));
    
    tmp_pose.position.x = atof(field_vector[1].c_str());
    tmp_pose.position.y = atof(field_vector[2].c_str());
    tmp_pose.position.z = atof(field_vector[3].c_str());
    
    tmp_pose.orientation.x = atof(field_vector[4].c_str());
    tmp_pose.orientation.y = atof(field_vector[5].c_str());
    tmp_pose.orientation.z = atof(field_vector[6].c_str());
    tmp_pose.orientation.w = atof(field_vector[7].c_str());
    
    ros_geometry_msg_pose_2_eigen_pose_matrix4d(tmp_pose, eigen_pose);
    
    
}

