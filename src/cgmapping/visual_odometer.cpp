//
// Created by spades on 15/03/18.
//

#include <cgmapping/visual_odometer.h>

template<typename T>
cgmapping::cuda::visual_odometer<T>::visual_odometer(const Mat &cur_img,
                                                     const Mat &cur_img_depth,
                                                     rgb_d_camera_model camera_model,
                                                     T eps_error,
                                                     uint k_max_iterations_for_optmization,
                                                     T k_degrees_of_freedom_t_student,
                                                     T k_epslon_accptance_stud_dev,
                                                     T *initial_position)
    : cur_img_(cgmapping::image_pyramid(cur_img.rows, cur_img.cols, cur_img.type())),
      cur_img_depth_(cgmapping::image_pyramid(cur_img_depth.rows, cur_img_depth.cols, cur_img_depth.type())),
      prev_img_(cgmapping::image_pyramid(cur_img.rows, cur_img.cols, cur_img.type())),
      prev_img_depth_(cgmapping::image_pyramid(cur_img_depth.rows, cur_img_depth.cols, cur_img_depth.type())),
      img_motion_estimator_(cgmapping::cuda::img_motion_estimator<T>((uint) cur_img.rows, (uint) cur_img.cols, k_degrees_of_freedom_t_student, k_epslon_accptance_stud_dev)){
    
    
    for (int j = 0; j < 4; ++j) {
        cudaCheckErrors(cudaStreamCreateWithFlags(&raw_strm_[j], cudaStreamNonBlocking), __FILE__, __FUNCTION__, __LINE__);
        strm_[j] = cv::cuda::StreamAccessor::wrapStream(raw_strm_[j]);
    }
    
    this->cur_img_._generate_pyramid(cur_img, strm_[0]);
    this->cur_img_depth_._generate_pyramid(cur_img_depth, strm_[1]);
    
    delta_inverse_transform_._setIdentity(&raw_strm_[0]);
    
    camera_model_ = camera_model;
    
    this->k_eps_error_ = eps_error;
    this->k_max_iterations_for_optmization_ = k_max_iterations_for_optmization;
    
    cudaError_t stat = cudaError_t::cudaErrorStartupFailure;
    
    if(initial_position != nullptr){
    
        stat = this->cur_transform_._uploadData(initial_position, 16, &raw_strm_[2]);
        cudaCheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
        
        if(stat == cudaSuccess){
            
            cgmapping::cuda::calculate_Dlog_map_se3(this->cur_linear_vel_,
                                                    this->cur_angular_vel_,
                                                    this->cur_transform_,
                                                    this->auxiliar_matrix_lie_algebra_1_,
                                                    strm_[2],
                                                    1.);
            
        }
        
        
    }
    
    if(initial_position == nullptr || stat != cudaSuccess ) {
        
        this->cur_transform_._setIdentity(&raw_strm_[1]);
        this->cur_linear_vel_._setZero(&raw_strm_[2]);
        this->cur_angular_vel_._setZero(&raw_strm_[3]);
        
    }
    
    this->cov_motion_prior_._allocateMatrixDataMemory();
    this->cov_motion_prior_._setDiagonalValue(0.0000000001, &raw_strm_[0]);
    
    this->motion_prior_._allocateMatrixDataMemory();
    this->motion_prior_._setZero(&raw_strm_[0]);
    
    cudaCheckErrors(cudaDeviceSynchronize(), __FILE__, __FUNCTION__, __LINE__);
    
}

template<typename T>
void cgmapping::cuda::visual_odometer<T>::_estimateCamera_motion(const Mat &cur_img,
                                                                 const Mat &cur_img_depth,
                                                                 cgmapping::image_size_t min_img_size,
                                                                 cgmapping::image_size_t max_img_size) {
    
    static culina_matrix<T, 3, 1> angular_vel;
    static culina_matrix<T, 3, 1> linear_vel;
    
    angular_vel._setZero(&raw_strm_[1]);
    linear_vel._setZero(&raw_strm_[2]);
    
//    swap(this->prev_img_, this->cur_img_);
//    swap(this->prev_img_depth_, this->cur_img_depth_);
//
//    std::string warped_jacob_file_name("/home/spades/kinetic_ws/src/cgmapping/warped_jacobian_matrix.m");
//    std::string full_jacob_file_name("/home/spades/kinetic_ws/src/cgmapping/full_jacobian_matrix.m");
//    std::string residual_file_name("/home/spades/kinetic_ws/src/cgmapping/residual_matrix.m");
//    std::string weight_matrix_file_name("/home/spades/kinetic_ws/src/cgmapping/weight_matrix.m");
//    std::string previous_point_cloud_file_name("/home/spades/kinetic_ws/src/cgmapping/pc_matrix.m");
//    std::string lower_bound_cov_file_name("/home/spades/kinetic_ws/src/cgmapping/lower_bound_cov_matrix.m");
    
    for (int i = min_img_size; i <= max_img_size ; i++) {
        
        auto img_size = static_cast<cgmapping::image_size_t>(i);
        
        this->cur_img_._getImageMat(img_size).copyTo(this->prev_img_._getImageMat(img_size), strm_[3]);
        this->cur_img_depth_._getImageMat(img_size).copyTo(this->prev_img_depth_._getImageMat(img_size), strm_[0]);
    
    }
    
    this->cur_img_._generate_pyramid(cur_img, strm_[3]);
    this->cur_img_depth_._generate_pyramid(cur_img_depth, strm_[0]);
    
    strm_[3].waitForCompletion();
    strm_[0].waitForCompletion();
    
    for (int i = min_img_size; i <= max_img_size ; i++) {
        
        auto img_size = static_cast<cgmapping::image_size_t>(i);
    
        this->img_motion_estimator_._estimateMotion_from_images(prev_img_,
                                                                prev_img_depth_,
                                                                cur_img_,
                                                                img_size,
                                                                linear_vel,
                                                                angular_vel,
                                                                //this->motion_prior_, //coment these lines in order to
                                                                //this->cov_motion_prior_, //no use motion prior
                                                                this->camera_model_,
                                                                this->k_eps_error_,
                                                                this->k_max_iterations_for_optmization_);
        
        linear_vel._loadData(this->img_motion_estimator_._getLinear_vel(), raw_strm_[0]);
        angular_vel._loadData(this->img_motion_estimator_._getAngular_vel(), raw_strm_[1]);
        
        strm_[0].waitForCompletion();
        strm_[1].waitForCompletion();
        
    }
    
    auto img_size = static_cast<cgmapping::image_size_t>(min_img_size);
    
//    cuLiNA::culina_download_matrix_file(this->_getResiduals_at_img_size(img_size), residual_file_name);
//    cuLiNA::culina_download_matrix_file(this->img_motion_estimator_._getCorrect_warp_jacobian(img_size), warped_jacob_file_name);
//    cuLiNA::culina_download_matrix_file(this->_getFull_jacobian_at_img_size(img_size), full_jacob_file_name);
//    cuLiNA::culina_download_matrix_file(this->_getWeight_at_img_size(img_size), weight_matrix_file_name);
//    cuLiNA::culina_download_matrix_file(this->_getPrev_img_point_cloud_at_img_size(img_size), previous_point_cloud_file_name);
//    cuLiNA::culina_download_matrix_file(const_cast<culina_matrix<T,6,6> &>(this->_getLower_bound_delta_twist_cov_matrix()), lower_bound_cov_file_name);
    
    this->motion_prior_._loadData(img_motion_estimator_._getDelta_twist_vels(), raw_strm_[1]);
    this->cov_motion_prior_._loadData(img_motion_estimator_._getLower_bound_twist_cov_matrix(), raw_strm_[1]);
    this->delta_inverse_transform_._loadData(const_cast<culina_matrix4d &>(this->img_motion_estimator_._getHomogenic_transformation()), raw_strm_[2]);
    
    cgmapping::cuda::compose_Dpose_3D(this->cur_transform_,
                                      this->delta_inverse_transform_,
                                      this->cur_transform_,
                                      strm_[2]);
    
    cgmapping::cuda::calculate_Dlog_map_se3(this->cur_linear_vel_,
                                            this->cur_angular_vel_,
                                            this->cur_transform_,
                                            this->auxiliar_matrix_lie_algebra_1_,
                                            strm_[2], 1);
    
    strm_[1].waitForCompletion();
    strm_[2].waitForCompletion();
    
//    this->motion_prior_._printMatrix();
//
//    std::cout << std::endl;
//
//    this->cov_motion_prior_._printMatrix();
    
}

template<typename T>
T cgmapping::cuda::visual_odometer<T>::_getK_eps_error() const {
    return k_eps_error_;
}
template<typename T>
uint cgmapping::cuda::visual_odometer<T>::_getK_max_iterations_for_optmization() const {
    return k_max_iterations_for_optmization_;
}
template<typename T>
uint cgmapping::cuda::visual_odometer<T>::_getIterations_at_img_resolution(cgmapping::image_size_t img_size) const {
    return this->img_motion_estimator_._getCorrect_iterations_at_img_resolution(img_size);
}
template<typename T>
cgmapping::image_size_t cgmapping::cuda::visual_odometer<T>::_getMin_resolution() const {
    return min_resolution_;
}
template<typename T>
cgmapping::image_size_t cgmapping::cuda::visual_odometer<T>::_getMax_resolution() const {
    return max_resolution_;
}

template<typename T>
const cuLiNA::culina_matrix<T, 3, 1> &cgmapping::cuda::visual_odometer<T>::_getLinear_vel() const {
    return this->img_motion_estimator_._getLinear_vel();
}
template<typename T>
const cuLiNA::culina_matrix<T, 3, 1> &cgmapping::cuda::visual_odometer<T>::_getAngular_vel() const {
    return this->img_motion_estimator_._getAngular_vel();
}
template<typename T>
const cuLiNA::culina_matrix<T, 3, 1> &cgmapping::cuda::visual_odometer<T>::_getDelta_linear_vel() const {
    return this->img_motion_estimator_._getDelta_linear_vel();
}
template<typename T>
const cuLiNA::culina_matrix<T, 3, 1> &cgmapping::cuda::visual_odometer<T>::_getDelta_angular_vel() const {
    return this->img_motion_estimator_._getDelta_angular_vel();
}
template<typename T>
const culina_matrix<T, 4, 4> &cgmapping::cuda::visual_odometer<T>::_getCur_estimated_pose() const {
    return cur_transform_;
}
template<typename T>
cgmapping::image_pyramid &cgmapping::cuda::visual_odometer<T>::_getCur_img_warped() const {
    return this->img_motion_estimator_._getCur_img_warped();
}
template<typename T>
cgmapping::image_pyramid &cgmapping::cuda::visual_odometer<T>::_getCur_warped_filter() const {
    return this->img_motion_estimator_._getCur_warped_filter();
}
template<typename T>
cgmapping::image_pyramid &cgmapping::cuda::visual_odometer<T>::_getCur_img_warped_x_derivative() const {
    return this->img_motion_estimator_._getCur_img_warped_x_derivative();
}
template<typename T>
cgmapping::image_pyramid &cgmapping::cuda::visual_odometer<T>::_getCur_img_warped_y_derivative() const {
    return this->img_motion_estimator_._getCur_img_warped_y_derivative();
}
template<typename T>
culina_tm<T> &cgmapping::cuda::visual_odometer<T>::_getResiduals_at_img_size(cgmapping::image_size_t img_size) const {
    return this->img_motion_estimator_._getCorrect_residuals(img_size);
}
template<typename T>
culina_tm<T> &cgmapping::cuda::visual_odometer<T>::_getFull_jacobian_at_img_size(cgmapping::image_size_t img_size) const {
    return this->img_motion_estimator_._getCorrect_full_jacobian(img_size);
}
template<typename T>
culina_tm<T> &cgmapping::cuda::visual_odometer<T>::_getPrev_img_point_cloud_at_img_size(cgmapping::image_size_t img_size) const {
    return this->img_motion_estimator_._getCorrect_prev_img_point_cloud(img_size);
}
template<typename T>
culina_tm<T> &cgmapping::cuda::visual_odometer<T>::_getWeight_at_img_size(cgmapping::image_size_t img_size) const {
    return this->img_motion_estimator_._getCorrect_weight(img_size);
}
template<typename T>
const cuLiNA::culina_matrix<T,
                            6,
                            6> &cgmapping::cuda::visual_odometer<T>::_getLower_bound_delta_twist_cov_matrix() const {
    return this->img_motion_estimator_._getLower_bound_twist_cov_matrix();
}
template<typename T>
const culina_matrix<T, 4, 4> &cgmapping::cuda::visual_odometer<T>::_getCur_delta_transform() const {
    return this->delta_inverse_transform_;
}

template class cgmapping::cuda::visual_odometer<double>;