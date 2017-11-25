//
// Created by spades on 03/04/18.
//


#include <cgmapping/img_motion_estimator.h>

template<typename T>
cgmapping::cuda::img_motion_estimator<T>::img_motion_estimator(uint img_rows,
                                                               uint img_cols,
                                                               T k_degrees_of_freedom_t_student,
                                                               T k_epslon_accptance_std_dev) {
    
    /////////////////////////////////////CUDA STREAMS & EVENTS//////////////////////////////////////////////////
    
    cudaError_t cuda_stat;
    
    for (int i = 0; i < 7; ++i) {
        cuda_stat = cudaStreamCreateWithFlags(&raw_strm_[i], cudaStreamNonBlocking);
        cudaCheckErrors(cuda_stat, __FILE__, __FUNCTION__, __LINE__);
        cuda_stat = cudaEventCreateWithFlags(&raw_evnt_[i], cudaEventDisableTiming);
        cudaCheckErrors(cuda_stat, __FILE__, __FUNCTION__, __LINE__);
        strm_[i] = cv::cuda::StreamAccessor::wrapStream(raw_strm_[i]);
        evnt_[i] = cv::cuda::EventAccessor::wrapEvent(raw_evnt_[i]);
    }
    
    /////////////////////////////////////IMG PYRAMIDS//////////////////////////////////////////////////////////

    cur_img_warped_ = cgmapping::image_pyramid(img_rows, img_cols, CV_8UC1);
    cur_warped_filter_ = cgmapping::image_pyramid(img_rows, img_cols, CV_8UC1);

    cur_img_warped_x_derivative_ = cgmapping::image_pyramid(img_rows, img_cols, CV_32F);
    cur_img_warped_y_derivative_ = cgmapping::image_pyramid(img_rows, img_cols, CV_32F);;

    /////////////////////////////////////SOBEL FITLERS///////////////////////////////////////////////////////////

    cv::Mat x_derivative_kernel(1,3,CV_32FC1);
    cv::Mat y_derivative_kernel(3,1,CV_32FC1);
    
    cv::Mat x_inert_kernel(1,3,CV_32FC1);
    cv::Mat y_inert_kernel(3,1,CV_32FC1);
    
    for (int i = 0; i < x_inert_kernel.cols; ++i)
        x_inert_kernel.at<float>(0,i) = 0;
    
    for (int i = 0; i < y_inert_kernel.rows; ++i)
        y_inert_kernel.at<float>(i,0) = 0;
    
    
    x_inert_kernel.at<float>(0,1) = 1;
    y_inert_kernel.at<float>(1,0) = 1;
    
    x_derivative_kernel.at<float>(0,0) = (float)-0.5;
    y_derivative_kernel.at<float>(0,0) = (float)-0.5;
    x_derivative_kernel.at<float>(0,1) = 0;
    y_derivative_kernel.at<float>(1,0) = 0;
    x_derivative_kernel.at<float>(0,2) = (float)0.5;
    y_derivative_kernel.at<float>(2,0) = (float)0.5;
    
    filter_x_ = cv::cuda::createSeparableLinearFilter(CV_8UC1, CV_32FC1, x_derivative_kernel, y_inert_kernel);
    filter_y_ = cv::cuda::createSeparableLinearFilter(CV_8UC1, CV_32FC1, x_inert_kernel, y_derivative_kernel);
    
//    filter_x_ = cv::cuda::createDerivFilter(CV_8UC1, CV_16S, 1, 0, 3, true, 0.5, BORDER_DEFAULT);
//    filter_y_ = cv::cuda::createDerivFilter(CV_8UC1, CV_16S, 0, 1, 3, true, 0.5, BORDER_DEFAULT);

    /////////////////////////////////////WARP JACOBIANS//////////////////////////////////////////////////////////

    auto n_rows = img_rows*img_cols*2;

    this->warp_jacobian_original_._setRows(n_rows);
    this->warp_jacobian_original_._setColumns(6);
    this->warp_jacobian_original_._allocateMatrixDataMemory();

    this->warp_jacobian_half_._setRows(n_rows/4);
    this->warp_jacobian_half_._setColumns(6);
    this->warp_jacobian_half_._allocateMatrixDataMemory();

    this->warp_jacobian_quarter_._setRows(n_rows/16);
    this->warp_jacobian_quarter_._setColumns(6);
    this->warp_jacobian_quarter_._allocateMatrixDataMemory();

    this->warp_jacobian_oct_._setRows(n_rows/64);
    this->warp_jacobian_oct_._setColumns(6);
    this->warp_jacobian_oct_._allocateMatrixDataMemory();

    this->warp_jacobian_hexadec_._setRows(n_rows/256);
    this->warp_jacobian_hexadec_._setColumns(6);
    this->warp_jacobian_hexadec_._allocateMatrixDataMemory();

    /////////////////////////////////////FULL JACOBIANS//////////////////////////////////////////////////////////

    n_rows /= 2;

    this->full_jacobian_original_._setRows(n_rows);
    this->full_jacobian_original_._setColumns(6);
    this->full_jacobian_original_._allocateMatrixDataMemory();

    this->full_jacobian_half_._setRows(n_rows/4);
    this->full_jacobian_half_._setColumns(6);
    this->full_jacobian_half_._allocateMatrixDataMemory();

    this->full_jacobian_quarter_._setRows(n_rows/16);
    this->full_jacobian_quarter_._setColumns(6);
    this->full_jacobian_quarter_._allocateMatrixDataMemory();

    this->full_jacobian_oct_._setRows(n_rows/64);
    this->full_jacobian_oct_._setColumns(6);
    this->full_jacobian_oct_._allocateMatrixDataMemory();

    this->full_jacobian_hexadec_._setRows(n_rows/256);
    this->full_jacobian_hexadec_._setColumns(6);
    this->full_jacobian_hexadec_._allocateMatrixDataMemory();

    /////////////////////////////////////IMG POINT CLOUDS//////////////////////////////////////////////////////////

    this->prev_img_point_cloud_original_._setRows(n_rows);
    this->prev_img_point_cloud_original_._setColumns(3);
    this->prev_img_point_cloud_original_._allocateMatrixDataMemory();

    this->prev_img_point_cloud_half_._setRows(n_rows/4);
    this->prev_img_point_cloud_half_._setColumns(3);
    this->prev_img_point_cloud_half_._allocateMatrixDataMemory();

    this->prev_img_point_cloud_quarter_._setRows(n_rows/16);
    this->prev_img_point_cloud_quarter_._setColumns(3);
    this->prev_img_point_cloud_quarter_._allocateMatrixDataMemory();

    this->prev_img_point_cloud_oct_._setRows(n_rows/64);
    this->prev_img_point_cloud_oct_._setColumns(3);
    this->prev_img_point_cloud_oct_._allocateMatrixDataMemory();

    this->prev_img_point_cloud_hexadec_._setRows(n_rows/256);
    this->prev_img_point_cloud_hexadec_._setColumns(3);
    this->prev_img_point_cloud_hexadec_._allocateMatrixDataMemory();

    /////////////////////////////////////WEIGHT MATRICES//////////////////////////////////////////////////////////

    this->weight_original_._setMatrix_type(cuLiNA::DIAGONAL);
    this->weight_original_._setRows(n_rows);
    this->weight_original_._allocateMatrixDataMemory();

    this->weight_half_._setMatrix_type(cuLiNA::DIAGONAL);
    this->weight_half_._setRows(n_rows/4);
    this->weight_half_._allocateMatrixDataMemory();

    this->weight_quarter_._setMatrix_type(cuLiNA::DIAGONAL);
    this->weight_quarter_._setRows(n_rows/16);
    this->weight_quarter_._allocateMatrixDataMemory();

    this->weight_oct_._setMatrix_type(cuLiNA::DIAGONAL);
    this->weight_oct_._setRows(n_rows/64);
    this->weight_oct_._allocateMatrixDataMemory();

    this->weight_hexadec_._setMatrix_type(cuLiNA::DIAGONAL);
    this->weight_hexadec_._setRows(n_rows/256);
    this->weight_hexadec_._allocateMatrixDataMemory();

    /////////////////////////////////////RESIDUAL MATRICES//////////////////////////////////////////////////////////

    this->residual_original_._setRows(img_rows);
    this->residual_original_._setColumns(img_cols);
    this->residual_original_._allocateMatrixDataMemory();

    this->residual_half_._setRows(img_rows/2);
    this->residual_half_._setColumns(img_cols/2);
    this->residual_half_._allocateMatrixDataMemory();

    this->residual_quarter_._setRows(img_rows/4);
    this->residual_quarter_._setColumns(img_cols/4);
    this->residual_quarter_._allocateMatrixDataMemory();

    this->residual_oct_._setRows(img_rows/8);
    this->residual_oct_._setColumns(img_cols/8);
    this->residual_oct_._allocateMatrixDataMemory();

    this->residual_hexadec_._setRows(img_rows/16);
    this->residual_hexadec_._setColumns(img_cols/16);
    this->residual_hexadec_._allocateMatrixDataMemory();

    /////////////////////////////////////AUXLIAR MATRICES//////////////////////////////////////////////////////////

    this->auxiliar_original_._setRows(1);
    this->auxiliar_original_._setColumns(n_rows);
    this->auxiliar_original_._allocateMatrixDataMemory();

    this->auxiliar_half_._setRows(1);
    this->auxiliar_half_._setColumns(n_rows/4);
    this->auxiliar_half_._allocateMatrixDataMemory();

    this->auxiliar_quarter_._setRows(1);
    this->auxiliar_quarter_._setColumns(n_rows/16);
    this->auxiliar_quarter_._allocateMatrixDataMemory();

    this->auxiliar_oct_._setRows(1);
    this->auxiliar_oct_._setColumns(n_rows/64);
    this->auxiliar_oct_._allocateMatrixDataMemory();

    this->auxiliar_hexadec_._setRows(1);
    this->auxiliar_hexadec_._setColumns(n_rows/256);
    this->auxiliar_hexadec_._allocateMatrixDataMemory();

    this->auxiliar_matrix_linear_solver_1_._setColumns(6);
    this->auxiliar_matrix_linear_solver_1_._setRows(this->full_jacobian_original_._getRows());
    this->auxiliar_matrix_linear_solver_1_._allocateMatrixDataMemory();
    
    this->auxiliar_matrix_linear_solver_2_._allocateMatrixDataMemory();
    
    this->lower_bound_twist_cov_matrix_._allocateMatrixDataMemory();
    
    this->auxiliar_matrix_lie_algebra_1_._allocateMatrixDataMemory();
    
    this->auxiliar_matrix_lie_algebra_2_._allocateMatrixDataMemory();
    
    /////////////////////////////////////CONSTANTS///////////////////////////////////////////////////////////////////

    this->k_degrees_of_freedom_t_student_ = k_degrees_of_freedom_t_student;
    this->k_epslon_accptance_std_dev_ = k_epslon_accptance_std_dev;
    
    /////////////////////////////////////HOMOGENIC TRANSFORMATION///////////////////////////////////////////////////
    
    this->homogenic_transformation_._allocateMatrixDataMemory();
    this->delta_homogenic_transformation_._allocateMatrixDataMemory();
    
    this->homogenic_transformation_._setIdentity(&this->raw_strm_[0]);
    this->delta_homogenic_transformation_._setIdentity(&this->raw_strm_[1]);
    
    this->tmp_linear_vel_._allocateMatrixDataMemory();
    this->tmp_linear_vel_._setZero(&this->raw_strm_[2]);
    this->tmp_angular_vel_._allocateMatrixDataMemory();
    this->tmp_angular_vel_._setZero(&this->raw_strm_[3]);
    
    this->linear_vel_._allocateMatrixDataMemory();
    this->linear_vel_._setZero(&this->raw_strm_[4]);
    this->angular_vel_._allocateMatrixDataMemory();
    this->angular_vel_._setZero(&this->raw_strm_[5]);
    
    this->twist_vector_._allocateMatrixDataMemory();
    this->linear_vel_._setZero(&this->raw_strm_[6]);
    
    cudaCheckErrors(cudaDeviceSynchronize(), __FILE__, __FUNCTION__, __LINE__);
    
}

template<typename T>
void cgmapping::cuda::img_motion_estimator<T>::_estimateMotion_from_images(cgmapping::image_pyramid &img1_pyr,
                                                                           cgmapping::image_pyramid &depth1_pyr,
                                                                           cgmapping::image_pyramid &img2_pyr,
                                                                           cgmapping::image_size_t img_size,
                                                                           culina_matrix<T, 3, 1> &init_linear_vel,
                                                                           culina_matrix<T, 3, 1> &init_angular_vel,
                                                                           cgmapping::rgb_d_camera_model &camera_model,
                                                                           T k_acceptance_error,
                                                                           uint k_max_interations) {
    
    T weighted_error = 1000;
    T last_weighted_error = 100000;
    
    T k_lambda_factor = 0;
    
    this->initializeTwist_variables(init_linear_vel, init_angular_vel);
    
    this->getCorrect_iterations_at_img_resolution(img_size) = 0;
    
    uint it_num = 0;
    
    auto delta_weighted_error = (last_weighted_error - weighted_error);
    //auto last_delta_weighted_error = 1000000.;
    
    bool decreased_error = true;
    
    while(abs(delta_weighted_error) > k_acceptance_error && decreased_error  && k_max_interations > it_num){
        
        this->calculateWarped_img(img2_pyr._getImageMat(img_size),
                                  depth1_pyr._getImageMat(img_size),
                                  camera_model,
                                  img_size);
        
        if(!it_num)
            this->calculateWarped_img_derivatives(img_size);
        
        this->calculateResidual(img1_pyr._getImageMat(img_size), img_size);
        
        auto number_of_valid_data = this->countValid_data(img_size);
        
        //std::cout << "number of valid data = " << number_of_valid_data << std::endl;
        
        auto std_dev = this->calculateStd_dev_t_student(img_size, number_of_valid_data);
    
        //std::cout << "std_dev = " << std_dev << std::endl;
        
        this->updateWeighted_error(img_size, std_dev, number_of_valid_data);
        
        if(!it_num)
            this->calculateWarped_jacobian(camera_model, img_size);
        
        if(it_num)
            this->calculateWarped_img_derivatives(img_size);
        
        this->calculateJacobian(img_size);
        
        cudaCheckErrors(cudaDeviceSynchronize(), __FILE__, __FUNCTION__, __LINE__);
        //STREAM barrier===================================================================================================
        /*
         *
         *  |
         *  |
         *  |
         *  |    From here on is all about deciding how to optmize stuff
         *  |
         *  |
         * \ /
         *  v
         *
         *
         * */
        
        auto tmp = last_weighted_error;
        last_weighted_error = weighted_error;
        weighted_error = this->sqrd_weighted_error_(0,0);
        delta_weighted_error = last_weighted_error - weighted_error;
        decreased_error = (last_weighted_error > weighted_error);
    
//        std::cout << "==================================================================" << std::endl;
//        std::cout << "img_size = " << img_size << std::endl;
//        std::cout << "last_weighted_error = " << last_weighted_error << std::endl;
//        std::cout << "weighted_error = " << weighted_error << std::endl;
//        std::cout << "last_delta_weighted_error = " << last_delta_weighted_error << std::endl;
//        std::cout << "delta_weighted_error = " << delta_weighted_error << std::endl;
//        std::cout << "std_dev" << std_dev << std::endl;

//        if(!decreased_error){
//
//            if(k_lambda_factor!=0 && abs(k_lambda_factor) < 0.0001) break;
//
//            weighted_error = last_weighted_error;
//            last_weighted_error = tmp;
//            this->tmp_homogenic_transformation_ = static_cast<cuLiNA::culina_tm<T> &>(homogenic_transformation_);
//
//            evnt_[0].record();
//        }
//
//        k_lambda_factor = (decreased_error?k_lambda_factor:(k_lambda_factor==0?(-1):k_lambda_factor/2));
//
//        std::cout << "k_lambda_factor = " << k_lambda_factor << std::endl;
        
        if(decreased_error){
            
            this->incorporate_estimation();
            
            this->optimizeDelta_twist(img_size, k_lambda_factor);
    
            this->updateTwist_variables(decreased_error);
            
        }
    
        it_num++;
        
    }
    
    this->getCorrect_iterations_at_img_resolution(img_size) = it_num;
    
    cudaCheckErrors(cudaDeviceSynchronize(), __FILE__, __FUNCTION__, __LINE__); //<-- This guaratees that nothing continues to run
                                                                                //after the end of this call
}

template<typename T>
void cgmapping::cuda::img_motion_estimator<T>::_estimateMotion_from_images(cgmapping::image_pyramid &img1_pyr,
                                                                           cgmapping::image_pyramid &depth1_pyr,
                                                                           cgmapping::image_pyramid &img2_pyr,
                                                                           cgmapping::image_size_t img_size,
                                                                           culina_matrix<T, 3, 1> &init_linear_vel,
                                                                           culina_matrix<T, 3, 1> &init_angular_vel,
                                                                           culina_matrix<T, 6, 1> &motion_prior,
                                                                           culina_matrix<T, 6, 6> &cov_motion_prior,
                                                                           cgmapping::rgb_d_camera_model &camera_model,
                                                                           T k_acceptance_error,
                                                                           uint k_max_interations) {
    
    T weighted_error = 1000;
    T last_weighted_error = 100000;
    
    T k_lambda_factor = 0.16;
    
    this->initializeTwist_variables(init_linear_vel, init_angular_vel);
    
    this->getCorrect_iterations_at_img_resolution(img_size) = 0;
    
    uint it_num = 0;
    
    auto delta_weighted_error = (last_weighted_error - weighted_error);
    //auto last_delta_weighted_error = 1000000.;
    
    bool decreased_error = true;
    
    while(abs(delta_weighted_error) > k_acceptance_error && decreased_error  && k_max_interations > it_num){
        
        this->calculateWarped_img(img2_pyr._getImageMat(img_size),
                                  depth1_pyr._getImageMat(img_size),
                                  camera_model,
                                  img_size);
        
        if(!it_num)
            this->calculateWarped_img_derivatives(img_size);
        
        this->calculateResidual(img1_pyr._getImageMat(img_size), img_size);
        
        auto number_of_valid_data = this->countValid_data(img_size);
        
        //std::cout << "number of valid data = " << number_of_valid_data << std::endl;
        
        auto std_dev = this->calculateStd_dev_t_student(img_size, number_of_valid_data);
        
        //std::cout << "std_dev = " << std_dev << std::endl;
        
        this->updateWeighted_error(img_size, std_dev, number_of_valid_data);
        
        if(!it_num)
            this->calculateWarped_jacobian(camera_model, img_size);
        
        if(it_num)
            this->calculateWarped_img_derivatives(img_size);
        
        this->calculateJacobian(img_size);
        
        cudaCheckErrors(cudaDeviceSynchronize(), __FILE__, __FUNCTION__, __LINE__);
        //STREAM barrier===================================================================================================
        /*
         *
         *  |
         *  |
         *  |
         *  |    From here on is all about deciding how to optmize stuff
         *  |
         *  |
         * \ /
         *  v
         *
         *
         * */
        
        auto tmp = last_weighted_error;
        last_weighted_error = weighted_error;
        weighted_error = this->sqrd_weighted_error_(0,0);
        delta_weighted_error = last_weighted_error - weighted_error;
        decreased_error = (last_weighted_error > weighted_error);
        
        
        if(decreased_error){
            
            this->incorporate_estimation();
            
            this->optimizeDelta_twist(img_size, k_lambda_factor, motion_prior, cov_motion_prior);
            
            this->updateTwist_variables(decreased_error);
            
        }
        
        it_num++;
        
    }
    
    this->getCorrect_iterations_at_img_resolution(img_size) = it_num;
    
    cudaCheckErrors(cudaDeviceSynchronize(), __FILE__, __FUNCTION__, __LINE__); //<-- This guaratees that nothing continues to run
    //after the end of this call
    
}

template<typename T>
cuLiNA::culina_tm<T> &cgmapping::cuda::img_motion_estimator<T>::getCorrect_residuals(cgmapping::image_size_t img_size) {
    switch(img_size){
        
        case image_size_t::ORIGINAL_SIZE: return this->residual_original_;
        case image_size_t::HALF_SIZE: return this->residual_half_;
        case image_size_t::QUARTER_SIZE: return this->residual_quarter_;
        case image_size_t::OCT_SIZE: return this->residual_oct_;
        default: return this->residual_hexadec_;
        
    }
}

template<typename T>
cuLiNA::culina_tm<T> &cgmapping::cuda::img_motion_estimator<T>::getCorrect_warp_jacobian(cgmapping::image_size_t img_size) {
    
    switch(img_size){
    
        case image_size_t::ORIGINAL_SIZE: return this->warp_jacobian_original_;
        case image_size_t::HALF_SIZE: return this->warp_jacobian_half_;
        case image_size_t::QUARTER_SIZE: return this->warp_jacobian_quarter_;
        case image_size_t::OCT_SIZE: return this->warp_jacobian_oct_;
        default: return this->warp_jacobian_hexadec_;
        
    }
    
}

template<typename T>
cuLiNA::culina_tm<T> &cgmapping::cuda::img_motion_estimator<T>::getCorrect_full_jacobian(cgmapping::image_size_t img_size) {
    
    switch(img_size){
        
        case image_size_t::ORIGINAL_SIZE: return this->full_jacobian_original_;
        case image_size_t::HALF_SIZE: return this->full_jacobian_half_;
        case image_size_t::QUARTER_SIZE: return this->full_jacobian_quarter_;
        case image_size_t::OCT_SIZE: return this->full_jacobian_oct_;
        default: return this->full_jacobian_hexadec_;
        
    }
    
}

template<typename T>
cuLiNA::culina_tm<T> &cgmapping::cuda::img_motion_estimator<T>::getCorrect_prev_img_point_cloud(cgmapping::image_size_t img_size) {
    
    switch(img_size){
        
        case image_size_t::ORIGINAL_SIZE: return this->prev_img_point_cloud_original_;
        case image_size_t::HALF_SIZE: return this->prev_img_point_cloud_half_;
        case image_size_t::QUARTER_SIZE: return this->prev_img_point_cloud_quarter_;
        case image_size_t::OCT_SIZE: return this->prev_img_point_cloud_oct_;
        default: return this->prev_img_point_cloud_hexadec_;
        
    }

}

template<typename T>
cuLiNA::culina_tm<T> &cgmapping::cuda::img_motion_estimator<T>::getCorrect_weight(cgmapping::image_size_t img_size) {
    
    switch(img_size){
        
        case image_size_t::ORIGINAL_SIZE: return this->weight_original_;
        case image_size_t::HALF_SIZE: return this->weight_half_;
        case image_size_t::QUARTER_SIZE: return this->weight_quarter_;
        case image_size_t::OCT_SIZE: return this->weight_oct_;
        default: return this->weight_hexadec_;
        
    }
    
}

template<typename T>
cuLiNA::culina_tm<T> &cgmapping::cuda::img_motion_estimator<T>::getCorrect_auxiliar(cgmapping::image_size_t img_size) {
    switch(img_size){
        
        case image_size_t::ORIGINAL_SIZE: return this->auxiliar_original_;
        case image_size_t::HALF_SIZE: return this->auxiliar_half_;
        case image_size_t::QUARTER_SIZE: return this->auxiliar_quarter_;
        case image_size_t::OCT_SIZE: return this->auxiliar_oct_;
        default: return this->auxiliar_hexadec_;
        
    }
}

template<typename T>
uint &cgmapping::cuda::img_motion_estimator<T>::getCorrect_iterations_at_img_resolution(cgmapping::image_size_t img_size) {
    switch(img_size){
        
        case image_size_t::ORIGINAL_SIZE: return this->iterations_at_highest_resolution_;
        case image_size_t::HALF_SIZE: return this->iterations_at_half_resolution_;
        case image_size_t::QUARTER_SIZE: return this->iterations_at_quarter_resolution_;
        case image_size_t::OCT_SIZE: return this->iterations_at_oct_resolution_;
        default: return this->iterations_at_hexadec_resolution_;
        
    }
}

template<typename T>
void cgmapping::cuda::img_motion_estimator<T>::initializeTwist_variables(culina_matrix<T, 3, 1> &init_linear_vel,
                                                                         culina_matrix<T, 3, 1> &init_angular_vel) {
    
    strm_[0].waitEvent(evnt_[0]);
    strm_[1].waitEvent(evnt_[0]);
    
    this->linear_vel_._loadData(init_linear_vel, raw_strm_[1]);
    this->angular_vel_._loadData(init_angular_vel, raw_strm_[1]);
    
    cgmapping::cuda::calculate_Dexp_map_se3(this->linear_vel_,
                                            this->angular_vel_,
                                            this->tmp_homogenic_transformation_,
                                            this->auxiliar_matrix_lie_algebra_1_,
                                            this->auxiliar_matrix_lie_algebra_2_,
                                            strm_[1],
                                            strm_[1]);
    
    strm_[1].waitForCompletion();
    
    this->homogenic_transformation_._loadData(tmp_homogenic_transformation_, raw_strm_[1]);
    
    evnt_[0].record(strm_[1]);
    
}

template<typename T>
void cgmapping::cuda::img_motion_estimator<T>::calculateWarped_img(cv::cuda::GpuMat &cur_img_original,
                                                                   cv::cuda::GpuMat &prev_depth_img,
                                                                   cgmapping::rgb_d_camera_model &camera_model,
                                                                   cgmapping::image_size_t img_size) {

    strm_[0].waitEvent(evnt_[0]);
    
    cgmapping::cuda::calculate_Dimage_warped(cur_img_original,
                                             this->cur_img_warped_._getImageMat(img_size),
                                             this->cur_warped_filter_._getImageMat(img_size),
                                             prev_depth_img,
                                             this->getCorrect_prev_img_point_cloud(img_size),
                                             this->tmp_homogenic_transformation_,
                                             camera_model,
                                             strm_[0]);
    
    evnt_[0].record(strm_[0]);
    
}

template<typename T>
void cgmapping::cuda::img_motion_estimator<T>::calculateResidual(cv::cuda::GpuMat &img1,
                                                                 cgmapping::image_size_t img_size) {
    
    strm_[1].waitEvent(evnt_[0]);
    
    cgmapping::cuda::calculate_Dimage_residual(img1,
                                               this->cur_img_warped_._getImageMat(img_size),
                                               this->cur_warped_filter_._getImageMat(img_size),
                                               this->getCorrect_residuals(img_size),
                                               strm_[1]);
    
    evnt_[1].record(strm_[1]);
    
}

template<typename T>
void cgmapping::cuda::img_motion_estimator<T>::calculateWarped_img_derivatives(cgmapping::image_size_t img_size) {
    
    strm_[2].waitEvent(evnt_[0]);
    strm_[3].waitEvent(evnt_[0]);
    
    filter_x_->apply(this->cur_img_warped_._getImageMat(img_size),
                     this->cur_img_warped_x_derivative_._getImageMat(img_size),
                     strm_[2]);
    
    filter_y_->apply(this->cur_img_warped_._getImageMat(img_size),
                     this->cur_img_warped_y_derivative_._getImageMat(img_size),
                     strm_[2]);
    
    evnt_[2].record(strm_[2]);
    
}

template<typename T>
void cgmapping::cuda::img_motion_estimator<T>::calculateWarped_jacobian(cgmapping::rgb_d_camera_model &camera_model,
                                                                        cgmapping::image_size_t img_size) {
    
    strm_[2].waitEvent(evnt_[0]);
    
    cgmapping::cuda::calculate_Dwarp_jacobian(this->getCorrect_warp_jacobian(img_size),
                                              this->getCorrect_prev_img_point_cloud(img_size),
                                              camera_model,
                                              strm_[2]);
    
    evnt_[4].record(strm_[2]);
    
}

template<typename T>
void cgmapping::cuda::img_motion_estimator<T>::calculateJacobian(cgmapping::image_size_t img_size) {
    
    strm_[2].waitEvent(evnt_[4]);
    
    cgmapping::cuda::calculate_Dfull_jacobian(this->cur_img_warped_x_derivative_._getImageMat(img_size),
                                              this->cur_img_warped_y_derivative_._getImageMat(img_size),
                                              this->getCorrect_warp_jacobian(img_size),
                                              this->getCorrect_full_jacobian(img_size),
                                              strm_[2]);
    
    evnt_[4].record(strm_[2]);
    
}

template<typename T>
int cgmapping::cuda::img_motion_estimator<T>::countValid_data(cgmapping::image_size_t img_size) {
    
    strm_[6].waitEvent(evnt_[1]);
    
    int number_of_valid_data = cgmapping::cuda::calculate_Dnumber_of_valid_data(this->getCorrect_residuals(img_size),
                                                                                strm_[6]);
    
    evnt_[6].record(strm_[6]);
    
    return number_of_valid_data;
    
}

template<typename T>
double cgmapping::cuda::img_motion_estimator<T>::calculateStd_dev_t_student(cgmapping::image_size_t img_size,
                                                                            int number_of_valid_data) {
    
    strm_[6].waitEvent(evnt_[1]);
    
    double std_dev = cgmapping::cuda::calculate_Dstandard_deviation_t_student(this->getCorrect_residuals(img_size),
                                                                              this->k_degrees_of_freedom_t_student_,
                                                                              1,
                                                                              number_of_valid_data,
                                                                              this->k_epslon_accptance_std_dev_,
                                                                              strm_[6]);
    
    evnt_[6].record(strm_[6]);
    
    return std_dev;
    
}

template<typename T>
void cgmapping::cuda::img_motion_estimator<T>::updateWeighted_error(cgmapping::image_size_t img_size,
                                                                    double std_dev,
                                                                    int number_of_valid_data) {
    
    strm_[6].waitEvent(evnt_[1]);
    
    auto variance = std_dev * std_dev;
    
    cgmapping::cuda::calculate_Dweight_matrix(this->getCorrect_residuals(img_size),
                                              this->getCorrect_weight(img_size),
                                              this->k_degrees_of_freedom_t_student_,
                                              variance,
                                              strm_[6]);
    
    auto rows = this->getCorrect_residuals(img_size)._getRows();
    auto columns = this->getCorrect_residuals(img_size)._getColumns();
    
    this->getCorrect_residuals(img_size)._setRows(rows * columns);
    this->getCorrect_residuals(img_size)._setColumns(1);
    
    cgmapping::cuda::calculate_Dsquared_weighted_error(this->getCorrect_residuals(img_size),
                                                       this->getCorrect_weight(img_size),
                                                       this->sqrd_weighted_error_,
                                                       this->getCorrect_auxiliar(img_size),
                                                       number_of_valid_data,
                                                       strm_[6]);
    
    this->getCorrect_residuals(img_size)._setRows(rows);
    this->getCorrect_residuals(img_size)._setColumns(columns);
    
    evnt_[6].record(strm_[6]);
    
}

template<typename T>
void cgmapping::cuda::img_motion_estimator<T>::optimizeDelta_twist(cgmapping::image_size_t img_size,
                                                                   T k_lambda_factor,
                                                                   culina_tm<T> &motion_prior,
                                                                   culina_tm<T> &cov_motion_prior) {
    
    auto n_rows = this->getCorrect_residuals(img_size)._getRows();
    auto n_cols = this->getCorrect_residuals(img_size)._getColumns();
    
    this->_getCorrect_residuals(img_size)._setRows(n_rows*n_cols);
    this->_getCorrect_residuals(img_size)._setColumns(1);
    
    strm_[0].waitEvent(evnt_[0]);
    strm_[1].waitEvent(evnt_[0]);
    strm_[2].waitEvent(evnt_[0]);
    
    cgmapping::cuda::solve_Dlinear_system_with_prior(this->getCorrect_full_jacobian(img_size),
                                                     this->delta_twist_,
                                                     this->getCorrect_residuals(img_size),
                                                     this->getCorrect_weight(img_size),
                                                     motion_prior,
                                                     cov_motion_prior,
                                                     this->twist_vector_,
                                                     k_lambda_factor,
                                                     this->auxiliar_matrix_linear_solver_1_,
                                                     this->auxiliar_matrix_linear_solver_2_,
                                                     this->lower_bound_twist_cov_matrix_,
                                                     strm_[0],
                                                     strm_[1],
                                                     strm_[2]);
    
    this->_getCorrect_residuals(img_size)._setRows(n_rows);
    this->_getCorrect_residuals(img_size)._setColumns(n_cols);
    
    twist_vec2linear_angular_vecs();
    
    evnt_[0].record(strm_[2]);
    evnt_[1].record(strm_[2]);
    evnt_[2].record(strm_[2]);
    
}

template<typename T>
void cgmapping::cuda::img_motion_estimator<T>::optimizeDelta_twist(cgmapping::image_size_t img_size,
                                                                   T k_lambda_factor) {
    
    auto n_rows = this->getCorrect_residuals(img_size)._getRows();
    auto n_cols = this->getCorrect_residuals(img_size)._getColumns();
    
    this->_getCorrect_residuals(img_size)._setRows(n_rows * n_cols);
    this->_getCorrect_residuals(img_size)._setColumns(1);
    
    strm_[0].waitEvent(evnt_[0]);
    strm_[1].waitEvent(evnt_[0]);
    strm_[2].waitEvent(evnt_[0]);
    
    cgmapping::cuda::solve_Dlinear_system(this->getCorrect_full_jacobian(img_size),
                                          this->delta_twist_,
                                          this->getCorrect_residuals(img_size),
                                          this->getCorrect_weight(img_size),
                                          k_lambda_factor,
                                          this->auxiliar_matrix_linear_solver_1_,
                                          this->auxiliar_matrix_linear_solver_2_,
                                          this->lower_bound_twist_cov_matrix_,
                                          strm_[0],
                                          strm_[1],
                                          strm_[2]);
    
    this->_getCorrect_residuals(img_size)._setRows(n_rows);
    this->_getCorrect_residuals(img_size)._setColumns(n_cols);
    
    twist_vec2linear_angular_vecs();
    
    evnt_[0].record(strm_[2]);
    evnt_[1].record(strm_[2]);
    evnt_[2].record(strm_[2]);
    
}
    
template<typename T>
void cgmapping::cuda::img_motion_estimator<T>::twist_vec2linear_angular_vecs()  {
    
    culiopD_t tmp_culiopD_1, tmp_culiopD_2;
    
    tmp_culiopD_1.alpha = 1;
    tmp_culiopD_2.alpha = 1;
    
    auto tmp_raw_strm_1 = cv::cuda::StreamAccessor::getStream(this->strm_[1]);
    auto tmp_raw_strm_2 = cv::cuda::StreamAccessor::getStream(this->strm_[2]);
    
    tmp_culiopD_1.strm = &tmp_raw_strm_1;
    tmp_culiopD_2.strm = &tmp_raw_strm_2;
    
    tmp_culiopD_1.op_m1 = CUBLAS_OP_N;
    tmp_culiopD_2.op_m1 = CUBLAS_OP_N;
    
    culina_Dblock_assignment_operation(&this->delta_twist_,
                                       &this->delta_linear_vel_,
                                       0, 0, 0, 0, 3, 1,
                                       tmp_culiopD_1);
    
    culina_Dblock_assignment_operation(&this->delta_twist_,
                                       &this->delta_angular_vel_,
                                       3, 0, 0, 0, 3, 1,
                                       tmp_culiopD_2);
    
}

template<typename T>
void cgmapping::cuda::img_motion_estimator<T>::updateTwist_variables(bool updateTwits_velocities) {
    
    strm_[3].waitEvent(evnt_[0]);
    strm_[4].waitEvent(evnt_[1]);
    
    cgmapping::cuda::calculate_Dexp_map_se3(delta_linear_vel_,
                                            delta_angular_vel_,
                                            delta_homogenic_transformation_,
                                            auxiliar_matrix_lie_algebra_1_,
                                            auxiliar_matrix_lie_algebra_2_,
                                            strm_[3],
                                            strm_[4]);
    
    //if(!evnt_[5].queryIfComplete()) evnt_[5].waitForCompletion();
    
    cgmapping::cuda::compose_Dpose_3D(tmp_homogenic_transformation_,
                                      delta_homogenic_transformation_,
                                      tmp_homogenic_transformation_,
                                      strm_[4]);
    
    cgmapping::cuda::calculate_Dlog_map_se3(tmp_linear_vel_,
                                            tmp_angular_vel_,
                                            tmp_homogenic_transformation_,
                                            auxiliar_matrix_lie_algebra_1_,
                                            strm_[4]);
    
    evnt_[0].record(strm_[4]);

}

template<typename T>
void cgmapping::cuda::img_motion_estimator<T>::incorporate_estimation()  {
    
    this->linear_vel_._loadData(this->tmp_linear_vel_, this->raw_strm_[0]);
    this->angular_vel_._loadData(this->tmp_angular_vel_, this->raw_strm_[0]);
    this->homogenic_transformation_._loadData(this->tmp_homogenic_transformation_, this->raw_strm_[0]);
    
    culiopD_t tmp_culiopD;
    
    tmp_culiopD.alpha = 1;
    
    tmp_culiopD.strm = &raw_strm_[0];
    
    tmp_culiopD.op_m1 = CUBLAS_OP_N;
    
    culina_Dblock_assignment_operation(&this->linear_vel_,
                                       &this->twist_vector_,
                                       0, 0, 0, 0, 3, 1,
                                       tmp_culiopD);
    
    culina_Dblock_assignment_operation(&this->angular_vel_,
                                       &this->twist_vector_,
                                       0, 0, 3, 0, 3, 1,
                                       tmp_culiopD);
    
    
    
}

template<typename T>
const cuLiNA::culina_matrix<T, 3, 1> &cgmapping::cuda::img_motion_estimator<T>::_getLinear_vel() const {
    return linear_vel_;
}

template<typename T>
const cuLiNA::culina_matrix<T, 3, 1> &cgmapping::cuda::img_motion_estimator<T>::_getAngular_vel() const {
    return angular_vel_;
}

template<typename T>
const cuLiNA::culina_matrix<T, 3, 1> &cgmapping::cuda::img_motion_estimator<T>::_getDelta_linear_vel() const {
    return delta_linear_vel_;
}

template<typename T>
const cuLiNA::culina_matrix<T, 3, 1> &cgmapping::cuda::img_motion_estimator<T>::_getDelta_angular_vel() const {
    return delta_angular_vel_;
}

template<typename T>
const culina_matrix<T, 4, 4> &cgmapping::cuda::img_motion_estimator<T>::_getHomogenic_transformation() const {
    return homogenic_transformation_;
}

template<typename T>
const cuLiNA::culina_matrix<T, 6, 6> &cgmapping::cuda::img_motion_estimator<T>::_getLower_bound_twist_cov_matrix() const {
    return static_cast<const cuLiNA::culina_matrix<T, 6, 6> &>(lower_bound_twist_cov_matrix_);
}

template<typename T>
const cuLiNA::culina_matrix<T, 6, 1> &cgmapping::cuda::img_motion_estimator<T>::_getDelta_twist_vels() const {
    return this->twist_vector_;
}

template<typename T>
cgmapping::image_pyramid & cgmapping::cuda::img_motion_estimator<T>::_getCur_img_warped() const {
    return const_cast<img_motion_estimator<T> *>(this)->cur_img_warped_;
}

template<typename T>
cgmapping::image_pyramid & cgmapping::cuda::img_motion_estimator<T>::_getCur_warped_filter() const {
    return const_cast<img_motion_estimator<T> *>(this)->cur_warped_filter_;
}

template<typename T>
cgmapping::image_pyramid & cgmapping::cuda::img_motion_estimator<T>::_getCur_img_warped_x_derivative() const {
    return const_cast<img_motion_estimator<T> *>(this)->cur_img_warped_x_derivative_;
}

template<typename T>
cgmapping::image_pyramid & cgmapping::cuda::img_motion_estimator<T>::_getCur_img_warped_y_derivative() const {
    return const_cast<img_motion_estimator<T> *>(this)->cur_img_warped_y_derivative_;
}

template<typename T>
culina_tm<T> & cgmapping::cuda::img_motion_estimator<T>::_getCorrect_residuals(cgmapping::image_size_t img_size) const {
    switch(img_size){
        
        case image_size_t::ORIGINAL_SIZE: return const_cast<img_motion_estimator<T> *>(this)->residual_original_;
        case image_size_t::HALF_SIZE: return const_cast<img_motion_estimator<T> *>(this)->residual_half_;
        case image_size_t::QUARTER_SIZE: return const_cast<img_motion_estimator<T> *>(this)->residual_quarter_;
        case image_size_t::OCT_SIZE: return const_cast<img_motion_estimator<T> *>(this)->residual_oct_;
        default: return const_cast<img_motion_estimator<T> *>(this)->residual_hexadec_;
        
    }
}

template<typename T>
culina_tm<T> & cgmapping::cuda::img_motion_estimator<T>::_getCorrect_warp_jacobian(cgmapping::image_size_t img_size) const  {
    
    switch(img_size){
        
        case image_size_t::ORIGINAL_SIZE: return const_cast<img_motion_estimator<T> *>(this)->warp_jacobian_original_;
        case image_size_t::HALF_SIZE: return const_cast<img_motion_estimator<T> *>(this)->warp_jacobian_half_;
        case image_size_t::QUARTER_SIZE: return const_cast<img_motion_estimator<T> *>(this)->warp_jacobian_quarter_;
        case image_size_t::OCT_SIZE: return const_cast<img_motion_estimator<T> *>(this)->warp_jacobian_oct_;
        default: return const_cast<img_motion_estimator<T> *>(this)->warp_jacobian_hexadec_;
        
    }
    
}

template<typename T>
culina_tm<T> & cgmapping::cuda::img_motion_estimator<T>::_getCorrect_full_jacobian(cgmapping::image_size_t img_size) const {
    
    switch(img_size){
        
        case image_size_t::ORIGINAL_SIZE: return const_cast<img_motion_estimator<T> *>(this)->full_jacobian_original_;
        case image_size_t::HALF_SIZE: return const_cast<img_motion_estimator<T> *>(this)->full_jacobian_half_;
        case image_size_t::QUARTER_SIZE: return const_cast<img_motion_estimator<T> *>(this)->full_jacobian_quarter_;
        case image_size_t::OCT_SIZE: return const_cast<img_motion_estimator<T> *>(this)->full_jacobian_oct_;
        default: return const_cast<img_motion_estimator<T> *>(this)->full_jacobian_hexadec_;
        
    }
    
}

template<typename T>
culina_tm<T> & cgmapping::cuda::img_motion_estimator<T>::_getCorrect_prev_img_point_cloud(cgmapping::image_size_t img_size) const {
    
    switch(img_size){
        
        case image_size_t::ORIGINAL_SIZE: return const_cast<img_motion_estimator<T> *>(this)->prev_img_point_cloud_original_;
        case image_size_t::HALF_SIZE: return const_cast<img_motion_estimator<T> *>(this)->prev_img_point_cloud_half_;
        case image_size_t::QUARTER_SIZE: return const_cast<img_motion_estimator<T> *>(this)->prev_img_point_cloud_quarter_;
        case image_size_t::OCT_SIZE: return const_cast<img_motion_estimator<T> *>(this)->prev_img_point_cloud_oct_;
        default: return const_cast<img_motion_estimator<T> *>(this)->prev_img_point_cloud_hexadec_;
        
    }
    
}

template<typename T>
culina_tm<T> & cgmapping::cuda::img_motion_estimator<T>::_getCorrect_weight(cgmapping::image_size_t img_size) const {
    
    switch(img_size){
        
        case image_size_t::ORIGINAL_SIZE: return const_cast<img_motion_estimator<T> *>(this)->weight_original_;
        case image_size_t::HALF_SIZE: return const_cast<img_motion_estimator<T> *>(this)->weight_half_;
        case image_size_t::QUARTER_SIZE: return const_cast<img_motion_estimator<T> *>(this)->weight_quarter_;
        case image_size_t::OCT_SIZE: return const_cast<img_motion_estimator<T> *>(this)->weight_oct_;
        default: return const_cast<img_motion_estimator<T> *>(this)->weight_hexadec_;
        
    }
    
}

template<typename T>
cuLiNA::culina_tm<T> &cgmapping::cuda::img_motion_estimator<T>::_getCorrect_auxiliar(cgmapping::image_size_t img_size) const {
    switch(img_size){
        
        case image_size_t::ORIGINAL_SIZE: return const_cast<img_motion_estimator<T> *>(this)->auxiliar_original_;
        case image_size_t::HALF_SIZE: return const_cast<img_motion_estimator<T> *>(this)->auxiliar_half_;
        case image_size_t::QUARTER_SIZE: return const_cast<img_motion_estimator<T> *>(this)->auxiliar_quarter_;
        case image_size_t::OCT_SIZE: return const_cast<img_motion_estimator<T> *>(this)->auxiliar_oct_;
        default: return const_cast<img_motion_estimator<T> *>(this)->auxiliar_hexadec_;
        
    }
}

template<typename T>
uint cgmapping::cuda::img_motion_estimator<T>::_getCorrect_iterations_at_img_resolution(cgmapping::image_size_t img_size) const {
    switch(img_size){
        
        case image_size_t::ORIGINAL_SIZE: return const_cast<img_motion_estimator<T> *>(this)->iterations_at_highest_resolution_;
        case image_size_t::HALF_SIZE: return const_cast<img_motion_estimator<T> *>(this)->iterations_at_half_resolution_;
        case image_size_t::QUARTER_SIZE: return const_cast<img_motion_estimator<T> *>(this)->iterations_at_quarter_resolution_;
        case image_size_t::OCT_SIZE: return const_cast<img_motion_estimator<T> *>(this)->iterations_at_oct_resolution_;
        default: return const_cast<img_motion_estimator<T> *>(this)->iterations_at_hexadec_resolution_;
        
    }
}

template<typename T>
cgmapping::cuda::img_motion_estimator<T>::~img_motion_estimator() {

}

template class cgmapping::cuda::img_motion_estimator<double>;
//template class cgmapping::cuda::img_motion_estimator<float>;