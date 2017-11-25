//
// Created by spades on 03/04/18.
//

#ifndef CGMAPPING_IMG_MOTION_ESTIMATOR_H
#define CGMAPPING_IMG_MOTION_ESTIMATOR_H

#include <cuLiNA/culina_definition.h>

#include <cgmapping/image_pyramid.h>
#include <cgmapping/rgb_d_camera_model.h>
#include <cgmapping/se3_lie_algebra_utils.cuh>
#include <cgmapping/cgmapping_utils.h>

#include <opencv2/cudafilters.hpp>

namespace cgmapping {
    
    namespace cuda {
    
        template<typename T>
        class img_motion_estimator {
            
            cgmapping::image_pyramid cur_img_warped_;
            cgmapping::image_pyramid cur_warped_filter_;
    
            cgmapping::image_pyramid cur_img_warped_x_derivative_;
            cgmapping::image_pyramid cur_img_warped_y_derivative_;
            
            cuLiNA::culina_base_matrix<T> warp_jacobian_original_;
            cuLiNA::culina_base_matrix<T> warp_jacobian_half_;
            cuLiNA::culina_base_matrix<T> warp_jacobian_quarter_;
            cuLiNA::culina_base_matrix<T> warp_jacobian_oct_;
            cuLiNA::culina_base_matrix<T> warp_jacobian_hexadec_;
        
            cuLiNA::culina_base_matrix<T> full_jacobian_original_;
            cuLiNA::culina_base_matrix<T> full_jacobian_half_;
            cuLiNA::culina_base_matrix<T> full_jacobian_quarter_;
            cuLiNA::culina_base_matrix<T> full_jacobian_oct_;
            cuLiNA::culina_base_matrix<T> full_jacobian_hexadec_;
        
            cuLiNA::culina_base_matrix<T> prev_img_point_cloud_original_;
            cuLiNA::culina_base_matrix<T> prev_img_point_cloud_half_;
            cuLiNA::culina_base_matrix<T> prev_img_point_cloud_quarter_;
            cuLiNA::culina_base_matrix<T> prev_img_point_cloud_oct_;
            cuLiNA::culina_base_matrix<T> prev_img_point_cloud_hexadec_;
        
            cuLiNA::culina_base_matrix<T> weight_original_;
            cuLiNA::culina_base_matrix<T> weight_half_;
            cuLiNA::culina_base_matrix<T> weight_quarter_;
            cuLiNA::culina_base_matrix<T> weight_oct_;
            cuLiNA::culina_base_matrix<T> weight_hexadec_;
    
            cuLiNA::culina_base_matrix<T> residual_original_;
            cuLiNA::culina_base_matrix<T> residual_half_;
            cuLiNA::culina_base_matrix<T> residual_quarter_;
            cuLiNA::culina_base_matrix<T> residual_oct_;
            cuLiNA::culina_base_matrix<T> residual_hexadec_;
    
            cuLiNA::culina_base_matrix<T> auxiliar_original_;
            cuLiNA::culina_base_matrix<T> auxiliar_half_;
            cuLiNA::culina_base_matrix<T> auxiliar_quarter_;
            cuLiNA::culina_base_matrix<T> auxiliar_oct_;
            cuLiNA::culina_base_matrix<T> auxiliar_hexadec_;
    
            cuLiNA::culina_base_matrix<T> auxiliar_matrix_linear_solver_1_;
            cuLiNA::culina_matrix<T, 6, 1> auxiliar_matrix_linear_solver_2_;
            cuLiNA::culina_matrix<T, 6, 6> lower_bound_twist_cov_matrix_;
            
            cuLiNA::culina_matrix<T,3,3> auxiliar_matrix_lie_algebra_1_;
            cuLiNA::culina_matrix<T,3,3> auxiliar_matrix_lie_algebra_2_;
            
            cuLiNA::culina_matrix<T, 3, 1> linear_vel_;
            cuLiNA::culina_matrix<T, 3, 1> angular_vel_;
    
            cuLiNA::culina_matrix<T, 3, 1> tmp_linear_vel_;
            cuLiNA::culina_matrix<T, 3, 1> tmp_angular_vel_;
    
            cuLiNA::culina_matrix<T, 3, 1> delta_linear_vel_;
            cuLiNA::culina_matrix<T, 3, 1> delta_angular_vel_;
            
            cuLiNA::culina_matrix<T, 6, 1> delta_twist_;
            cuLiNA::culina_matrix<T, 6, 1> twist_vector_;
            
            cuLiNA::culina_matrix<T, 4, 4> homogenic_transformation_;
            cuLiNA::culina_matrix<T, 4, 4> tmp_homogenic_transformation_;
            cuLiNA::culina_matrix<T, 4, 4> delta_homogenic_transformation_;
            
            cuLiNA::culina_matrix<T, 1, 1> sqrd_weighted_error_;
            
            uint iterations_at_highest_resolution_;
            uint iterations_at_half_resolution_;
            uint iterations_at_quarter_resolution_;
            uint iterations_at_oct_resolution_;
            uint iterations_at_hexadec_resolution_;
    
            Ptr<cv::cuda::Filter> filter_x_, filter_y_;
            
            cudaStream_t raw_strm_[7];
            cudaEvent_t raw_evnt_[7];
            cv::cuda::Stream strm_[7];
            cv::cuda::Event evnt_[7];
    
            T k_degrees_of_freedom_t_student_;
            T k_epslon_accptance_std_dev_;
            
            cuLiNA::culina_tm<T> &getCorrect_residuals(cgmapping::image_size_t img_size);
            cuLiNA::culina_tm<T> &getCorrect_warp_jacobian(cgmapping::image_size_t img_size);
            cuLiNA::culina_tm<T> &getCorrect_full_jacobian(cgmapping::image_size_t img_size);
            cuLiNA::culina_tm<T> &getCorrect_prev_img_point_cloud(cgmapping::image_size_t img_size);
            cuLiNA::culina_tm<T> &getCorrect_weight(cgmapping::image_size_t img_size);
            cuLiNA::culina_tm<T> &getCorrect_auxiliar(cgmapping::image_size_t img_size);
            uint &getCorrect_iterations_at_img_resolution(cgmapping::image_size_t img_size);
            
            void calculateWarped_img(cv::cuda::GpuMat &cur_img_original,
                                     cv::cuda::GpuMat &prev_depth_img,
                                     cgmapping::rgb_d_camera_model &camera_model,
                                     cgmapping::image_size_t img_size);
            
            void calculateResidual(cv::cuda::GpuMat &img1,
                                   cgmapping::image_size_t img_size);
            
            void calculateWarped_img_derivatives(cgmapping::image_size_t img_size);
            
            void calculateWarped_jacobian(rgb_d_camera_model &camera_model, cgmapping::image_size_t img_size);
            
            void calculateJacobian(cgmapping::image_size_t img_size);
            
            void updateTwist_variables(bool updateTwits_velocities);
            
            int countValid_data(cgmapping::image_size_t img_size);
            
            double calculateStd_dev_t_student(cgmapping::image_size_t img_size, int number_of_valid_data);
            
            void updateWeighted_error(cgmapping::image_size_t img_size,
                                                  double std_dev,
                                                  int number_of_valid_data);
    
            void initializeTwist_variables(culina_matrix<T, 3, 1> &init_linear_vel, culina_matrix<T, 3, 1> &init_angular_vel);
            
            void optimizeDelta_twist(cgmapping::image_size_t img_size,
                                                 T k_lambda_factor,
                                                 culina_tm<T> &motion_prior,
                                                 culina_tm<T> &cov_motion_prior);
    
            void optimizeDelta_twist(cgmapping::image_size_t img_size,
                                     T k_lambda_factor);
    
            void incorporate_estimation();
            
            void twist_vec2linear_angular_vecs();
            
         public:
        
            //img_motion_estimator(){};
    
            img_motion_estimator(uint img_rows,
                                 uint img_cols,
                                 T k_degrees_of_freedom_t_student = 0.5,
                                 T k_epslon_accptance_std_dev = 0.00001);
    
            void _estimateMotion_from_images(cgmapping::image_pyramid &img1_pyr,
                                                         cgmapping::image_pyramid &depth1_pyr,
                                                         cgmapping::image_pyramid &img2_pyr,
                                                         cgmapping::image_size_t img_size,
                                                         culina_matrix<T, 3, 1> &init_linear_vel,
                                                         culina_matrix<T, 3, 1> &init_angular_vel,
                                                         cgmapping::rgb_d_camera_model &camera_model,
                                                         T k_acceptance_error,
                                                         uint k_max_interations);
    
            void _estimateMotion_from_images(cgmapping::image_pyramid &img1_pyr,
                                             cgmapping::image_pyramid &depth1_pyr,
                                             cgmapping::image_pyramid &img2_pyr,
                                             cgmapping::image_size_t img_size,
                                             culina_matrix<T, 3, 1> &init_linear_vel,
                                             culina_matrix<T, 3, 1> &init_angular_vel,
                                             culina_matrix<T, 6, 1> &motion_prior,
                                             culina_matrix<T, 6, 6> &cov_motion_prior,
                                             cgmapping::rgb_d_camera_model &camera_model,
                                             T k_acceptance_error,
                                             uint k_max_interations);
    
            const cuLiNA::culina_matrix<T, 3, 1> &_getLinear_vel() const;
            const cuLiNA::culina_matrix<T, 3, 1> &_getAngular_vel() const;
            const cuLiNA::culina_matrix<T, 3, 1> &_getDelta_linear_vel() const;
            const cuLiNA::culina_matrix<T, 3, 1> &_getDelta_angular_vel() const;
            const culina_matrix<T, 4, 4> &_getHomogenic_transformation() const;
            const cuLiNA::culina_matrix<T, 6, 6> &_getLower_bound_twist_cov_matrix() const;
            const cuLiNA::culina_matrix<T, 6, 1> &_getDelta_twist_vels() const;
            
            cgmapping::image_pyramid & _getCur_img_warped() const;
            cgmapping::image_pyramid & _getCur_warped_filter() const;
            cgmapping::image_pyramid & _getCur_img_warped_x_derivative() const;
            cgmapping::image_pyramid & _getCur_img_warped_y_derivative() const;
            culina_tm<T> & _getCorrect_residuals(cgmapping::image_size_t img_size) const;
            culina_tm<T> & _getCorrect_warp_jacobian(cgmapping::image_size_t img_size) const;
            culina_tm<T> & _getCorrect_full_jacobian(cgmapping::image_size_t img_size) const;
            culina_tm<T> & _getCorrect_prev_img_point_cloud(cgmapping::image_size_t img_size) const;
            culina_tm<T> & _getCorrect_weight(cgmapping::image_size_t img_size) const;
            cuLiNA::culina_tm<T> &_getCorrect_auxiliar(cgmapping::image_size_t img_size) const;
            uint _getCorrect_iterations_at_img_resolution(cgmapping::image_size_t img_size) const;
    
            virtual ~img_motion_estimator();
    
        };
    
    }

}

#endif //CGMAPPING_IMG_MOTION_ESTIMATOR_H
