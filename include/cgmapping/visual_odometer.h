//
// Created by spades on 15/03/18.
//

#ifndef CGMAPPING_VISUAL_ODOMETER_H
#define CGMAPPING_VISUAL_ODOMETER_H

#include <cgmapping/image_pyramid.h>
#include <cuLiNA/culina_base_matrix.h>
#include <cuLiNA/culina_definition.h>
#include <cgmapping/rgb_d_camera_model.h>
#include <cgmapping/img_motion_estimator.h>

#include <Eigen/Dense>

namespace cgmapping {
    
    namespace cuda {
    
        template<typename T = double>
        class visual_odometer {
        
            cgmapping::image_pyramid prev_img_;
            cgmapping::image_pyramid prev_img_depth_;
            
            cgmapping::image_pyramid cur_img_;
            
            cgmapping::image_pyramid cur_img_depth_;
     
            cgmapping::rgb_d_camera_model camera_model_;
    
            img_motion_estimator<T> img_motion_estimator_;
            
            cuLiNA::culina_matrix<T, 3, 1> cur_linear_vel_;
            cuLiNA::culina_matrix<T, 3, 1> cur_angular_vel_;
            
            cuLiNA::culina_matrix<T, 6, 6> cov_motion_prior_;
            cuLiNA::culina_matrix<T, 6, 1> motion_prior_;
            
            cuLiNA::culina_matrix<T, 4, 4> cur_transform_;
            cuLiNA::culina_matrix<T, 4, 4> delta_inverse_transform_;
            
            cuLiNA::culina_matrix<T,3,3> auxiliar_matrix_lie_algebra_1_;
            
            T k_eps_error_ = 0.000001;
            
            uint k_max_iterations_for_optmization_ = 20;
            
            cgmapping::image_size_t min_resolution_;
            cgmapping::image_size_t max_resolution_;
            
            cudaStream_t raw_strm_[4];
            cv::cuda::Stream strm_[4];
            
         public:
        
            /***
             *
             * @brief This object is responsible for performing visual odometry based on images coming from a RGB-D source.
             *
             * This objects needs two pair of images to work properly, one from the previous reading and one from the current
             * moment, where theses pairs are composed by a depth map and a gray scale image.
             *
             * The objective of this object is to simply calculate the delta of displacement between two images
             *
             *
             */

            //visual_odometer();
    
            visual_odometer(const Mat &cur_img,
                            const Mat &cur_img_depth,
                            rgb_d_camera_model camera_model,
                            T eps_error,
                            uint k_max_iterations_for_optmization,
                            T k_degrees_of_freedom_t_student = 0.5,
                            T k_epslon_accptance_stud_dev = 0.0001,
                            T *initial_position = nullptr);
    
            void _estimateCamera_motion(const Mat &cur_img,
                                        const Mat &cur_img_depth,
                                        cgmapping::image_size_t min_img_size,
                                        cgmapping::image_size_t max_img_size);
            
            T _getK_eps_error() const;
            uint _getK_max_iterations_for_optmization() const;
            uint _getIterations_at_img_resolution(cgmapping::image_size_t img_size) const;
            
            cgmapping::image_pyramid & _getCur_img_warped() const;
            cgmapping::image_pyramid & _getCur_warped_filter() const;
            cgmapping::image_pyramid & _getCur_img_warped_x_derivative() const;
            cgmapping::image_pyramid & _getCur_img_warped_y_derivative() const;
            culina_tm<T> & _getResiduals_at_img_size(cgmapping::image_size_t img_size) const;
            culina_tm<T> & _getFull_jacobian_at_img_size(cgmapping::image_size_t img_size) const;
            culina_tm<T> & _getPrev_img_point_cloud_at_img_size(cgmapping::image_size_t img_size) const;
            culina_tm<T> & _getWeight_at_img_size(cgmapping::image_size_t img_size) const;
      
            image_size_t _getMin_resolution() const;
            image_size_t _getMax_resolution() const;
            
            const cuLiNA::culina_matrix<T, 3, 1> &_getLinear_vel() const;
            const cuLiNA::culina_matrix<T, 3, 1> &_getAngular_vel() const;
            const cuLiNA::culina_matrix<T, 3, 1> &_getDelta_linear_vel() const;
            const cuLiNA::culina_matrix<T, 3, 1> &_getDelta_angular_vel() const;
            const culina_matrix<T, 4, 4> &_getCur_estimated_pose() const;
            const culina_matrix<T, 4, 4> &_getCur_delta_transform() const;
            const cuLiNA::culina_matrix<T, 6, 6> &_getLower_bound_delta_twist_cov_matrix() const;
    
        };
    }
}

#endif //CGMAPPING_VISUAL_ODOMETER_H
