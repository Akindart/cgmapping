//
// Created by spades on 21/08/17.
//

#ifndef CGMAPPING_CGMAPPING_UTILS_CUH
#define CGMAPPING_CGMAPPING_UTILS_CUH

#include <cuda.h>
#include <cuda_device_runtime_api.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <cuLiNA/culina_definition.h>
#include <cuLiNA/culina_base_matrix.h>
#include <cuda_parsing_helper_in_clion/clion_helper.h>
#include <cgmapping/cgmapping_utils_kernels.cuh>

#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>


#include "rgb_d_camera_model.h"

namespace cgmapping {
    
    namespace cuda {
        
        __host__
        extern long count_valid_data(cuLiNA::culina_tm<double> &data, cudaStream_t *stream = NULL);
        
        __host__
        extern double variance_t_student_step_Dcalculation_operation(cuLiNA::culina_tm<double> &data,
                                                                     double degrees_of_freedom,
                                                                     double variance_k_minus_1,
                                                                     int number_of_valid_data,
                                                                     cudaStream_t *stream = NULL);
        
        __host__
        extern void weight_matrix_t_student_Dcalculation_operation(cuLiNA::culina_tm<double> &data,
                                                                   cuLiNA::culina_tm<double> &weight_matrix,
                                                                   double degrees_of_freedom,
                                                                   double variance,
                                                                   cudaStream_t *stream = NULL);
        
        __host__
        extern bool compute_kernel_size_for_matrix_operation(int n_rows,
                                                             int n_columns,
                                                             int depth,
                                                             dim3 &block_dim,
                                                             dim3 &grid_dim,
                                                             int *max_number_threads = NULL,
                                                             int *max_number_blocks = NULL);
        
        __host__
        extern void warped_image_Dcalculation_operation(cv::cuda::GpuMat &img_rgb_t,
                                                        cv::cuda::GpuMat &img_rgb_t_warped,
                                                        cv::cuda::GpuMat &img_rgb_t_warped_filter,
                                                        cv::cuda::GpuMat &img_depth_t_minus_1,
                                                        cuLiNA::culina_tm<double> &d_img_t_minus_1_point_cloud,
                                                        cuLiNA::culina_matrix4d &d_homogenic_transformation,
                                                        float camera_focus_x,
                                                        float camera_focus_y,
                                                        float camera_centroid_x,
                                                        float camera_centroid_y,
                                                        int scale_factor,
                                                        cudaStream_t *strm  = NULL);
        
        __host__
        extern void image_raw_derivatives_x_y_operation(cv::cuda::GpuMat &img,
                                                        cv::cuda::GpuMat &img_derivative,
                                                        bool x,
                                                        bool y,
                                                        cudaStream_t *strm = NULL);
        
        /**
         *
         * This function requires both rgb and depth to be aligned
         *
         * */
        __host__
        extern void pixel_residual_Dcalculation_operation(cv::cuda::GpuMat &img_rgb_t_minus_1,
                                                          cv::cuda::GpuMat &img_rgb_t_warped,
                                                          cv::cuda::GpuMat &img_rgb_t_warped_filter,
                                                          cuLiNA::culina_tm<double> &residual_matrix,
                                                          cudaStream_t *strm);
        
//        /**
//         *
//         * This function requires both rgb and depth to be aligned
//         *
//         * */
//        __host__
//        extern void residual_jacobian_calculation(cv::cuda::GpuMat &img_rgb_t_minus_1,
//                                                  cv::cuda::GpuMat &img_rgb_t,
//                                                  cv::cuda::GpuMat &img_depth_t_minus_1,
//                                                  cv::cuda::GpuMat &img_depth_t,
//                                                  cuLiNA::culina_tm<double> &homogenic_transformation_positive_disturb,
//                                                  cuLiNA::culina_tm<double> &homogenic_transformation_negative_disturb,
//                                                  cuLiNA::culina_tm<double> &jacobian,
//                                                  int var_number,
//                                                  float camera_focus_x,
//                                                  float camera_focus_y,
//                                                  float camera_centroid_x,
//                                                  float camera_centroid_y,
//                                                  float delta,
//                                                  int scale_factor,
//                                                  cudaStream_t *strm = NULL);
    
        __host__
        extern void warp_jacobian_Dcalculation_operation(cuLiNA::culina_tm<double> &d_warp_jacobian,
                                                         cuLiNA::culina_tm<double> &d_img1_point_cloud,
                                                         cgmapping::rgb_d_camera_model &h_camera_model,
                                                         cv::cuda::Stream &strm = cv::cuda::Stream::Null());
    
        __host__
        extern void full_jacobian_Dcalculation_operation(cv::cuda::GpuMat &warped_img_x_derivative,
                                                         cv::cuda::GpuMat &warped_img_y_derivative,
                                                         cuLiNA::culina_tm<double> &d_warp_jacobian,
                                                         cuLiNA::culina_tm<double> &d_jacobian,
                                                         cv::cuda::Stream &strm = cv::cuda::Stream::Null());
        
    }
    
}

#endif //CGMAPPING_CGMAPPING_UTILS_CUH
