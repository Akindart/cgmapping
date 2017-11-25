//
// Created by spades on 21/08/17.
//

#ifndef CGMAPPING_CGMAPPING_UTILS_CUH
#define CGMAPPING_CGMAPPING_UTILS_CUH

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <opencv2/core/mat.hpp>
#include <cuLiNA/culina_base_matrix.h>
#include <cuda_parsing_helper_in_clion/clion_helper.h>
#include <cgmapping/cgmapping_utils_kernels.cuh>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

namespace cgmapping {
    
    namespace cuda {
        
        __host__
        extern long count_valid_data(cuLiNA::culina_base_matrix<double> &data, cudaStream_t *stream = NULL);
        
        __host__
        extern double calculate_standart_deviation_t_student_step(cuLiNA::culina_base_matrix<double> &data,
                                                                  double degrees_of_freedom,
                                                                  double standard_deviation_k_minus_1,
                                                                  int number_of_valid_data,
                                                                  cudaStream_t *stream = NULL);
        
        __host__
        extern void define_data_weight_t_student(cuLiNA::culina_base_matrix<double> &data,
                                                 cuLiNA::culina_base_matrix<double> &data_weighted,
                                                 double degrees_of_freedom,
                                                 double standard_deviation,
                                                 cudaStream_t *stream = NULL);
        
        __host__
        extern bool compute_kernel_size_for_matrix_operation(int n_rows,
                                                             int n_columns,
                                                             int depth,
                                                             dim3 &block_dim,
                                                             dim3 &grid_dim,
                                                             int *max_number_threads = NULL,
                                                             int *max_number_blocks = NULL);
        
        /**
         *
         * This function requires both rgb and depth to be aligned
         *
         * */
        __host__
        extern void pixel_residual_calculation(cv::cuda::GpuMat &img_rgb_t_minus_1,
                                               cv::cuda::GpuMat &img_rgb_t,
                                               cv::cuda::GpuMat &img_depth_t_minus_1,
                                               cv::cuda::GpuMat &img_depth_t,
                                               cuLiNA::culina_base_matrix<double> &homogenic_transformation,
                                               cuLiNA::culina_base_matrix<double> &residual_matrix,
                                               float camera_focus_x,
                                               float camera_focus_y,
                                               float camera_centroid_x,
                                               float camera_centroid_y,
                                               int scale_factor = 1,
                                               cudaStream_t *strm = NULL);
        
        /**
         *
         * This function requires both rgb and depth to be aligned
         *
         * */
        __host__
        extern void residual_jacobian_calculation(cv::cuda::GpuMat &img_rgb_t_minus_1,
                                                  cv::cuda::GpuMat &img_rgb_t,
                                                  cv::cuda::GpuMat &img_depth_t_minus_1,
                                                  cv::cuda::GpuMat &img_depth_t,
                                                  cuLiNA::culina_base_matrix<double> &homogenic_transformation_positive_disturb,
                                                  cuLiNA::culina_base_matrix<double> &homogenic_transformation_negative_disturb,
                                                  cuLiNA::culina_base_matrix<double> &jacobian,
                                                  int var_number,
                                                  float camera_focus_x,
                                                  float camera_focus_y,
                                                  float camera_centroid_x,
                                                  float camera_centroid_y,
                                                  float delta,
                                                  int scale_factor,
                                                  cudaStream_t *strm = NULL);
        
    }
    
}

#endif //CGMAPPING_CGMAPPING_UTILS_CUH
