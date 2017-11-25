//
// Created by spades on 21/08/17.
//

#ifndef CGMAPPING_CGMAPPING_UTILS_KERNELS_CUH
#define CGMAPPING_CGMAPPING_UTILS_KERNELS_CUH

#include <cuda_parsing_helper_in_clion/clion_helper.h>
#include <boost/mpl/size_t.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/core/cuda.hpp>


__forceinline__ __device__ __host__ int idx2c_rm(int i, int j, int ld) { return (ld * i + j); }
__forceinline__ __device__ __host__ int idx2c_cm(int i, int j, int ld) { return (ld * j + i); }

template<class V>
__forceinline__ __device__ __host__ V get_gpumat_img_pixel_val(V * img_ptr, int i, int j, size_t step) {
    
    V *tmp_prt = (V *)(((char *)img_ptr) + i*step);
   
    return tmp_prt[j];

}


__forceinline__ __device__ extern void base_changeD(double *d_homogeneous_matrix,
                                    bool inverse_transformation,
                                    double *d_original_vector,
                                    double *d_result_vector);

__forceinline__ __device__ extern void projection_space2image(double pixel_space_pos[3],
                                              int *pixel_pos,
                                              float camera_focus_x,
                                              float camera_focus_y,
                                              float camera_centroid_x,
                                              float camera_centroid_y);

__forceinline__ __device__ extern void projection_image2space(int *pixel_pos,
                                              double *pixel_space_pos,
                                              float camera_focus_x,
                                              float camera_focus_y,
                                              float camera_centroid_x,
                                              float camera_centroid_y);

__forceinline__ __device__ extern void projection_pixel_img1_into_img2(double *d_homogenic_transformation,
                                                                       int *pixel_pos_img1,
                                                                       int *pixel_pos_img2,
                                                                       cv::cuda::PtrStepSz<unsigned short> d_depth_image1,
                                                                       float camera_focus_x,
                                                                       float camera_focus_y,
                                                                       float camera_centroid_x,
                                                                       float camera_centroid_y,
                                                                       int scale_factor);

__global__ extern void pixel_residual_calculation_kernel(cv::cuda::PtrStepSzb d_image1,
                                                         cv::cuda::PtrStepSzb d_image2,
                                                         cv::cuda::PtrStepSz<unsigned short> d_depth_image1,
                                                         double *d_homogenic_transformation,
                                                         double *d_residuals,
                                                         float camera_focus_x,
                                                         float camera_focus_y,
                                                         float camera_centroid_x,
                                                         float camera_centroid_y,
                                                         unsigned short scale_factor);

__global__ extern void residual_jacobian_calculation_kernel(cv::cuda::PtrStepSzb d_image,
                                                            cv::cuda::PtrStepSz<unsigned short> d_depth_image,
                                                            double *d_homogenic_transformation_positive_disturb,
                                                            double *d_homogenic_transformation_negative_disturb,
                                                            double *d_jacobian,
                                                            int var_number,
                                                            float camera_focus_x,
                                                            float camera_focus_y,
                                                            float camera_centroid_x,
                                                            float camera_centroid_y,
                                                            float delta,
                                                            unsigned short scale_factor);

#endif //CGMAPPING_CGMAPPING_UTILS_KERNELS_CUH
