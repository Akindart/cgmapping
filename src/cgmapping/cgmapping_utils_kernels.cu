//
// Created by spades on 21/08/17.
//

#include <cgmapping/cgmapping_utils_kernels.cuh>
#include <cuLiNA/culina_utils.cuh>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/core/hal/interface.h>

__forceinline__ __device__ void base_changeD(double *d_homogeneous_matrix,
                                             bool inverse_transformation,
                                             double *d_original_vector,
                                             double *d_result_vector) {
    
    if (inverse_transformation) {
        
        d_result_vector[0] = d_homogeneous_matrix[idx2c_rm(0, 0, 4)] * d_original_vector[0] +
            d_homogeneous_matrix[idx2c_rm(1, 0, 4)] * d_original_vector[1] +
            d_homogeneous_matrix[idx2c_rm(2, 0, 4)] * d_original_vector[2];
        
        d_result_vector[1] = d_homogeneous_matrix[idx2c_rm(0, 1, 4)] * d_original_vector[0] +
            d_homogeneous_matrix[idx2c_rm(1, 1, 4)] * d_original_vector[1] +
            d_homogeneous_matrix[idx2c_rm(2, 1, 4)] * d_original_vector[2];
        
        d_result_vector[2] = d_homogeneous_matrix[idx2c_rm(0, 2, 4)] * d_original_vector[0] +
            d_homogeneous_matrix[idx2c_rm(1, 2, 4)] * d_original_vector[1] +
            d_homogeneous_matrix[idx2c_rm(2, 2, 4)] * d_original_vector[2];
        
        d_result_vector[0] += (-1) * d_homogeneous_matrix[idx2c_rm(0, 0, 4)] * d_homogeneous_matrix[idx2c_rm(0, 3, 4)] +
            d_homogeneous_matrix[idx2c_rm(1, 0, 4)] * d_homogeneous_matrix[idx2c_rm(1, 3, 4)] +
            d_homogeneous_matrix[idx2c_rm(2, 0, 4)] * d_homogeneous_matrix[idx2c_rm(2, 3, 4)];
        
        d_result_vector[1] += (-1) * d_homogeneous_matrix[idx2c_rm(0, 1, 4)] * d_homogeneous_matrix[idx2c_rm(0, 3, 4)] +
            d_homogeneous_matrix[idx2c_rm(1, 1, 4)] * d_homogeneous_matrix[idx2c_rm(1, 3, 4)] +
            d_homogeneous_matrix[idx2c_rm(2, 1, 4)] * d_homogeneous_matrix[idx2c_rm(2, 3, 4)];
        
        d_result_vector[2] += (-1) * (d_homogeneous_matrix[idx2c_rm(0, 2, 4)] * d_homogeneous_matrix[idx2c_rm(0, 3, 4)] +
            d_homogeneous_matrix[idx2c_rm(1, 2, 4)] * d_homogeneous_matrix[idx2c_rm(1, 3, 4)] +
            d_homogeneous_matrix[idx2c_rm(2, 2, 4)] * d_homogeneous_matrix[idx2c_rm(2, 3, 4)]);
        
    } else {
        
        d_result_vector[0] = d_homogeneous_matrix[idx2c_rm(0, 0, 4)] * d_original_vector[0] +
            d_homogeneous_matrix[idx2c_rm(0, 1, 4)] * d_original_vector[1] +
            d_homogeneous_matrix[idx2c_rm(0, 2, 4)] * d_original_vector[2] + d_homogeneous_matrix[idx2c_rm(0, 3, 4)];
        
        d_result_vector[1] = d_homogeneous_matrix[idx2c_rm(1, 0, 4)] * d_original_vector[0] +
            d_homogeneous_matrix[idx2c_rm(1, 1, 4)] * d_original_vector[1] +
            d_homogeneous_matrix[idx2c_rm(1, 2, 4)] * d_original_vector[2] + d_homogeneous_matrix[idx2c_rm(1, 3, 4)];
        
        d_result_vector[2] = d_homogeneous_matrix[idx2c_rm(2, 0, 4)] * d_original_vector[0] +
            d_homogeneous_matrix[idx2c_rm(2, 1, 4)] * d_original_vector[1] +
            d_homogeneous_matrix[idx2c_rm(2, 2, 4)] * d_original_vector[2] + d_homogeneous_matrix[idx2c_rm(2, 3, 4)];
        
    }
    
    return;
    
}

__forceinline__ __device__ void projection_space2image(double *pixel_space_pos,
                                                       int *pixel_pos,
                                                       float camera_focus_x,
                                                       float camera_focus_y,
                                                       float camera_centroid_x,
                                                       float camera_centroid_y) {
    
    pixel_pos[0] = (int) (((camera_focus_x * pixel_space_pos[0]) / pixel_space_pos[2]) + camera_centroid_x);
    pixel_pos[1] = (int) (((camera_focus_y * pixel_space_pos[1]) / pixel_space_pos[2]) + camera_centroid_y);
    
}

__forceinline__ __device__ void projection_image2space(int *pixel_pos,
                                                       double *pixel_space_pos,
                                                       float camera_focus_x,
                                                       float camera_focus_y,
                                                       float camera_centroid_x,
                                                       float camera_centroid_y) {
    
    pixel_space_pos[0] = (((pixel_pos[0] - camera_centroid_x) / camera_focus_x) * pixel_space_pos[2]);
    pixel_space_pos[1] = (((pixel_pos[1] - camera_centroid_y) / camera_focus_y) * pixel_space_pos[2]);
    
}

__forceinline__ __device__ void projection_pixel_img1_into_img2(double *d_homogenic_transformation,
                                                                int *pixel_pos_img1,
                                                                int *pixel_pos_img2,
                                                                cv::cuda::PtrStepSz<unsigned short> d_depth_image1,
                                                                float camera_focus_x,
                                                                float camera_focus_y,
                                                                float camera_centroid_x,
                                                                float camera_centroid_y,
                                                                int scale_factor) {
    
    double pixel_img1_space_pos_wrt_img1[3];
    double pixel_img1_space_pos_wrt_img2[3] = {0, 0, 0};
    int u, v;
    
    u = pixel_pos_img1[1];
    v = pixel_pos_img1[0];
    
    pixel_img1_space_pos_wrt_img1[0] = 0;
    pixel_img1_space_pos_wrt_img1[1] = 0;
    pixel_img1_space_pos_wrt_img1[2] = (double) ((int)d_depth_image1(u, v) / scale_factor);
    
    projection_image2space(pixel_pos_img1,
                           pixel_img1_space_pos_wrt_img1,
                           camera_focus_x,
                           camera_focus_y,
                           camera_centroid_x,
                           camera_centroid_y);
    
    base_changeD(d_homogenic_transformation,
                 false,
                 pixel_img1_space_pos_wrt_img1,
                 pixel_img1_space_pos_wrt_img2);
    
    projection_space2image(pixel_img1_space_pos_wrt_img2,
                           pixel_pos_img2,
                           camera_focus_x,
                           camera_focus_y,
                           camera_centroid_x,
                           camera_centroid_y);
    
}

__global__ void pixel_residual_calculation_kernel(cv::cuda::PtrStepSzb d_image1,
                                                  cv::cuda::PtrStepSzb d_image2,
                                                  cv::cuda::PtrStepSz<unsigned short> d_depth_image1,
                                                  double *d_homogenic_transformation,
                                                  double *d_residuals,
                                                  float camera_focus_x,
                                                  float camera_focus_y,
                                                  float camera_centroid_x,
                                                  float camera_centroid_y,
                                                  unsigned short scale_factor) {
    
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (j < d_image1.cols && i < d_image1.rows) {
        
        int residual_index = idx2c_cm(i, j, d_image2.rows);
        
        d_residuals[residual_index] = -100000;
        
        int pixel_pos_img1[2] = {j, i};
        int pixel_pos_img2[2] = {j, i};
    
        projection_pixel_img1_into_img2(d_homogenic_transformation,
                                        pixel_pos_img1,
                                        pixel_pos_img2,
                                        d_depth_image1,
                                        camera_focus_x,
                                        camera_focus_y,
                                        camera_centroid_x,
                                        camera_centroid_y,
                                        scale_factor);
        
        if (pixel_pos_img2[0] < d_image2.cols && pixel_pos_img2[1] < d_image2.rows &&
            pixel_pos_img2[0] >= 0 && pixel_pos_img2[1] >= 0) {
            
            int pixel_i_j_img1 = (int) d_image1(i, j);
            int pixel_i_j_img2 = (int) d_image2(pixel_pos_img2[1], pixel_pos_img2[0]);
            
            d_residuals[residual_index] = int2double(pixel_i_j_img2 - pixel_i_j_img1);
            
        }
        
    }
    
}

__global__ void residual_jacobian_calculation_kernel(cv::cuda::PtrStepSzb d_image,
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
                                                     unsigned short scale_factor) {
    
    
    /**
     *
     * d_jacobian is nx6, where n is the number of pixels in the image
     *
     */
    
    
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (j < d_image.cols && i < d_depth_image.rows) {
        
        int jacobian_index = idx2c_cm(i, j, d_image.cols);
        
        jacobian_index += (var_number/*1 <= var_number <= 6*/ - 1) * d_image.rows * d_image.cols;
        
        d_jacobian[jacobian_index] = 0;
        
        int pixel_pos_img1[2] = {j, i};
        int pixel_pos_img2_positive_disturb[2];
        int pixel_pos_img2_negative_disturb[2];
    
        projection_pixel_img1_into_img2(d_homogenic_transformation_positive_disturb,
                                        pixel_pos_img1,
                                        pixel_pos_img2_positive_disturb,
                                        d_depth_image,
                                        camera_focus_x,
                                        camera_focus_y,
                                        camera_centroid_x,
                                        camera_centroid_y,
                                        scale_factor);
    
        projection_pixel_img1_into_img2(d_homogenic_transformation_negative_disturb,
                                        pixel_pos_img1,
                                        pixel_pos_img2_negative_disturb,
                                        d_depth_image,
                                        camera_focus_x,
                                        camera_focus_y,
                                        camera_centroid_x,
                                        camera_centroid_y,
                                        scale_factor);
        
        int intensity_pos_disturb;
        int intensity_neg_disturb;
        
        if (pixel_pos_img2_positive_disturb[0] < d_image.cols && pixel_pos_img2_positive_disturb[1] < d_image.rows &&
            pixel_pos_img2_positive_disturb[0] >= 0 && pixel_pos_img2_positive_disturb[1] >= 0) {
            
            intensity_pos_disturb =
                (int) d_image(pixel_pos_img2_positive_disturb[1], pixel_pos_img2_positive_disturb[0]);
            
        } else intensity_pos_disturb = -100000;
        
        if (pixel_pos_img2_negative_disturb[0] < d_image.cols && pixel_pos_img2_negative_disturb[1] < d_image.rows &&
            pixel_pos_img2_negative_disturb[0] >= 0 && pixel_pos_img2_negative_disturb[1] >= 0) {
            
            intensity_neg_disturb =
                (int) d_image(pixel_pos_img2_negative_disturb[1], pixel_pos_img2_negative_disturb[0]);
            
        } else intensity_neg_disturb = 100000;
        
        d_jacobian[jacobian_index] = (int2double((intensity_pos_disturb - intensity_neg_disturb)) / (2 * delta));
        
    }
    
}

