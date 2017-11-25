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
        
        d_result_vector[0] = d_homogeneous_matrix[idx2c_cm(0, 0, 4)] * d_original_vector[0] +
            d_homogeneous_matrix[idx2c_cm(1, 0, 4)] * d_original_vector[1] +
            d_homogeneous_matrix[idx2c_cm(2, 0, 4)] * d_original_vector[2];
        
        d_result_vector[1] = d_homogeneous_matrix[idx2c_rm(0, 1, 4)] * d_original_vector[0] +
            d_homogeneous_matrix[idx2c_cm(1, 1, 4)] * d_original_vector[1] +
            d_homogeneous_matrix[idx2c_cm(2, 1, 4)] * d_original_vector[2];
        
        d_result_vector[2] = d_homogeneous_matrix[idx2c_cm(0, 2, 4)] * d_original_vector[0] +
            d_homogeneous_matrix[idx2c_cm(1, 2, 4)] * d_original_vector[1] +
            d_homogeneous_matrix[idx2c_cm(2, 2, 4)] * d_original_vector[2];
        
        d_result_vector[0] += (-1) * d_homogeneous_matrix[idx2c_rm(0, 0, 4)] * d_homogeneous_matrix[idx2c_rm(0, 3, 4)] +
            d_homogeneous_matrix[idx2c_cm(1, 0, 4)] * d_homogeneous_matrix[idx2c_cm(1, 3, 4)] +
            d_homogeneous_matrix[idx2c_cm(2, 0, 4)] * d_homogeneous_matrix[idx2c_cm(2, 3, 4)];
        
        d_result_vector[1] += (-1) * d_homogeneous_matrix[idx2c_rm(0, 1, 4)] * d_homogeneous_matrix[idx2c_rm(0, 3, 4)] +
            d_homogeneous_matrix[idx2c_cm(1, 1, 4)] * d_homogeneous_matrix[idx2c_cm(1, 3, 4)] +
            d_homogeneous_matrix[idx2c_cm(2, 1, 4)] * d_homogeneous_matrix[idx2c_cm(2, 3, 4)];
        
        d_result_vector[2] += (-1) * (d_homogeneous_matrix[idx2c_rm(0, 2, 4)] * d_homogeneous_matrix[idx2c_rm(0, 3, 4)] +
            d_homogeneous_matrix[idx2c_cm(1, 2, 4)] * d_homogeneous_matrix[idx2c_cm(1, 3, 4)] +
            d_homogeneous_matrix[idx2c_cm(2, 2, 4)] * d_homogeneous_matrix[idx2c_cm(2, 3, 4)]);
        
    } else {
        
        d_result_vector[0] = d_homogeneous_matrix[idx2c_cm(0, 0, 4)] * d_original_vector[0] +
            d_homogeneous_matrix[idx2c_cm(0, 1, 4)] * d_original_vector[1] +
            d_homogeneous_matrix[idx2c_cm(0, 2, 4)] * d_original_vector[2] + d_homogeneous_matrix[idx2c_cm(0, 3, 4)];
        
        d_result_vector[1] = d_homogeneous_matrix[idx2c_cm(1, 0, 4)] * d_original_vector[0] +
            d_homogeneous_matrix[idx2c_cm(1, 1, 4)] * d_original_vector[1] +
            d_homogeneous_matrix[idx2c_cm(1, 2, 4)] * d_original_vector[2] + d_homogeneous_matrix[idx2c_cm(1, 3, 4)];
    
        d_result_vector[2] = d_homogeneous_matrix[idx2c_cm(2, 0, 4)] * d_original_vector[0] +
            d_homogeneous_matrix[idx2c_cm(2, 1, 4)] * d_original_vector[1] +
            d_homogeneous_matrix[idx2c_cm(2, 2, 4)] * d_original_vector[2] + d_homogeneous_matrix[idx2c_cm(2, 3, 4)];
        
    }
    
    return;
    
}

__forceinline__ __device__ void projection_space2image(double *pixel_space_pos,
                                                       int *pixel_pos,
                                                       float camera_focus_x,
                                                       float camera_focus_y,
                                                       float camera_centroid_x,
                                                       float camera_centroid_y) {
    
    pixel_pos[1] = (int) (((camera_focus_x * pixel_space_pos[0]) / pixel_space_pos[2]) + camera_centroid_x);
    pixel_pos[0] = (int) (((camera_focus_y * pixel_space_pos[1]) / pixel_space_pos[2]) + camera_centroid_y);
    
}

__forceinline__ __device__ void projection_image2space(int *pixel_pos,
                                                       double *pixel_space_pos,
                                                       float camera_focus_x,
                                                       float camera_focus_y,
                                                       float camera_centroid_x,
                                                       float camera_centroid_y) {
    
    pixel_space_pos[0] = (((((double)pixel_pos[1]) - camera_centroid_x) * pixel_space_pos[2])/camera_focus_x) ;
    pixel_space_pos[1] = (((((double)pixel_pos[0]) - camera_centroid_y) * pixel_space_pos[2])/camera_focus_y );
    
}

__forceinline__ __device__ void projection_pixel_img1_into_img2(double *d_homogenic_transformation,
                                                                int *pixel_pos_img1,
                                                                int *pixel_pos_img2,
                                                                double *d_point_cloud_img1,
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
    pixel_img1_space_pos_wrt_img1[2] = (((double)d_depth_image1(v, u)) / scale_factor);
    
    
    //project pixel (v,u) form img1 into 3D space
    projection_image2space(pixel_pos_img1,
                           pixel_img1_space_pos_wrt_img1,
                           camera_focus_x,
                           camera_focus_y,
                           camera_centroid_x,
                           camera_centroid_y);
    
    d_point_cloud_img1[idx2c_cm(idx2c_cm(v,u, d_depth_image1.rows), 0, d_depth_image1.rows*d_depth_image1.cols)] = pixel_img1_space_pos_wrt_img1[0];
    d_point_cloud_img1[idx2c_cm(idx2c_cm(v,u, d_depth_image1.rows), 1, d_depth_image1.rows*d_depth_image1.cols)] = pixel_img1_space_pos_wrt_img1[1];
    d_point_cloud_img1[idx2c_cm(idx2c_cm(v,u, d_depth_image1.rows), 2, d_depth_image1.rows*d_depth_image1.cols)] = pixel_img1_space_pos_wrt_img1[2];
    
    //writing pixel in 3D world from img1 into pixel em 3D world from img2
    base_changeD(d_homogenic_transformation,
                 false,
                 pixel_img1_space_pos_wrt_img1,
                 pixel_img1_space_pos_wrt_img2);
    
    //putinf pixel in 3D from img2 into img2 frame (v', u')
    projection_space2image(pixel_img1_space_pos_wrt_img2,
                           pixel_pos_img2,
                           camera_focus_x,
                           camera_focus_y,
                           camera_centroid_x,
                           camera_centroid_y);
    
}

__forceinline__ __device__ void projection_pixel_img1_into_img2(double *d_homogenic_transformation,
                                                                int *pixel_pos_img1,
                                                                int *pixel_pos_img1_shared_memory,
                                                                int *pixel_pos_img2,
                                                                unsigned short *d_depth_image1,
                                                                int n_rows_depth_image,
                                                                float camera_focus_x,
                                                                float camera_focus_y,
                                                                float camera_centroid_x,
                                                                float camera_centroid_y,
                                                                int scale_factor) {
    
    double pixel_img1_space_pos_wrt_img1[3];
    double pixel_img1_space_pos_wrt_img2[3] = {0, 0, 0};
    int u, v;
    
    u = pixel_pos_img1_shared_memory[1]; //column index
    v = pixel_pos_img1_shared_memory[0]; //row index
    
    pixel_img1_space_pos_wrt_img1[0] = 0.;
    pixel_img1_space_pos_wrt_img1[1] = 0.;
    pixel_img1_space_pos_wrt_img1[2] = (((double)d_depth_image1[idx2c_cm(v, u, n_rows_depth_image)] )/ scale_factor);
    
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

__global__ void warped_image_calculation_kernel(cv::cuda::PtrStepSzb d_image2,
                                                cv::cuda::PtrStepSzb d_image2_warped,
                                                cv::cuda::PtrStepSzb d_image2_warped_filter,
                                                cv::cuda::PtrStepSz<unsigned short> d_depth_image1,
                                                double *d_point_cloud_img1,
                                                double *d_homogenic_transformation,
                                                float camera_focus_x,
                                                float camera_focus_y,
                                                float camera_centroid_x,
                                                float camera_centroid_y,
                                                unsigned short scale_factor) {
    
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (j < d_image2.cols && i < d_image2.rows) {
    
        d_image2_warped.ptr(i)[j] = 0;
        d_image2_warped_filter.ptr(i)[j] = 0;
        
        int pixel_pos_img1[2] = {i, j};
        int pixel_pos_img2[2] = {1000, 1000};
        //int pixel_pos_img1_shared_memory[2] = {i_prime, j_prime};
    
        float depth_valor = (((float)d_depth_image1(i,j))/scale_factor);
        
        if(depth_valor >= 0.8 && depth_valor <= 3.5)
            projection_pixel_img1_into_img2(d_homogenic_transformation,
                                            pixel_pos_img1,
                                            pixel_pos_img2,
                                            d_point_cloud_img1,
                                            d_depth_image1,
                                            camera_focus_x,
                                            camera_focus_y,
                                            camera_centroid_x,
                                            camera_centroid_y,
                                            scale_factor);
        
        
        if (pixel_pos_img2[0] < d_image2.rows && pixel_pos_img2[1] < d_image2.cols &&
            pixel_pos_img2[0] >= 0 && pixel_pos_img2[1] >= 0) {
    
            d_image2_warped.ptr(i)[j] = d_image2(pixel_pos_img2[0], pixel_pos_img2[1]);
            d_image2_warped_filter.ptr(i)[j] = 255;
            
        }
    
//        printf("d_depth(%d, %d) = %f\n", i, j, depth_valor);
//        printf("d_depth(%d, %d) = %d\n", i, j, d_depth_image1(i,j));
        
    }
    
}

__global__ void pixel_residual_calculation_kernel(cv::cuda::PtrStepSzb d_image1,
                                                  cv::cuda::PtrStepSzb d_image2_warped,
                                                  cv::cuda::PtrStepSzb d_image2_warped_filter,
                                                  double *d_residuals) {
    
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (j < d_image1.cols && i < d_image1.rows) {
        
        int index = idx2c_cm(i, j, d_image2_warped_filter.rows);
    
        int pixel_i_j_img1 = d_image1(i,j);
        int pixel_i_j_img2 = d_image2_warped(i,j);
        
        double tmp1 = int2double(pixel_i_j_img2 - pixel_i_j_img1);
        
        double tmp = (d_image2_warped_filter(i, j)?tmp1:10000);
        
        d_residuals[index] = tmp;
        
        //if(tmp1)
        //printf("residual_%d_%d: %d - %d = %lf \n", i, j, pixel_i_j_img2, pixel_i_j_img1, tmp1);
        //printf("pixel_%d_%d_img1: %d - \n", i, j, pixel_i_j_img1);
        //printf("pixel_%d_%d_img2: %d\n", i, j, d_image2_warped.ptr(i)[j]);
        
    }
    
}

__global__
void warp_jacobian_Dcalculation_kernel(double *d_warp_jacobian,
                                       int warp_jacobian_rows,
                                       int warp_jacobian_cols,
                                       double *d_img1_point_cloud,
                                       int point_cloud_size,
                                       float camera_focus_x,
                                       float camera_focus_y){
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(idx < point_cloud_size){
        
        double point[3] = {
            
            d_img1_point_cloud[idx2c_cm(idx, 0, point_cloud_size)],
            d_img1_point_cloud[idx2c_cm(idx, 1, point_cloud_size)],
            d_img1_point_cloud[idx2c_cm(idx, 2, point_cloud_size)]
        
        };
        
        auto x_sqrd = point[0]*point[0];
        auto y_sqrd = point[1]*point[1];
        auto z_sqrd = point[2]*point[2];
        
        auto xy = point[0]*point[1];
        
        d_warp_jacobian[idx2c_cm(2*idx, 0, warp_jacobian_rows)] = (point[2]?camera_focus_x*(1/point[2]):0);
        d_warp_jacobian[idx2c_cm(2*idx, 1, warp_jacobian_rows)] = 0;
        d_warp_jacobian[idx2c_cm(2*idx, 2, warp_jacobian_rows)] = (point[2]?-camera_focus_x*(point[0]/z_sqrd):0);
        d_warp_jacobian[idx2c_cm(2*idx, 3, warp_jacobian_rows)] = (point[2]?-camera_focus_x*(xy/z_sqrd):0);
        d_warp_jacobian[idx2c_cm(2*idx, 4, warp_jacobian_rows)] = (point[2]?camera_focus_x*(1 + (x_sqrd/z_sqrd)):0);
        d_warp_jacobian[idx2c_cm(2*idx, 5, warp_jacobian_rows)] = (point[2]?-camera_focus_x*(point[1]/point[2]):0);
    
        d_warp_jacobian[idx2c_cm(2*idx+1, 0, warp_jacobian_rows)] = 0;
        d_warp_jacobian[idx2c_cm(2*idx+1, 1, warp_jacobian_rows)] = (point[2]?camera_focus_y*(1/point[2]):0);
        d_warp_jacobian[idx2c_cm(2*idx+1, 2, warp_jacobian_rows)] = (point[2]?-camera_focus_y*(point[1]/z_sqrd):0);
        d_warp_jacobian[idx2c_cm(2*idx+1, 3, warp_jacobian_rows)] = (point[2]?-camera_focus_y*(1 + (y_sqrd/z_sqrd)):0);
        d_warp_jacobian[idx2c_cm(2*idx+1, 4, warp_jacobian_rows)] = (point[2]?camera_focus_y*(xy/z_sqrd):0);
        d_warp_jacobian[idx2c_cm(2*idx+1, 5, warp_jacobian_rows)] = (point[2]?camera_focus_y*(point[0]/point[2]):0);
        
    }
    
    return;
    
}

__global__
void full_jacobian_Dcalculation_kernel(cv::cuda::PtrStepSz<float> warped_img_x_derivative,
                                       cv::cuda::PtrStepSz<float> warped_img_y_derivative,
                                       double *d_warp_jacobian,
                                       int warp_jacobian_rows,
                                       int warp_jacobian_cols,
                                       double *d_jacobian,
                                       int jacobian_rows,
                                       int jacobian_cols){
    
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(i < warped_img_x_derivative.rows && j < warped_img_x_derivative.cols){
        
        auto full_jacobian_row = idx2c_cm(i,j,warped_img_x_derivative.rows);
        
        auto wrpd_img_x_vec_pixel_value = warped_img_x_derivative(i,j);
        auto wrpd_img_y_vec_pixel_value = warped_img_y_derivative(i,j);
        
        auto warp_jacob_0_0 = d_warp_jacobian[idx2c_cm(2*full_jacobian_row, 0, warp_jacobian_rows)];
        auto warp_jacob_0_1 = d_warp_jacobian[idx2c_cm(2*full_jacobian_row, 1, warp_jacobian_rows)];
        auto warp_jacob_0_2 = d_warp_jacobian[idx2c_cm(2*full_jacobian_row, 2, warp_jacobian_rows)];
        auto warp_jacob_0_3 = d_warp_jacobian[idx2c_cm(2*full_jacobian_row, 3, warp_jacobian_rows)];
        auto warp_jacob_0_4 = d_warp_jacobian[idx2c_cm(2*full_jacobian_row, 4, warp_jacobian_rows)];
        auto warp_jacob_0_5 = d_warp_jacobian[idx2c_cm(2*full_jacobian_row, 5, warp_jacobian_rows)];
    
        auto warp_jacob_1_0 = d_warp_jacobian[idx2c_cm(2*full_jacobian_row+1, 0, warp_jacobian_rows)];
        auto warp_jacob_1_1 = d_warp_jacobian[idx2c_cm(2*full_jacobian_row+1, 1, warp_jacobian_rows)];
        auto warp_jacob_1_2 = d_warp_jacobian[idx2c_cm(2*full_jacobian_row+1, 2, warp_jacobian_rows)];
        auto warp_jacob_1_3 = d_warp_jacobian[idx2c_cm(2*full_jacobian_row+1, 3, warp_jacobian_rows)];
        auto warp_jacob_1_4 = d_warp_jacobian[idx2c_cm(2*full_jacobian_row+1, 4, warp_jacobian_rows)];
        auto warp_jacob_1_5 = d_warp_jacobian[idx2c_cm(2*full_jacobian_row+1, 5, warp_jacobian_rows)];
        
        d_jacobian[idx2c_cm(full_jacobian_row, 0, jacobian_rows)] = wrpd_img_x_vec_pixel_value*warp_jacob_0_0 +
                                                                            wrpd_img_y_vec_pixel_value*warp_jacob_1_0;
        
        d_jacobian[idx2c_cm(full_jacobian_row, 1, jacobian_rows)] = wrpd_img_x_vec_pixel_value*warp_jacob_0_1 +
                                                                            wrpd_img_y_vec_pixel_value*warp_jacob_1_1;
        
        d_jacobian[idx2c_cm(full_jacobian_row, 2, jacobian_rows)] = wrpd_img_x_vec_pixel_value*warp_jacob_0_2 +
                                                                            wrpd_img_y_vec_pixel_value*warp_jacob_1_2;
        
        d_jacobian[idx2c_cm(full_jacobian_row, 3, jacobian_rows)] = wrpd_img_x_vec_pixel_value*warp_jacob_0_3 +
                                                                            wrpd_img_y_vec_pixel_value*warp_jacob_1_3;
        
        d_jacobian[idx2c_cm(full_jacobian_row, 4, jacobian_rows)] = wrpd_img_x_vec_pixel_value*warp_jacob_0_4 +
                                                                            wrpd_img_y_vec_pixel_value*warp_jacob_1_4;
        
        d_jacobian[idx2c_cm(full_jacobian_row, 5, jacobian_rows)] = wrpd_img_x_vec_pixel_value*warp_jacob_0_5 +
                                                                            wrpd_img_y_vec_pixel_value*warp_jacob_1_5;
        
    }
    
    return;
    
}

__global__
void image_raw_derivatives_x_y_kernel(cv::cuda::PtrStepSzb d_image,
                                      cv::cuda::PtrStepSz<short> d_img_derivative,
                                      bool x_dev,
                                      bool y_dev) {
    
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(i < d_image.rows && j < d_image.cols && i > 0 && j > 0)
        d_img_derivative(i,j) = (d_image(i-(int)y_dev, j-(int)x_dev) - d_image(i+(int)y_dev, j+(int)x_dev))/((short)2);
    
}

//__global__ void residual_jacobian_calculation_kernel(cv::cuda::PtrStepSzb d_image,
//                                                     cv::cuda::PtrStepSz<unsigned short> d_depth_image,
//                                                     double *d_homogenic_transformation_positive_disturb,
//                                                     double *d_homogenic_transformation_negative_disturb,
//                                                     double *d_jacobian,
//                                                     int var_number,
//                                                     float camera_focus_x,
//                                                     float camera_focus_y,
//                                                     float camera_centroid_x,
//                                                     float camera_centroid_y,
//                                                     float delta,
//                                                     unsigned short scale_factor) {
//
//    /**
//     *
//     * d_jacobian is nx6, where n is the number of pixels in the image
//     *
//     */
//
//
//    int j = blockDim.x * blockIdx.x + threadIdx.x;
//    int i = blockDim.y * blockIdx.y + threadIdx.y;
//
//    if (j < d_image.cols && i < d_depth_image.rows) {
//
//        int jacobian_index = idx2c_cm(i, j, d_image.cols);
//
//        jacobian_index += (var_number/*1 <= var_number <= 6*/ - 1) * d_image.rows * d_image.cols;
//
//        d_jacobian[jacobian_index] = 0;
//
//        int pixel_pos_img1[2] = {j, i};
//        int pixel_pos_img2_positive_disturb[2];
//        int pixel_pos_img2_negative_disturb[2];
//
//        projection_pixel_img1_into_img2(d_homogenic_transformation_positive_disturb,
//                                        pixel_pos_img1,
//                                        pixel_pos_img2_positive_disturb,
//                                        nullptr,
//                                        d_depth_image,
//                                        camera_focus_x,
//                                        camera_focus_y,
//                                        camera_centroid_x,
//                                        camera_centroid_y,
//                                        scale_factor);
//
//        projection_pixel_img1_into_img2(d_homogenic_transformation_negative_disturb,
//                                        pixel_pos_img1,
//                                        pixel_pos_img2_negative_disturb,
//                                        nullptr,
//                                        d_depth_image,
//                                        camera_focus_x,
//                                        camera_focus_y,
//                                        camera_centroid_x,
//                                        camera_centroid_y,
//                                        scale_factor);
//
//        int intensity_pos_disturb;
//        int intensity_neg_disturb;
//
//        if (pixel_pos_img2_positive_disturb[0] < d_image.cols && pixel_pos_img2_positive_disturb[1] < d_image.rows &&
//            pixel_pos_img2_positive_disturb[0] >= 0 && pixel_pos_img2_positive_disturb[1] >= 0) {
//
//            intensity_pos_disturb =
//                (int) d_image(pixel_pos_img2_positive_disturb[1], pixel_pos_img2_positive_disturb[0]);
//
//        } else intensity_pos_disturb = -100000;
//
//        if (pixel_pos_img2_negative_disturb[0] < d_image.cols && pixel_pos_img2_negative_disturb[1] < d_image.rows &&
//            pixel_pos_img2_negative_disturb[0] >= 0 && pixel_pos_img2_negative_disturb[1] >= 0) {
//
//            intensity_neg_disturb =
//                (int) d_image(pixel_pos_img2_negative_disturb[1], pixel_pos_img2_negative_disturb[0]);
//
//        } else intensity_neg_disturb = 100000;
//
//        d_jacobian[jacobian_index] = (int2double((intensity_pos_disturb - intensity_neg_disturb)) / (2 * delta));
//
//    }
//
//}

