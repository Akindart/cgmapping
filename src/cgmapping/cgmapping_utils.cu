//
// Created by spades on 31/08/17.
//

#include <cgmapping/cgmapping_utils.cuh>
#include <opencv2/core/cuda.hpp>

__host__
bool cgmapping::cuda::compute_kernel_size_for_matrix_operation(int n_rows,
                                                               int n_columns,
                                                               int depth,
                                                               dim3 &block_dim,
                                                               dim3 &grid_dim,
                                                               int *max_number_threads,
                                                               int *max_number_blocks) {
    
    int max_block_size[3] = {1, 1, 1};
    int max_grid_size[3] = {1, 1, 1};
    
    bool dim_used[3] = {false, false, false};
    int n_dim_used = 0;
    
    if (n_rows != 0) {
        dim_used[1] = true;
        n_dim_used++;
    }
    if (n_columns != 0) {
        dim_used[0] = true;
        n_dim_used++;
    }
    if (depth != 0) {
        dim_used[2] = true;
        n_dim_used++;
    }
    
    if (max_number_threads == NULL) {

#ifdef CGMAPPING_GPU_CUDA_DEVICE_PROPERTIES_H
        
        if (dim_used[2]) {
            
            if (n_dim_used == 1)
                max_block_size[2] = gpu_cuda_device_properties::_getProperties().maxThreadsDim[2];
            else if (n_dim_used == 2) {
                
                max_block_size[2] = gpu_cuda_device_properties::_getProperties().maxThreadsDim[2] / 4;
                
                if (dim_used[0]) max_block_size[0] = gpu_cuda_device_properties::_getProperties().maxThreadsDim[0] / 16;
                else if (dim_used[1])
                    max_block_size[1] = gpu_cuda_device_properties::_getProperties().maxThreadsDim[1] / 16;
                
            } else if (n_dim_used == 3) {
                
                max_block_size[0] = gpu_cuda_device_properties::_getProperties().maxThreadsDim[0] / 64;
                //std::cout << "max_block_size[0] = " << max_block_size[0] << std::endl;
                max_block_size[1] = gpu_cuda_device_properties::_getProperties().maxThreadsDim[1] / 64;
                max_block_size[2] = gpu_cuda_device_properties::_getProperties().maxThreadsDim[2] / 16;
                
            }
            
        } else {
            
            if (n_dim_used == 2) {
                
                max_block_size[0] = (int) sqrt(gpu_cuda_device_properties::_getProperties().maxThreadsPerBlock);
                max_block_size[1] = (int) sqrt(gpu_cuda_device_properties::_getProperties().maxThreadsPerBlock);
                
            } else if (n_dim_used == 1) {
                
                if (dim_used[0]) max_block_size[0] = gpu_cuda_device_properties::_getProperties().maxThreadsDim[0];
                else if (dim_used[1])max_block_size[1] = gpu_cuda_device_properties::_getProperties().maxThreadsDim[1];
                
            }
            
        }

#endif
#ifndef CGMAPPING_GPU_CUDA_DEVICE_PROPERTIES_H
        
        //this shit is something Ill do only if I need to
        //in this case we set for the worst case possible, in other words, we adjust for the weakest cuda arch known that
        //can run 3 dimension blocks/grids
        if (dim_used[2]) {
        
            if (n_dim_used == 1)
                max_block_size[2] = 64;
            else if (n_dim_used == 2) {
            
                max_block_size[2] = 16;
            
                if (dim_used[0]) max_block_size[0] = 64;
                else if (dim_used[1])
                    max_block_size[1] = 64;
            
            } else if (n_dim_used == 3) {
            
                max_block_size[0] = 16;
                max_block_size[1] = 16;
                max_block_size[2] = 4;
            
            }
        
        } else {
        
            if (n_dim_used == 2) {
            
                max_block_size[0] = 32;
                max_block_size[1] = 32;
            
            } else if (n_dim_used == 1) {
            
                if (dim_used[0]) max_block_size[0] = 1024;
                else if (dim_used[1])max_block_size[1] = 1024;
            
            }
        
        }

#endif
    
    } else
        for (int i = 0; i < 3; ++i)
            max_block_size[i] = max_number_threads[i];
    
    if (max_number_blocks == NULL) {
        
        int size_dim_grid = 0;

#ifdef CGMAPPING_GPU_CUDA_DEVICE_PROPERTIES_H
        
        if (dim_used[2]) {
            
            if (n_dim_used == 1)
                max_grid_size[2] = gpu_cuda_device_properties::_getProperties().maxGridSize[2];
            else if (n_dim_used == 2) {
                
                size_dim_grid = (int) sqrt(gpu_cuda_device_properties::_getProperties().maxGridSize[2]);
                
                max_grid_size[2] = size_dim_grid;
                
                if (dim_used[0]) max_grid_size[0] = size_dim_grid;
                else if (dim_used[1])
                    max_grid_size[1] = size_dim_grid;
                
            } else if (n_dim_used == 3) {
                
                size_dim_grid = (int) cbrt((double) gpu_cuda_device_properties::_getProperties().maxGridSize[1]);
                
                max_grid_size[0] = size_dim_grid;
                max_grid_size[1] = size_dim_grid;
                max_grid_size[2] = size_dim_grid;
                
            }
            
        } else {
            
            if (n_dim_used == 2) {
                
                size_dim_grid = (int) sqrt(gpu_cuda_device_properties::_getProperties().maxGridSize[1]);
                
                max_grid_size[0] = size_dim_grid;
                max_grid_size[1] = size_dim_grid;
                
            } else if (n_dim_used == 1) {
                
                if (dim_used[0]) max_grid_size[0] = gpu_cuda_device_properties::_getProperties().maxGridSize[0];
                else if (dim_used[1]) max_grid_size[1] = gpu_cuda_device_properties::_getProperties().maxGridSize[1];
                
            }
            
        }

#endif
#ifndef CGMAPPING_GPU_CUDA_DEVICE_PROPERTIES_H
        
        //this shit is something Ill do only if I need to
        //in this case we set for the worst case possible, in other words, we adjust for the weakest cuda arch known that
        //can run 3 dimension blocks/grids

        if (dim_used[2]) {
            
            if (n_dim_used == 1)
                max_grid_size[2] = 65535;
            else if (n_dim_used == 2) {
    
                size_dim_grid = (int)sqrt(65535);
                
                max_grid_size[2] = size_dim_grid;
                
                if (dim_used[0]) max_grid_size[0] = size_dim_grid;
                else if (dim_used[1])
                    max_grid_size[1] = size_dim_grid;
                
            } else if (n_dim_used == 3) {
                
                size_dim_grid = (int)__cbrt(65535);
                
                max_grid_size[0] = size_dim_grid;
                max_grid_size[1] = size_dim_grid;
                max_grid_size[2] = size_dim_grid;
                
            }
            
        } else {
            
            if (n_dim_used == 2) {
    
                size_dim_grid = (int)sqrt(65535);
                
                max_grid_size[0] = size_dim_grid;
                max_grid_size[1] = size_dim_grid;
                
            } else if (n_dim_used == 1) {
                
                if (dim_used[0]) max_grid_size[0] = 65535;
                else if (dim_used[1]) max_grid_size[1] = 65535;
                
            }
            
        }

#endif
    
    } else
        for (int i = 0; i < 3; ++i)
            max_grid_size[i] = max_number_blocks[i];
    
    int c1, c2, c3;
    int leftover_of_c1, leftover_of_c2, leftover_of_c3;
    
    int block_size_x, block_size_y, block_size_z;
    int grid_size_x, grid_size_y, grid_size_z;
    
    if (max_block_size[0] > 1) {
        
        c1 = n_columns / max_block_size[0];
        leftover_of_c1 = n_columns % max_block_size[0];
        grid_size_x = 1;
        block_size_x = n_columns;
        
    } else {
        
        c1 = 0;
        leftover_of_c1 = 0;
        grid_size_x = 1;
        block_size_x = 1;
        
    }
    
    if (max_block_size[1] > 1) {
        
        c2 = n_rows / max_block_size[1];
        leftover_of_c2 = n_rows % max_block_size[1];
        grid_size_y = 1;
        block_size_y = n_rows;
        
    } else {
        
        c2 = 0;
        leftover_of_c2 = 0;
        grid_size_y = 1;
        block_size_y = 1;
        
    }
    
    if (max_block_size[2] > 1) {
        
        c3 = depth / max_block_size[2];
        leftover_of_c3 = depth % max_block_size[2];
        grid_size_z = 1;
        block_size_z = depth;
        
    } else {
        
        c3 = 0;
        leftover_of_c3 = 0;
        grid_size_z = 1;
        block_size_z = 1;
        
    }
    
    if (c1 >= 1) {
        
        if (leftover_of_c1) {
            
            block_size_x = n_columns / (c1 + 1);
            
            if (n_columns % (c1 + 1))
                block_size_x++;
            
            grid_size_x = c1 + 1;
            
        } else {
            
            block_size_x = n_columns / c1;
            
            if (n_columns % c1)
                block_size_x++;
            
            grid_size_x = c1;
            
        }
        
    }
    if (c2 >= 1) {
        
        if (leftover_of_c2) {
            
            block_size_y = n_rows / (c2 + 1);
            
            if (n_rows % (c2 + 1))
                block_size_y++;
            
            grid_size_y = c2 + 1;
            
        } else {
            
            block_size_y = n_rows / c2;
            
            if (n_rows % c2)
                block_size_y++;
            
            grid_size_y = c2;
            
        }
        
    }
    
    if (c3 >= 1) {
        
        if (leftover_of_c3) {
            
            block_size_z = depth / (c3 + 1);
            
            if (depth % (c3 + 1))
                block_size_z++;
            
            grid_size_z = c3 + 1;
            
        } else {
            
            block_size_z = depth / c3;
            
            if (depth % c3)
                block_size_z++;
            
            grid_size_z = c3;
            
        }
        
    }
    
    block_dim = dim3((uint) block_size_x, (uint) block_size_y, (uint) block_size_z);
    grid_dim = dim3((uint) grid_size_x, (uint) grid_size_y, (uint) grid_size_z);
    
    return true;
    
}

__host__
void cgmapping::cuda::warped_image_Dcalculation_operation(cv::cuda::GpuMat &img_rgb_t,
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
                                                          cudaStream_t *strm) {
    
    auto stat = cuLiNA::CULINA_SUCCESS;
    
    auto line = __LINE__;
    
    if(img_rgb_t.rows != img_rgb_t_warped.rows ||
        img_rgb_t.cols != img_rgb_t_warped.cols){
        
        stat = cuLiNA::CULINA_PARAMETERS_MISMATCH;
        line = __LINE__;
        
    }
    
    if(img_rgb_t.rows != img_depth_t_minus_1.rows ||
        img_rgb_t.cols != img_depth_t_minus_1.cols){
    
        stat = cuLiNA::CULINA_PARAMETERS_MISMATCH;
        line = __LINE__;
        
    }
    
    if(d_img_t_minus_1_point_cloud._getRows() != img_rgb_t.rows*img_rgb_t.cols ||
        d_img_t_minus_1_point_cloud._getColumns() != 3){
    
        stat = cuLiNA::CULINA_PARAMETERS_MISMATCH;
        line = __LINE__;
        
    }
    
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, line);
    
    if(stat != cuLiNA::CULINA_SUCCESS) return ;
    
    dim3 block_dim;
    dim3 grid_dim;
    
    cgmapping::cuda::compute_kernel_size_for_matrix_operation(img_rgb_t.rows, img_rgb_t.cols, 1, block_dim, grid_dim);
    
//    std::cout <<  img_depth_t_minus_1.type() << std::endl;
    
//    std::cout << "block(" << block_dim.x << " ," << block_dim.y << " ," << block_dim.z << ") - ";
//    std::cout << "grid(" << grid_dim.x << " ," << grid_dim.y << " ," << grid_dim.z << ")" << std::endl;
    
    warped_image_calculation_kernel << < grid_dim, block_dim, 0, ((strm==NULL)?NULL:*strm) >> >
        (img_rgb_t, img_rgb_t_warped, img_rgb_t_warped_filter, img_depth_t_minus_1,
            d_img_t_minus_1_point_cloud._getRawData(), d_homogenic_transformation._getRawData(),
            camera_focus_x, camera_focus_y, camera_centroid_x, camera_centroid_y, scale_factor);
    
}


__host__
void cgmapping::cuda::pixel_residual_Dcalculation_operation(cv::cuda::GpuMat &img_rgb_t_minus_1,
                                                            cv::cuda::GpuMat &img_rgb_t_warped,
                                                            cv::cuda::GpuMat &img_rgb_t_warped_filter,
                                                            cuLiNA::culina_tm<double> &residual_matrix,
                                                            cudaStream_t *strm) {
    
    //And here comes the error verifications, just in case;
    //First let us check if all images and matrices are of the same size;
    
    //Comparing both rgb images
    
    auto stat = cuLiNA::CULINA_SUCCESS;
    
    if(img_rgb_t_warped.cols != img_rgb_t_minus_1.cols ||
        img_rgb_t_warped.rows != img_rgb_t_minus_1.rows){
    
        stat = cuLiNA::CULINA_PARAMETERS_MISMATCH;
        
    }
    
    if(residual_matrix._getNumber_of_elements() != img_rgb_t_minus_1.cols*img_rgb_t_minus_1.rows){
    
        stat = cuLiNA::CULINA_PARAMETERS_MISMATCH;
        
    }
    
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    if(stat != cuLiNA::CULINA_SUCCESS) return ;
    
    dim3 block_dim;
    dim3 grid_dim;
    
    int n_cols, n_rows;
    
    n_cols = img_rgb_t_warped.cols;
    n_rows = img_rgb_t_warped.rows;
    
    cgmapping::cuda::compute_kernel_size_for_matrix_operation(n_rows, n_cols, 1, block_dim, grid_dim);

//    img_depth_t_minus_1.convertTo(img_depth_t_minus_1, CV_64FC1);
//    img_depth_t.convertTo(img_depth_t, CV_64FC1);

//    std::cout << "block(" << block_dim.x << " ," << block_dim.y << " ," << block_dim.z << ") - ";
//    std::cout << "grid(" << grid_dim.x << " ," << grid_dim.y << " ," << grid_dim.z << ")" << std::endl;
    
    
    pixel_residual_calculation_kernel << < grid_dim, block_dim, 0, ((strm==NULL)?NULL:*strm) >> >
        (img_rgb_t_minus_1, img_rgb_t_warped, img_rgb_t_warped_filter, residual_matrix._getRawData());
    
}

__host__
void cgmapping::cuda::warp_jacobian_Dcalculation_operation(cuLiNA::culina_tm<double> &d_warp_jacobian,
                                                           cuLiNA::culina_tm<double> &d_img1_point_cloud,
                                                           cgmapping::rgb_d_camera_model &h_camera_model,
                                                           cv::cuda::Stream &strm) {
    
    auto stat = cuLiNA::CULINA_SUCCESS;
    
    auto line = __LINE__;
    
    auto tmp_strm = cv::cuda::StreamAccessor::getStream(strm);
    
    if(d_img1_point_cloud._getRows()*2 != d_warp_jacobian._getRows()){
        
        stat = cuLiNA::CULINA_PARAMETERS_MISMATCH;
        line = __LINE__;
    }
    
    if(d_img1_point_cloud._getColumns() != 3){
        
        stat = cuLiNA::CULINA_PARAMETERS_MISMATCH;
        line = __LINE__;
        
    }
    
    if(d_warp_jacobian._getColumns() != 6) {
     
        stat = cuLiNA::CULINA_PARAMETERS_MISMATCH;
        line = __LINE__;
        
    }
    
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    if(stat != cuLiNA::CULINA_SUCCESS) return ;
    
    dim3 block_dim;
    dim3 grid_dim;
    
    cgmapping::cuda::compute_kernel_size_for_matrix_operation(1, d_img1_point_cloud._getRows(), 1, block_dim, grid_dim);
    
//    std::cout << "block(" << block_dim.x << " ," << block_dim.y << " ," << block_dim.z << ") - ";
//    std::cout << "grid(" << grid_dim.x << " ," << grid_dim.y << " ," << grid_dim.z << ")" << std::endl;
    
    warp_jacobian_Dcalculation_kernel << < grid_dim, block_dim, 0, tmp_strm >> >
        (d_warp_jacobian._getRawData(), d_warp_jacobian._getRows(),
            d_warp_jacobian._getColumns(), d_img1_point_cloud._getRawData(),
            d_img1_point_cloud._getRows(), h_camera_model._getFocus_x(),
            h_camera_model._getFocus_y());
    
    
    
}

__host__
void cgmapping::cuda::full_jacobian_Dcalculation_operation(cv::cuda::GpuMat &warped_img_x_derivative,
                                                           cv::cuda::GpuMat &warped_img_y_derivative,
                                                           cuLiNA::culina_tm<double> &d_warp_jacobian,
                                                           cuLiNA::culina_tm<double> &d_jacobian,
                                                           cv::cuda::Stream &strm) {
    
    auto stat = cuLiNA::CULINA_SUCCESS;
    
    auto line = __LINE__;
    
    auto tmp_strm = cv::cuda::StreamAccessor::getStream(strm);
    
    if(warped_img_x_derivative.rows != warped_img_y_derivative.rows ||
        warped_img_x_derivative.cols != warped_img_y_derivative.cols){
    
        stat = cuLiNA::CULINA_PARAMETERS_MISMATCH;
        line = __LINE__;
        
    }
    
    if(d_warp_jacobian._getRows() != warped_img_x_derivative.rows*warped_img_x_derivative.cols*2 ||
        d_warp_jacobian._getColumns() != 6){
    
        stat = cuLiNA::CULINA_PARAMETERS_MISMATCH;
        line = __LINE__;
        
    }
    
    if(d_jacobian._getRows() != warped_img_x_derivative.rows*warped_img_x_derivative.cols ||
        d_jacobian._getColumns() != 6){
        
        stat = cuLiNA::CULINA_PARAMETERS_MISMATCH;
        line = __LINE__;
        
    }
    
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    if(stat != cuLiNA::CULINA_SUCCESS) return ;
    
    dim3 block_dim;
    dim3 grid_dim;
    
    cgmapping::cuda::compute_kernel_size_for_matrix_operation(warped_img_x_derivative.rows,
                                                              warped_img_x_derivative.cols,
                                                              1,
                                                              block_dim,
                                                              grid_dim);
    
//    std::cout << "block(" << block_dim.x << " ," << block_dim.y << " ," << block_dim.z << ") - ";
//    std::cout << "grid(" << grid_dim.x << " ," << grid_dim.y << " ," << grid_dim.z << ")" << std::endl;
    
    full_jacobian_Dcalculation_kernel<< < grid_dim, block_dim, 0, tmp_strm >> >
        (warped_img_x_derivative, warped_img_y_derivative, d_warp_jacobian._getRawData(),
            d_warp_jacobian._getRows(), d_warp_jacobian._getColumns(), d_jacobian._getRawData(),
            d_jacobian._getRows(), d_jacobian._getColumns());
    
}


//__host__
//void cgmapping::cuda::residual_jacobian_calculation(cv::cuda::GpuMat &img_rgb_t_minus_1,
//                                                    cv::cuda::GpuMat &img_rgb_t,
//                                                    cv::cuda::GpuMat &img_depth_t_minus_1,
//                                                    cv::cuda::GpuMat &img_depth_t,
//                                                    cuLiNA::culina_tm<double> &homogenic_transformation_positive_disturb,
//                                                    cuLiNA::culina_tm<double> &homogenic_transformation_negative_disturb,
//                                                    cuLiNA::culina_tm<double> &jacobian,
//                                                    int var_number,
//                                                    float camera_focus_x,
//                                                    float camera_focus_y,
//                                                    float camera_centroid_x,
//                                                    float camera_centroid_y,
//                                                    float delta,
//                                                    int scale_factor,
//                                                    cudaStream_t *strm) {
//
//    //And here comes the error verifications, just in case;
//    //First let us check if all images and matrices are of the same size;
//
//    //Comparing both rgb images
//
//    assert(img_rgb_t.cols == img_rgb_t_minus_1.cols);
//    assert(img_rgb_t.rows == img_rgb_t_minus_1.rows);
//
//    //Comparing both depth images
//
//    assert(img_depth_t.cols == img_depth_t_minus_1.cols);
//    assert(img_depth_t.rows == img_depth_t_minus_1.rows);
//
//    //Now assuming everything went corretly we can compare rgb with depth image
//
//    assert(img_depth_t.cols == img_rgb_t.cols);
//    assert(img_depth_t.rows == img_rgb_t.rows);
//
//    //And now simply compare any image with the size of the residual matrix
//
//    //assert(residual_matrix._getRows() == img_rgb_t.cols*img_rgb_t.rows);
//
//    assert(jacobian._getRows() == img_rgb_t.cols * img_rgb_t.rows);
//    assert(var_number <= jacobian._getColumns());
//
//    dim3 block_dim;
//    dim3 grid_dim;
//
//    int n_cols, n_rows;
//
//    n_cols = img_rgb_t.cols;
//    n_rows = img_rgb_t.rows;
//
//    cgmapping::cuda::compute_kernel_size_for_matrix_operation(n_rows, n_cols, 1, block_dim, grid_dim);
//
////    img_depth_t_minus_1.convertTo(img_depth_t_minus_1, CV_64FC1);
////    img_depth_t.convertTo(img_depth_t, CV_64FC1);
//
////    std::cout << "block(" << block_dim.x << " ," << block_dim.y << " ," << block_dim.z << ") - ";
////    std::cout << "grid(" << grid_dim.x << " ," << grid_dim.y << " ," << grid_dim.z << ")" << std::endl;
//
//    residual_jacobian_calculation_kernel << < grid_dim, block_dim, 0, ((strm==NULL)?NULL:*strm) >> >
//        (img_rgb_t, img_depth_t_minus_1, homogenic_transformation_negative_disturb._getRawData(),
//            homogenic_transformation_positive_disturb._getRawData(), jacobian._getRawData(), var_number, camera_focus_x,
//            camera_focus_y, camera_centroid_x, camera_centroid_y, delta, scale_factor);
//
//    return;
//
//}

__host__
long cgmapping::cuda::count_valid_data(cuLiNA::culina_tm<double> &data, cudaStream_t *stream) {
    
    auto lambda_is_smaller_predicate = [] __host__ __device__(const double &x) -> bool { return (abs(x) < 1000); };
    
    auto exec = thrust::cuda::par.on(((stream==NULL)?NULL:*stream));
    
    return thrust::count_if(exec,
                            data._getRawData(),
                            data._getRawData() + data._getNumber_of_elements(),
                            lambda_is_smaller_predicate);
    
};

__host__
double cgmapping::cuda::variance_t_student_step_Dcalculation_operation(cuLiNA::culina_tm<double> &data,
                                                                       double degrees_of_freedom,
                                                                       double variance_k_minus_1,
                                                                       int number_of_valid_data,
                                                                       cudaStream_t *stream) {
    
    auto lambda_unary_operator =
        [degrees_of_freedom, variance_k_minus_1]
            __device__ __device__(const double &x) -> double {
            
            if (abs(x) > 1000)
                return 0.0;
            
            auto tmp1 = (x*x);
            auto tmp2 = (tmp1 / variance_k_minus_1);
            
            return tmp1 * ((degrees_of_freedom + 1)
                            / (degrees_of_freedom + tmp2));
            
        };
    
    thrust::plus<double> binary_operator;
    double initial_sum_value = 0;
    
    double result;
    
    auto exec = thrust::cuda::par.on(((stream==NULL)?NULL:*stream));
 
    result = thrust::transform_reduce(exec,
                                      data._getRawData(),
                                      data._getRawData() + data._getNumber_of_elements(),
                                      lambda_unary_operator,
                                      initial_sum_value,
                                      binary_operator);
    
    return (result/number_of_valid_data);
    
};

__host__
void cgmapping::cuda::weight_matrix_t_student_Dcalculation_operation(cuLiNA::culina_tm<double> &data,
                                                                     cuLiNA::culina_tm<double> &weight_matrix,
                                                                     double degrees_of_freedom,
                                                                     double variance,
                                                                     cudaStream_t *stream) {
    
    assert(data._getNumber_of_elements()*data._getNumber_of_elements() == weight_matrix._getNumber_of_elements());
    
    auto lambda_weight_function =
        [degrees_of_freedom, variance]
            __host__ __device__(const double &x) -> const double {
            
            if(abs(x) > 1000) return 0;
            
            auto tmp = ((x*x)/ variance);
            
            return (degrees_of_freedom + 1)
                    / (degrees_of_freedom + tmp);
            
        };
    
    auto exec = thrust::cuda::par.on(((stream==NULL)?NULL:*stream));
    
    thrust::transform(exec,
                      data._getRawData(),
                      data._getRawData() + data._getNumber_of_elements(),
                      weight_matrix._getRawData(),
                      lambda_weight_function);
    
    return;
}
