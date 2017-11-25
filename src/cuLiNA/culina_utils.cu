//
// Created by spades on 07/06/17.
//

#include <cuLiNA/culina_utils.cuh>

__host__
cuLiNA::cuLiNA_error_t
cuLiNA::set_Didentity_matrix(double *d_matrix, int n_rows, int n_columns, cudaStream_t *strm) {
    
    if (n_rows != n_columns) {

#ifndef DEBUG
        
        std::cerr << "File: " << __FILE__ << " - ERROR INFO SUPPRESSED - use SET(CMAKE_CXX_FLAGS_DEBUG \"-DDEBUG\") to see it" << std::endl;

#endif
#ifdef DEBUG
        
        std::cerr << std::endl << "###########################################################################"
                  << std::endl << std::endl;
        
        std::cerr << "ERROR HAPPENED FROM WITHIN cuLiNA::" << __FUNCTION__ << std::endl;
        std::cerr << "File: \"" << __FILE__ << "\"." << std::endl;
        std::cerr << "Error between number of rows and columns, they are different." << std::endl;
        std::cerr << "Number of n_rows: " << n_rows << " - Number of columns: " << n_columns << std::endl;
        std::cerr << "Identity matrix is only defined for squared matrices." << std::endl;
        
        std::cerr << std::endl << "###########################################################################"
                  << std::endl;

#endif
        
        return cuLiNA::CULINA_PARAMETERS_MISMATCH;
        
    }
    
    dim3 block_dim;
    dim3 grid_dim;
    
    cuLiNA::compute_kernel_size_for_matrix_operation(n_rows, n_columns, 0, block_dim, grid_dim);
//    std::cout << "block(" << block_dim.x << " ," << block_dim.y << " ," << block_dim.z << ") - ";
//    std::cout << "grid(" << grid_dim.x << " ," << grid_dim.y << " ," << grid_dim.z << ")" << std::endl;
    
    set_identity_matrix_kernel << < block_dim, grid_dim, 0, (strm == NULL ? 0 : *strm) >> > (d_matrix, n_rows, n_columns);
    
    return cuLiNA::CULINA_SUCCESS;
    
}

__host__
cuLiNA::cuLiNA_error_t cuLiNA::set_Ddiagonal_value_matrix(double *d_matrix,
                                                          int n_rows,
                                                          int n_columns,
                                                          double value,
                                                          cudaStream_t *strm) {
    dim3 block_dim;
    dim3 grid_dim;
    
    cuLiNA::compute_kernel_size_for_matrix_operation(n_rows, n_columns, 0, block_dim, grid_dim);
//    std::cout << "block(" << block_dim.x << " ," << block_dim.y << " ," << block_dim.z << ") - ";
//    std::cout << "grid(" << grid_dim.x << " ," << grid_dim.y << " ," << grid_dim.z << ")" << std::endl;
    
    set_diagonal_value_matrix_kernel << < block_dim, grid_dim, 0, (strm == NULL ? 0 : *strm) >> > (d_matrix, n_rows, n_columns, value);
    
    return CULINA_SUCCESS;
}


cuLiNA::cuLiNA_error_t cuLiNA::set_Dzero_matrix(double *d_matrix, int n_rows, int n_columns, cudaStream_t *strm) {
    
    dim3 block_dim;
    dim3 grid_dim;
    
    cuLiNA::compute_kernel_size_for_matrix_operation(n_rows, n_columns, 0, block_dim, grid_dim);
//    std::cout << "block(" << block_dim.x << " ," << block_dim.y << " ," << block_dim.z << ") - ";
//    std::cout << "grid(" << grid_dim.x << " ," << grid_dim.y << " ," << grid_dim.z << ")" << std::endl;
    
    set_zero_matrix_kernel << < block_dim, grid_dim, 0, (strm == NULL ? 0 : *strm) >> > (d_matrix, n_rows, n_columns);
    
    return CULINA_SUCCESS;
}

__host__ cuLiNA::cuLiNA_error_t cuLiNA::compute_kernel_size_for_matrix_operation(int n_rows,
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
                std::cout << "max_block_size[0] = " << max_block_size[0] << std::endl;
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
    
    return cuLiNA::CULINA_SUCCESS;
    
}

__host__ cuLiNA::cuLiNA_error_t cuLiNA::culina_Dsumm(double *d_matrix1,
                                                     bool transpose_m1,
                                                     double alpha,
                                                     int n_rows_m1,
                                                     int n_columns_m1,
                                                     int ld_m1,
                                                     double *d_matrix2,
                                                     bool transpose_m2,
                                                     double beta,
                                                     int n_rows_m2,
                                                     int n_columns_m2,
                                                     int ld_m2,
                                                     double *d_matrix_result,
                                                     double gamma,
                                                     int n_rows_m3,
                                                     int n_columns_m3,
                                                     int ld_m3,
                                                     cudaStream_t *strm) {
    
    if ((transpose_m1 ? n_columns_m1 : n_rows_m1) != (transpose_m2 ? n_columns_m2 : n_rows_m2) &&
        (transpose_m1 ? n_rows_m1 : n_columns_m1) != (transpose_m2 ? n_rows_m2 : n_columns_m2))
        return CULINA_PARAMETERS_MISMATCH;
    
    if ((transpose_m1 ? n_columns_m1 : n_rows_m1) != n_rows_m3 &&
        (transpose_m1 ? n_rows_m1 : n_columns_m1) != n_columns_m3)
        return CULINA_PARAMETERS_MISMATCH;
    
    dim3 block_dim;
    dim3 grid_dim;
    
    cuLiNA::compute_kernel_size_for_matrix_operation(n_rows_m3, n_columns_m3, 0, block_dim, grid_dim);
    
    culina_Dsumm_kernel << < grid_dim, block_dim, 0, (strm == NULL ? 0 : *strm) >> >
        (d_matrix1, transpose_m1, alpha, ld_m1, d_matrix2, transpose_m2, beta, ld_m2, d_matrix_result, gamma, ld_m3, n_rows_m3, n_columns_m3);

    return CULINA_SUCCESS;
    
}

__host__ cuLiNA::cuLiNA_error_t cuLiNA::culina_Ddiagonal_multiplication(double *d_matrix1,
                                                                        bool transpose_m1,
                                                                        double alpha,
                                                                        int n_rows_m1,
                                                                        int n_columns_m1,
                                                                        int ld_m1,
                                                                        double *d_matrix_diag,
                                                                        int n_rows_diag,
                                                                        int n_columns_diag,
                                                                        int ld_diag,
                                                                        double *d_matrix_result,
                                                                        double beta,
                                                                        int n_rows_result,
                                                                        int n_columns_result,
                                                                        int ld_result,
                                                                        cudaStream_t *strm) {
    
    if(transpose_m1){ //xor swap style
        n_rows_m1 ^= n_columns_m1;
        n_columns_m1 ^= n_rows_m1;
        n_rows_m1 ^= n_columns_m1;
    }
    
    if (n_columns_diag != n_rows_diag){
        //std::cout << "3" << std::endl;
        return cuLiNA::cuLiNA_error_t::CULINA_PARAMETERS_MISMATCH;
    }
    if (n_columns_m1 != n_columns_diag) {
        //std::cout << "1 " << n_columns_m1 << "   " << n_columns_diag << std::endl;
        return cuLiNA::cuLiNA_error_t::CULINA_PARAMETERS_MISMATCH;
    }
    if (n_columns_m1 != n_columns_result && n_rows_m1 != n_rows_result){
        //std::cout << "2" << std::endl;
        return cuLiNA::cuLiNA_error_t::CULINA_PARAMETERS_MISMATCH;
    }
    
    if(transpose_m1){ //xor swap style
        n_rows_m1 ^= n_columns_m1;
        n_columns_m1 ^= n_rows_m1;
        n_rows_m1 ^= n_columns_m1;
    }
    
    dim3 block_dim;
    dim3 grid_dim;
    
    cuLiNA::compute_kernel_size_for_matrix_operation(n_rows_m1, n_columns_m1, 0, block_dim, grid_dim);
    
    culina_diagonal_Dmultiplication_kernel << < grid_dim, block_dim, 0, ((strm==NULL)?NULL:*strm) >> >
        (d_matrix1, transpose_m1, alpha, n_rows_m1, n_columns_m1, ld_m1, d_matrix_diag, ld_diag, d_matrix_result, beta);
    
    return cuLiNA::cuLiNA_error_t::CULINA_SUCCESS;
    
}

__device__ cuLiNA::cuLiNA_error_t cuLiNA::culina_in_device_base_changeD(double *d_homogeneous_matrix,
                                                                        bool inverse_transformation,
                                                                        double *d_original_vector,
                                                                        double *d_result_vector) {
    
    if (inverse_transformation) {
        
        d_result_vector[0] = d_homogeneous_matrix[idx2c(0, 0, 4)] * d_original_vector[0] +
            d_homogeneous_matrix[idx2c(1, 0, 4)] * d_original_vector[1] +
            d_homogeneous_matrix[idx2c(2, 0, 4)] * d_original_vector[2];
        
        d_result_vector[1] = d_homogeneous_matrix[idx2c(0, 1, 4)] * d_original_vector[0] +
            d_homogeneous_matrix[idx2c(1, 1, 4)] * d_original_vector[1] +
            d_homogeneous_matrix[idx2c(2, 1, 4)] * d_original_vector[2];
        
        d_result_vector[2] = d_homogeneous_matrix[idx2c(0, 2, 4)] * d_original_vector[0] +
            d_homogeneous_matrix[idx2c(1, 2, 4)] * d_original_vector[1] +
            d_homogeneous_matrix[idx2c(2, 2, 4)] * d_original_vector[2];
        
        d_result_vector[0] += (-1) * d_homogeneous_matrix[idx2c(0, 0, 4)] * d_homogeneous_matrix[idx2c(0, 3, 4)] +
            d_homogeneous_matrix[idx2c(1, 0, 4)] * d_homogeneous_matrix[idx2c(1, 3, 4)] +
            d_homogeneous_matrix[idx2c(2, 0, 4)] * d_homogeneous_matrix[idx2c(2, 3, 4)];
        
        d_result_vector[1] += (-1) * d_homogeneous_matrix[idx2c(0, 1, 4)] * d_homogeneous_matrix[idx2c(0, 3, 4)] +
            d_homogeneous_matrix[idx2c(1, 1, 4)] * d_homogeneous_matrix[idx2c(1, 3, 4)] +
            d_homogeneous_matrix[idx2c(2, 1, 4)] * d_homogeneous_matrix[idx2c(2, 3, 4)];
        
        d_result_vector[2] += (-1) * (d_homogeneous_matrix[idx2c(0, 2, 4)] * d_homogeneous_matrix[idx2c(0, 3, 4)] +
            d_homogeneous_matrix[idx2c(1, 2, 4)] * d_homogeneous_matrix[idx2c(1, 3, 4)] +
            d_homogeneous_matrix[idx2c(2, 2, 4)] * d_homogeneous_matrix[idx2c(2, 3, 4)]);
        
    } else {
        
        d_result_vector[0] = d_homogeneous_matrix[idx2c(0, 0, 4)] * d_original_vector[0] +
            d_homogeneous_matrix[idx2c(0, 1, 4)] * d_original_vector[1] +
            d_homogeneous_matrix[idx2c(0, 2, 4)] * d_original_vector[2] + d_homogeneous_matrix[idx2c(0, 3, 4)];
        
        d_result_vector[1] = d_homogeneous_matrix[idx2c(1, 0, 4)] * d_original_vector[0] +
            d_homogeneous_matrix[idx2c(1, 1, 4)] * d_original_vector[1] +
            d_homogeneous_matrix[idx2c(1, 2, 4)] * d_original_vector[2] + d_homogeneous_matrix[idx2c(1, 3, 4)];
        
        d_result_vector[2] = d_homogeneous_matrix[idx2c(2, 0, 4)] * d_original_vector[0] +
            d_homogeneous_matrix[idx2c(2, 1, 4)] * d_original_vector[1] +
            d_homogeneous_matrix[idx2c(2, 2, 4)] * d_original_vector[2] + d_homogeneous_matrix[idx2c(2, 3, 4)];
        
    }
    
    return cuLiNA::CULINA_SUCCESS;
    
}

__host__
cuLiNA::cuLiNA_error_t cuLiNA::culina_Dskew_matrix3x3_operator(double *d_vector,
                                                                double alpha,
                                                                int n_rows_vector,
                                                                int ld_vector,
                                                                double *d_matrix_result,
                                                                int n_rows_result,
                                                                int n_columns_result,
                                                                int ld_result,
                                                                cudaStream_t *strm) {
    
    if(n_rows_vector != 3) return cuLiNA_error_t::CULINA_PARAMETERS_MISMATCH;
    if(n_rows_result != 3 && n_columns_result != 3) return cuLiNA_error_t::CULINA_PARAMETERS_MISMATCH;
    
    culina_Dskew_matrix3x3_operator_kernel <<< 1, 3, 0, ((strm==NULL)?0:*strm) >>>
        (d_vector, alpha, n_rows_vector, ld_vector, d_matrix_result, n_rows_result, n_columns_result, ld_result);
    
    
    return CULINA_SUCCESS;
}

__host__
cuLiNA::cuLiNA_error_t cuLiNA::culina_Dvector_from_skew_matrix3x3_operator(double *d_skew_matrix,
                                                                           double alpha,
                                                                           int n_rows_matrix,
                                                                           int n_columns_matrix,
                                                                           int ld_matrix,
                                                                           double *d_vector_result,
                                                                           int n_rows_vector,
                                                                           int ld_vector,
                                                                           cudaStream_t *strm) {
    
    if(n_rows_vector != 3) return cuLiNA_error_t::CULINA_PARAMETERS_MISMATCH;
    if(n_rows_matrix != 3 && n_columns_matrix != 3) return cuLiNA_error_t::CULINA_PARAMETERS_MISMATCH;
   
    culina_Dvector_from_skew_matrix3x3_operator_kernel<<< 1, 3, 0, ((strm==NULL)?0:*strm) >>>
        (d_skew_matrix, alpha, n_rows_matrix, n_columns_matrix, ld_matrix, d_vector_result, n_rows_vector, ld_vector);
    
    return CULINA_SUCCESS;

}

__host__
cuLiNA::cuLiNA_error_t cuLiNA::culina_Dblock_assingment(double *d_matrix1,
                                                        bool transpose_m1,
                                                        double alpha,
                                                        int n_rows_m1,
                                                        int n_columns_m1,
                                                        int row_m1_init,
                                                        int columns_m1_init,
                                                        int ld_m1,
                                                        double *d_matrix_result,
                                                        int n_rows_result,
                                                        int n_columns_result,
                                                        int rows_result_init,
                                                        int columns_result_init,
                                                        int ld_result,
                                                        int n_rows,
                                                        int n_columns,
                                                        cudaStream_t *strm) {
    
    int n_rows_interval_result = n_rows;
    int n_columns_interval_result = n_columns;
    
    if(transpose_m1){
        
        n_rows_interval_result ^= n_columns_interval_result;
        n_columns_interval_result ^= n_rows_interval_result;
        n_rows_interval_result ^= n_columns_interval_result;
        
    }
    
    if(n_rows_m1 < row_m1_init+n_rows-1) return cuLiNA_error_t::CULINA_PARAMETERS_MISMATCH;
    if(n_columns_m1 < columns_m1_init+n_columns-1) return cuLiNA_error_t::CULINA_PARAMETERS_MISMATCH;
    if(n_rows_result < rows_result_init+n_rows_interval_result-1) return cuLiNA_error_t::CULINA_PARAMETERS_MISMATCH;
    if(n_columns_result < columns_result_init+n_columns_interval_result-1) return cuLiNA_error_t::CULINA_PARAMETERS_MISMATCH;
    
    if(transpose_m1){
        
        n_rows_interval_result ^= n_columns_interval_result;
        n_columns_interval_result ^= n_rows_interval_result;
        n_rows_interval_result ^= n_columns_interval_result;
        
    }
    
    dim3 block_dim;
    dim3 grid_dim;
    
    cuLiNA::compute_kernel_size_for_matrix_operation(n_rows, n_columns, 0, block_dim, grid_dim);
    
    culina_Dblock_assingment_kernel << < grid_dim, block_dim, 0, ((strm==NULL)?0:*strm) >>>
        (d_matrix1, transpose_m1, alpha, row_m1_init, columns_m1_init, ld_m1, d_matrix_result,
            rows_result_init, columns_result_init, ld_result, n_rows, n_columns);
    
    return CULINA_SUCCESS;
}

__host__
cuLiNA::cuLiNA_error_t cuLiNA::culina_Ddiagonal_to_vector(double *d_matrix1,
                                                          double alpha,
                                                          int n_rows_m1,
                                                          int n_columns_m1,
                                                          int ld_m1,
                                                          double *d_vector_result,
                                                          int rows_result,
                                                          int ld_result,
                                                          cudaStream_t *strm){
    
    if(n_rows_m1 != n_columns_m1) return CULINA_PARAMETERS_MISMATCH;
    if(n_rows_m1 != rows_result) return CULINA_PARAMETERS_MISMATCH;
    
    dim3 block_dim;
    dim3 grid_dim;

    cuLiNA::compute_kernel_size_for_matrix_operation(rows_result, 0, 0, block_dim, grid_dim);
    
    culina_Ddiagonal_to_vector_kernel<<<grid_dim, block_dim, 0 , ((strm==NULL)?0:*strm)>>>
        (d_matrix1, alpha, n_rows_m1, n_columns_m1, ld_m1, d_vector_result, rows_result, ld_result );

    return CULINA_SUCCESS;

}

__host__
cuLiNA::cuLiNA_error_t cuLiNA::culina_Dreduction(double *data, int number_of_elements, double &result, cudaStream_t *strm) {
    
    thrust::plus<double> binary_operator;
    double initial_sum_value = 0;
    
    auto exec = thrust::cuda::par.on((strm==NULL)?NULL:*strm);
    
    result = thrust::reduce(exec,
                            data,
                            data + number_of_elements,
                            initial_sum_value,
                            binary_operator);

    
    return CULINA_SUCCESS;
    
}



//template <typename T, typename Alloc = thrust::device_malloc_allocator<T>>
//cuLiNA::cuLiNA_error_t cuLiNA::vector_resize(thrust::device_vector<T, Alloc> &vec, size_t new_vec_size, const T value) {
//
//    vec.resize(new_vec_size, value);
//
//    return CULINA_SUCCESS;
//
//}

//template cuLiNA::cuLiNA_error_t cuLiNA::vector_resize(thrust::device_vector<double, cuLiNA::culina_matrix_allocator<double> >,
//                                                        uint,
//                                                      const double);