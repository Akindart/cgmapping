//
// Created by spades on 07/06/17.
//

#ifndef CGMAPPING_CUBLAS_WRAPPER_UTILS_H
#define CGMAPPING_CUBLAS_WRAPPER_UTILS_H

#include <iostream>

#include <vector_types.h>

#include <cuLiNA/culina_error_data_types.h>
#include <cuLiNA/culina_utils_kernels.cuh>
#include <cuLiNA/culina_matrix_allocator.h>

#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

#include <cuda_device_properties/gpu_cuda_device_properties.h>

#include <cuda_parsing_helper_in_clion/clion_helper.h>

namespace cuLiNA {
    
    __host__
    extern cuLiNA::cuLiNA_error_t compute_kernel_size_for_matrix_operation(int n_rows,
                                                                           int n_columns,
                                                                           int depth,
                                                                           dim3 &block_dim,
                                                                           dim3 &grid_dim,
                                                                           int *max_number_threads = NULL,
                                                                           int *max_number_blocks = NULL);
    
    __host__
    extern cuLiNA::cuLiNA_error_t set_Didentity_matrix(double *d_matrix,
                                                       int n_rows,
                                                       int n_columns,
                                                       cudaStream_t *strm = NULL);
    
    __host__
    extern cuLiNA::cuLiNA_error_t set_Ddiagonal_value_matrix(double *d_matrix,
                                                             int n_rows,
                                                             int n_columns,
                                                             double value,
                                                             cudaStream_t *strm);
    
    __host__
    extern cuLiNA::cuLiNA_error_t set_Dzero_matrix(double *d_matrix,
                                                       int n_rows,
                                                       int n_columns,
                                                       cudaStream_t *strm = NULL);
    
    __host__
    extern cuLiNA::cuLiNA_error_t culina_Dsumm(double *d_matrix1,
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
                                               cudaStream_t *strm = NULL);
    
    __host__
    extern cuLiNA::cuLiNA_error_t culina_Ddiagonal_multiplication(double *d_matrix1,
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
                                                                  cudaStream_t *strm = NULL);
    
    __host__
    extern cuLiNA::cuLiNA_error_t culina_Dskew_matrix3x3_operator(double *d_vector,
                                                                  double alpha,
                                                                  int n_rows_vector,
                                                                  int ld_vector,
                                                                  double *d_matrix_result,
                                                                  int n_rows_result,
                                                                  int n_columns_result,
                                                                  int ld_result,
                                                                  cudaStream_t *strm = NULL);
    
    __host__
    extern cuLiNA::cuLiNA_error_t culina_Dvector_from_skew_matrix3x3_operator(double *d_skew_matrix,
                                                                              double alpha,
                                                                              int n_rows_matrix,
                                                                              int n_columns_matrix,
                                                                              int ld_matrix,
                                                                              double *d_vector_result,
                                                                              int n_rows_vector,
                                                                              int ld_vector,
                                                                              cudaStream_t *strm = NULL);
    
    __host__
    extern cuLiNA::cuLiNA_error_t culina_Dblock_assingment(double *d_matrix1,
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
                                                               cudaStream_t *strm);
    
    __host__
    extern cuLiNA::cuLiNA_error_t culina_Ddiagonal_to_vector(double *d_matrix1,
                                                             double alpha,
                                                             int n_rows_m1,
                                                             int n_columns_m1,
                                                             int ld_m1,
                                                             double *d_vector_result,
                                                             int rows_result,
                                                             int ld_result,
                                                             cudaStream_t *strm = NULL);
    
    
    __host__
    extern cuLiNA::cuLiNA_error_t culina_Dreduction(double *data, int number_of_elements, double &result, cudaStream_t *strm);
    
    /**
     *
     * This is function is not a general function, here a 4x4 homogeneous matrix is received alongside a 3x1 vector
     * to be transformed by the matrix. Remember that following cuoubleBLAS convertion, here matrices are row-oriented.
     * The result vector is a also a 3x1 one.
     *
     * vec_result = R(homogeneous_matrix)*vec_o + t(homogeneous_matrix)
     *
     * where R(homogeneous_matrix) means the use of the rotation part of the homogeneous matrix and t(homogeneous_matrix)
     * is the tranlational part.
     *
     * The use of the inverse homogeneous matrix makes that the inverse operation is carried out first and then the
     * transformation takes place.
     *
     * */
    __device__ extern cuLiNA::cuLiNA_error_t culina_in_device_base_changeD(double *d_homogeneous_matrix,
                                                                           bool inverse_transformation,
                                                                           double *d_original_vector,
                                                                           double *d_result_vector);
    
    
//    template <typename T, typename Alloc = thrust::device_malloc_allocator<T>>
//    cuLiNA::cuLiNA_error_t vector_resize(thrust::device_vector<T, Alloc>& vec, size_t new_vec_size, const T value = 0){
//
//        vec.resize(new_vec_size, value);
//
//        return CULINA_SUCCESS;
//
//    };
    

}

#endif //CGMAPPING_CUBLAS_WRAPPER_UTILS_H
