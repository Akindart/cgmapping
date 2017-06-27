//
// Created by spades on 07/06/17.
//


#include <cuLiNA/culina_utils_kernels.cuh>
#include <stdio.h>

__global__
void set_identity_matrix_kernel(double *d_matrix, int n_rows, int n_columns) {
    
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (i < n_rows && j < n_columns)
        if (i == j) {
            
            d_matrix[IDX2C(i, j, n_rows)] = 1.0;
            //printf("%lf \n", d_matrix[IDX2C(i, j, n_rows)]);
            
        } else
            d_matrix[IDX2C(i, j, n_rows)] = 0.0;
    
}