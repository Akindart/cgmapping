//
// Created by spades on 07/06/17.
//

#ifndef CGMAPPING_CUBLAS_WRAPPER_UTILS_KERNELS_H
#define CGMAPPING_CUBLAS_WRAPPER_UTILS_KERNELS_H

#define IDX2C(i, j, ld) (((j)*(ld))+(i))

__global__
extern void set_identity_matrix_kernel(double *d_matrix, int n_rows, int n_columns);

#endif //CGMAPPING_CUBLAS_WRAPPER_UTILS_KERNELS_H
