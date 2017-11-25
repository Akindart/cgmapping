//
// Created by spades on 07/06/17.
//

#ifndef CGMAPPING_CUBLAS_WRAPPER_UTILS_KERNELS_H
#define CGMAPPING_CUBLAS_WRAPPER_UTILS_KERNELS_H

#define IDX2C(i, j, ld) (((j)*(ld))+(i))

__global__
extern void set_identity_matrix_kernel(double *d_matrix, int n_rows, int n_columns);

__global__
extern void culina_Dsumm_kernel(double *d_matrix1,
                                bool transpose_m1,
                                double alpha,
                                int n_rows,
                                int n_columns,
                                int ld,
                                double *d_matrix2,
                                bool transpose_m2,
                                double beta,
                                double *d_matrix_result,
                                double gamma);

/***TODO create comments for this kernel
 *
 * result = alpha*op(m1)*m2 + beta*result
 *
 * if m1 is transposed than
 *
 * result = alpha*m2*op(m1) + beta*result
 *
 * */
__global__
extern void culina_diagonal_Dmultiplication_kernel(double *d_matrix1,
                                                   bool transpose_m1,
                                                   double alpha,
                                                   int n_rows_m1,
                                                   int n_columns_m1,
                                                   int ld_m1,
                                                   double *d_matrix_diag,
                                                   int ld_diag,
                                                   double *d_matrix_result,
                                                   double beta);


/***TODO create comments for this kernel
 *
 * result = alpha*S(d_vector)
 *
 * where S(.) is the skew matrix operator
 *
 *
 * */
__global__
extern void culina_Dskew_matrix3x3_operator_kernel(double *d_vector,
                                                   double alpha,
                                                   int n_rows_vector,
                                                   int ld_vector,
                                                   double *d_matrix_result,
                                                   int n_rows_result,
                                                   int n_columns_result,
                                                   int ld_result);

/***TODO create comments for this kernel
 *
 * m2.block(row_m2_init, col_m2_init, n_rows, n_cols) = alpha*op(m1.block(row_m1_init, col_m1_init, n_rows, n_cols))
 *
 * op(.) is the transpose operator used before copy
 *
 * */
__global__
extern void culina_Dblock_assingment_kernel(double *d_matrix1,
                                            bool transpose_m1,
                                            double alpha,
                                            int row_m1_init,
                                            int columns_m1_init,
                                            int ld_m1,
                                            double *d_matrix_result,
                                            int row_result_init,
                                            int columns_result_init,
                                            int ld_result,
                                            int n_rows,
                                            int n_columns);


/***TODO create comments for this kernel
 *
 * This kernel copies the weighted diagonal of a squared matrix to a column vector
 *
 * vector = diag(matrix)*alpha
 *
 * */
__global__
extern void culina_Ddiagonal_to_vector_kernel(double *d_matrix1,
                                              double alpha,
                                              int n_rows_m1,
                                              int n_columns_m1,
                                              int ld_m1,
                                              double *d_vector_result,
                                              int rows_result,
                                              int ld_result);

__forceinline__ __device__ int idx2c(int i, int j, int ld) { return (((j) * (ld)) + (i)); };

#endif //CGMAPPING_CUBLAS_WRAPPER_UTILS_KERNELS_H
