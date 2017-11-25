//
// Created by spades on 07/06/17.
//


#include <cuLiNA/culina_utils_kernels.cuh>
#include <cstdio>

__global__
void set_identity_matrix_kernel(double *d_matrix, int n_rows, int n_columns) {
    
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (i < n_rows && j < n_columns)
        if (i == j) {
            
            d_matrix[idx2c(i, j, n_rows)] = 1.0;
            //printf("%lf \n", d_matrix[IDX2C(i, j, n_rows)]);
            
        } else
            d_matrix[idx2c(i, j, n_rows)] = 0.0;
    
}

__global__
void culina_Dsumm_kernel(double *d_matrix1,
                         bool transpose_m1,
                         double alpha,
                         int n_rows,
                         int n_columns,
                         int ld,
                         double *d_matrix2,
                         bool transpose_m2,
                         double beta,
                         double *d_matrix_result,
                         double gamma) {
    
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (i < n_rows && j < n_columns) {
        
        double d_matrix1_element = 0;
        double d_matrix2_element = 0;
        double d_matrix_result_element = 0;
        
        if (alpha != 0)
            if (transpose_m1) d_matrix1_element = d_matrix1[idx2c(j, i, ld)];
            else d_matrix1_element = d_matrix1[idx2c(i, j, ld)];
        else d_matrix1_element = 0;
        
        if (beta != 0)
            if (transpose_m2) d_matrix2_element = d_matrix2[idx2c(j, i, ld)];
            else d_matrix2_element = d_matrix2[idx2c(i, j, ld)];
        else d_matrix2_element = 0;

        if(gamma != 0)
            d_matrix_result_element = gamma*d_matrix_result[idx2c(i,j,ld)] + d_matrix1_element*alpha + d_matrix2_element*beta;
        else d_matrix_result_element = d_matrix1_element*alpha + d_matrix2_element*beta;

//#if __CUDA_ARCH__ >= 200
//        printf("matrix3[%d, %d] = alpha*matrix1[%d, %d] + beta*matrix2[%d, %d] = %lf*%lf + %lf*%lf \n",
//            i,j,i,i,j,alpha,d_matrix1_element,beta,d_matrix2_element);
//#endif
        
        d_matrix_result[idx2c(i, j, ld)] = d_matrix_result_element;
        
    }
    
}

__global__
void culina_diagonal_Dmultiplication_kernel(double *d_matrix1,
                                            bool transpose_m1,
                                            double alpha,
                                            int n_rows_m1,
                                            int n_columns_m1,
                                            int ld_m1,
                                            double *d_matrix_diag,
                                            int ld_diag,
                                            double *d_matrix_result,
                                            double beta) {
    
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (i < n_rows_m1 && j < n_columns_m1){
    
        int idx_i = i;
        int idx_j = j;
        int ld_r = ld_m1;
        
        if(transpose_m1){ //xor swap procedure
            
            idx_i ^= idx_j;
            idx_j ^= idx_i;
            idx_i ^= idx_j;
            
            ld_r = n_columns_m1;
            
        }
        
        int m1_index = idx2c(i, j, ld_m1);
        int r_index =  idx2c(idx_i, idx_j, ld_r);
        int diag_index = idx2c(idx_j, 0, ld_diag);
    
        double tmp_var = 0;
        
        if(alpha != 0);
            tmp_var = alpha*d_matrix_diag[diag_index]*d_matrix1[m1_index];
        
//        printf("d_matrix_diag[%d, %d] = %lf \n", test_, test_, d_matrix_diag[diag_index]);
//        printf("d_matrix1[%d, %d] = %lf\n", r_i, r_j , d_matrix1[m1_index]);
        
        double tmp_var2 = 0;
        if(beta != 0)
            tmp_var2 = beta*d_matrix_result[m1_index];
        
        d_matrix_result[r_index] = tmp_var + tmp_var2;
//        printf("d_matrix_result[%d, %d] = %lf \n", r_i, r_j, d_matrix_result[r_index]);
    
    
//        printf("d_matrix_result[%d, %d] = d_matrix1[%d, %d]*d_matrix_diag[%d, %d] \n", idx_i, idx_j, i, j, idx_j, idx_j);
    }
    
    
}

__global__
extern void culina_Dskew_matrix3x3_operator_kernel(double *d_vector,
                                                   double alpha,
                                                   int n_rows_vector,
                                                   int ld_vector,
                                                   double *d_matrix_result,
                                                   int n_rows_result,
                                                   int n_columns_result,
                                                   int ld_result){
    
    int i = threadIdx.x;
    
    if (i < n_rows_result ){
        
            int k, l, idx;
        
            k = i;
            l = i+1;
            if(i == 2) l = 0;
        
            idx = i-1;
            if(i == 0) idx = 2;
        
            d_matrix_result[idx2c(i,i, ld_result)] = 0;
            d_matrix_result[idx2c(k,l, ld_result)] = -d_vector[idx]*alpha;
            d_matrix_result[idx2c(l,k, ld_result)] = d_vector[idx]*alpha;
        
    }
    
}

/***TODO create comments for this kernel
 *
 * m2.block(row_m2_init, col_m2_init, n_rows, n_cols) = alpha*op(m1.block(row_m1_init, col_m1_init, n_rows, n_cols))
 *
 * op(.) is the transpose operator used before copy
 *
 * */
__global__
void culina_Dblock_assingment_kernel(double *d_matrix1,
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
                                     int n_columns) {
    
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(i < n_rows && j < n_columns){
    
        int d_matrix_idx = idx2c(row_m1_init+i, columns_m1_init+j, ld_m1);
        int d_result_idx = idx2c(row_result_init+i, columns_result_init+j, ld_result);
        
        if(transpose_m1) d_result_idx = idx2c(row_result_init+j, columns_result_init+i, ld_result);
        
        //printf("d_result_idx = %d - d_matrix_idx = %d \n", d_result_idx, d_matrix_idx );
        
        d_matrix_result[d_result_idx] = d_matrix1[d_matrix_idx]*alpha;
    
    }
    
}

/***TODO create comments for this kernel
 *
 * This kernel copies the weighted diagonal of a squared matrix to a column vector
 *
 * vector = diag(matrix)*alpha
 *
 * */
__global__
void culina_Ddiagonal_to_vector_kernel(double *d_matrix1,
                                       double alpha,
                                       int n_rows_m1,
                                       int n_columns_m1,
                                       int ld_m1,
                                       double *d_vector_result,
                                       int rows_result,
                                       int ld_result) {
    
    
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    
    //printf("%d" , i);
    
    if(i < rows_result && i < n_rows_m1 && j < n_columns_m1){
        
        d_vector_result[idx2c(i,j,ld_result)] = d_matrix1[idx2c(i,i,ld_m1)]*alpha;
    
    }
    
}