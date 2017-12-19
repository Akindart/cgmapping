//
// Created by spades on 16/06/17.
//

#ifndef CGMAPPING_CULINA_OPERATIONS_H
#define CGMAPPING_CULINA_OPERATIONS_H

#include <cuLiNA/culina_base_matrix.h>
#include <cuLiNA/cuSOLVER_wrapper/cusolver_wrapper.h>
#include <cuLiNA/cuBLAS_wrapper/cublas_wrapper.h>
#include <cuLiNA/culina_error_data_types.h>
#include <cuLiNA/culina_operation_data_types.h>
#include <cuLiNA/culina_definition.h>
#include <general_utils.h>
#include <fstream>

namespace cuLiNA {
    
    //matrices are now pointers because there will be times when only one matrix is used
    
    /***
     *
     *  culinaOp[cublasOp(A)]
     *
     *
     * */
    //TODO: generate comments, and add option to invert cu_matrix1 or cu_matrix2 before multiplication procedure
    extern cuLiNA::cuLiNA_error_t culina_matrix_Dmultiplication(culina_tm<double> *cu_matrix1,
                                                                culina_tm<double> *cu_matrix2,
                                                                culina_tm<double> *cu_matrix3,
                                                                cuLiNA::culiopD_t &culiopD = culiopD_default);
    
    extern cuLiNA::cuLiNA_error_t culina_Dnorm(culina_tm<double> *cu_matrix1,
                                               double *result,
                                               cuLiNA::culiopD_t &culiopD = culiopD_default);
    
    /***
     *
     *  m3 = alpha*m1 + beta*m2 + gamma*m3
     *
     *  new version
     *
     *  m3 = alpha*op(m1) + beta*op(m2)
     *
     * */
    extern cuLiNA::cuLiNA_error_t culina_matrix_Dsum(culina_tm<double> *cu_matrix1,
                                                     culina_tm<double> *cu_matrix2,
                                                     culina_tm<double> *cu_matrix3,
                                                     cuLiNA::culiopD_t &culiopD = culiopD_default);

    /***
     *
     * This function is a little bit complex, observe that here we are dealing with the following system
     *
     * ((J^T)*W*J)*DELTA_x = -(J^T)*W*data_x
     *
     * an so we can divide the operations as
     *
     * r1 = (J^T)*W
     *
     * r2 = -r1*data_x
     *
     * r3 = r1*J
     *
     * r4 = inv(r3)*r2
     *
     * DELTA_x = r4
     *
     * Where r2 and r3 can run concurrently, which means 3 threads are needed;
     *
     * culiopD_1 serves for r1 and r4, culiopD_2 serves for r2, culiopD_3 serves for r3
     *
     * buffers in each culiopD are used only to store intermediary results, final result is stored
     * directly in delta.
     *
     * */
    
    extern cuLiNA::cuLiNA_error_t culina_Dsolve_gradient_descent_first_order(culina_tm<double> *jacobian,
                                                                             culina_tm<double> *delta,
                                                                             culina_tm<double> *data,
                                                                             culina_tm<double> *weight,
                                                                             cuLiNA::culiopD_t &culiopD_1,
                                                                             cuLiNA::culiopD_t &culiopD_2,
                                                                             cuLiNA::culiopD_t &culiopD_3);
    
    
    /**
    *
    * This function instantiate internal parameters of buffer matrix, one just need to pass the type of buffer
    * one wants to be generated. Observe that buffer matrix has to be at least
    *
    * */
    extern cuLiNA::cuLiNA_error_t culina_Dcreate_buffer(culina_tm<double> &target_matrix,
                                                        culina_tm<double> &buffer,
                                                        cuLiNA_buffer_t buffer_t);
    
    extern cuLiNA::cuLiNA_error_t culina_Dskew_matrix3x3_operator(culina_tm<double> *vector,
                                                                  culina_tm<double> *result_matrix,
                                                                  cuLiNA::culiopD_t &culiopD = culiopD_default);
    
    extern cuLiNA::cuLiNA_error_t culina_Dblock_assignment_operation(culina_tm<double> *cu_matrix,
                                                                     culina_tm<double> *cu_matrix_result,
                                                                     int n_row_m_init,
                                                                     int n_column_m_init,
                                                                     int n_row_result_init,
                                                                     int n_column_result_init,
                                                                     int n_rows,
                                                                     int n_columns,
                                                                     culiopD_t &culiopD = culiopD_default);
    
    extern cuLiNA::cuLiNA_error_t culina_Ddiagonal_to_vector_operation(culina_tm<double> *cu_matrix,
                                                                       culina_tm<double> *cu_vector_result,
                                                                       culiopD_t &culiopD = culiopD_default);
    
    extern cuLiNA::cuLiNA_error_t culina_Dtrace_operation(culina_tm<double> *cu_matrix,
                                                          culina_tm<double> *cu_auxiliar_vector,
                                                          double &result,
                                                          culiopD_t &culiopD = culiopD_default);
    
    
    /**
     *
     * This function receives a host array and matrix parameters alongside an adress to a file that holds matrix information
     * in a matlab style. cu_matrix rows and columns parameters have to have been already initialized.
     *
     * Also this function is slow because it depends on dynamic allocation of host data of the same size of the matrix being
     * passed.
     *
     * Observe that this function does not care for the structure of the file being passed. It assumes that a line in the file
     * is a line of the matrix, the same for columns. Thus if the matrix in the file is smaller than the matrix being passed
     * probably the matrix passed will have several zeros. The orther way around, the matrix in the file will not be fully loaded.
     *
     * Also this function uses OutputIterator thrust::copy (InputIterator first, InputIterator last, OutputIterator result)
     * with copy being made in the host.
     *
     *
    */
    template<typename T, typename Alloc = culina_matrix_allocator<T> >
    cuLiNA::cuLiNA_error_t culina_load_matrix_file(culina_base_matrix<T, Alloc> &cu_matrix,
                                                   std::string &matrix_file_name){
        
        thrust::host_vector<T>  matrix_data((uint) cu_matrix._getNumber_of_elements());
    
        std::fstream matrix_file_stream(matrix_file_name, std::ios_base::in);
    
        T a;
    
        for (int i = 0; i < cu_matrix._getRows(); ++i) {
    
            for (int j = 0; j < cu_matrix._getColumns() && (matrix_file_stream >> a); ++j) {
                
                matrix_data[idx2c(i,j,cu_matrix._getLeading_dimension())] = a;
                
            }
            
        }
        
        cu_matrix._allocateMatrixDataMemory();
        
        cu_matrix._setData(matrix_data);
        
        return CULINA_SUCCESS;
        
    }
    
}

#endif //CGMAPPING_CULINA_OPERATIONS_H
