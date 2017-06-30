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

namespace cuLiNA {
    
    //matrices are now pointers because there will be times when only one matrix is used
    
    /***
     *
     *  culinaOp[cublasOp(A)]
     *
     *
     * */
    //TODO: generate function definition, comments, and add option to invert cu_matrix1 or cu_matrix2 before multiplication procedure
    extern cuLiNA::cuLiNA_error_t culina_matrix_Dmultiplication(cuLiNA::culina_base_matrix<double> *cu_matrix1,
                                                                cuLiNA::culina_base_matrix<double> *cu_matrix2,
                                                                cuLiNA::culina_base_matrix<double> *cu_matrix3,
                                                                cuLiNA::culiopD_t& culiopD = culiopD_default);
    
    
    
}

#endif //CGMAPPING_CULINA_OPERATIONS_H
