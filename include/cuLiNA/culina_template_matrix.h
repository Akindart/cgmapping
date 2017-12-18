//
// Created by spades on 12/12/17.
//

#ifndef CULINA_TEMPLATE_MATRIX_H
#define CULINA_TEMPLATE_MATRIX_H

#include <cuda_runtime_api.h>
#include <cuLiNA/culina_error_data_types.h>

namespace cuLiNA {
    
    typedef enum {
        
        NOTHING,
        IDENTITY,
        DIAGONAL
        
    } matrix_advanced_initialization_t;
    
    template<typename T>
    typedef
    class culina_template_matrix {
     
     public:
        
        virtual inline int _getRows() const = 0;
        
        virtual inline void _setRows(int rows_) = 0;
        
        virtual inline int _getColumns() const = 0;
        
        virtual inline void _setColumns(int columns_) = 0;
        
        virtual inline int _getLeading_dimension() const = 0;
        
        virtual inline void _setLeading_dimension(int leading_dimension_) = 0;
        
        virtual inline int _getNumber_of_elements() const = 0;
        
        virtual inline void _setNumber_of_elements(int number_of_elements_) = 0;
        
        virtual inline T *_getRawData() = 0;
        
        virtual inline matrix_advanced_initialization_t _getMatrix_type() const = 0;
        
        virtual inline void _setMatrix_type(matrix_advanced_initialization_t matrix_type) = 0;
        
        virtual inline int _allocateMatrixDataMemory()  = 0;
        
        virtual inline cuLiNA::cuLiNA_error_t _setIdentity(cudaStream_t *strm = NULL) = 0;
        
        virtual inline bool _isSquare() = 0;
        
        virtual inline cudaError_t _loadData(T *h_data, int num_of_elements, cudaStream_t *strm = NULL) {};
        
        virtual inline int _downloadData(T *h_data) {};
        
        virtual void _printMatrix(bool print_matrix_content = true, bool print_matrix_info = false) = 0;
        
        //virtual int _getTrace(cudaStream_t *strm = NULL)=0;
        
        
    } culina_tm;

}
#endif //CULINA_TEMPLATE_MATRIX_H
