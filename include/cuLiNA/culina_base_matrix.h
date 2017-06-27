//
// Created by spades on 02/06/17.
//

#ifndef CGMAPPING_CUBLAS_BASE_MATRIX_H
#define CGMAPPING_CUBLAS_BASE_MATRIX_H

#include <thrust/device_vector.h>
#include <cuLiNA/culina_utils.cuh>
#include <cuLiNA/culina_data_types.h>

namespace cuLiNA {
    
    template<typename T>
    class culina_base_matrix {
        
        int rows_;
        int columns_;
        int leading_dimension_;
        int number_of_elements_;
        thrust::device_vector<T> data_;
        thrust::device_vector<T> inverse_;
     
     public:
        
        culina_base_matrix() {
            
            rows_ = 0;
            columns_ = 0;
            leading_dimension_ = 0;
            number_of_elements_ = 0;
            
        };
        
        /***
         *
         * This constructor simply set matrix meta-informations passed through the parameters. Observe that data_ should have its
         * space allocated separately in order to not cause over usage of time during allocation on gpu.
         *
         * @param rows the number of rows that this matrix have
         * @param columns the number of columns that this matrix have
         * @param leading_dimension the number of rows that this matrix have in order to cuBLAS be capable of using the matrix
         *
         *
         * */
        culina_base_matrix(int rows, int columns, int leading_dimension) {
            
            rows_ = rows;
            columns_ = columns;
            leading_dimension_ = leading_dimension;
            number_of_elements_ = rows_ * columns_;
            //data_.reserve((uint) number_of_elements_);
            
            
        };
        
        culina_base_matrix(thrust::device_vector<T> &data, int rows, int columns, int leading_dimension) {
            
            data_ = data;
            rows_ = rows;
            columns_ = columns;
            leading_dimension_ = leading_dimension;
            number_of_elements_ = rows_ * columns_;
            
        };
        
        inline int _getRows() const {
            return rows_;
        }
        
        inline void _setRows(int rows_) {
            culina_base_matrix::rows_ = rows_;
        }
        
        inline int _getColumns() const {
            return columns_;
        }
        
        inline void _setColumns(int columns_) {
            culina_base_matrix::columns_ = columns_;
        }
        
        inline int _getLeading_dimension() const {
            return leading_dimension_;
        }
        
        inline void _setLeading_dimension(int leading_dimension_) {
            culina_base_matrix::leading_dimension_ = leading_dimension_;
        }
        
        inline int _getNumber_of_elements() const {
            return number_of_elements_;
        }
        
        inline void _setNumber_of_elements(int number_of_elements_) {
            culina_base_matrix::number_of_elements_ = number_of_elements_;
        }
        
        inline T *_getRawData() {
            
            return thrust::raw_pointer_cast(data_.data());
            
        }
        
        inline thrust::device_vector<T> &_getData() {
            
            return data_;
            
        }
        
        inline void _setData(thrust::device_vector<T> &data_) {
            
            this->data_ = data_;
            
        }
        
        inline int _allocateMatrixDataMemory() {
            
            this->data_.reserve((uint) this->number_of_elements_);
            
            return 1;
            
        };
        
        inline cuLiNA::cuLiNA_error_t _setIdentity(cudaStream_t *strm = NULL) {
            
            if (this->data_.capacity() == 0)
                this->_allocateMatrixDataMemory();
            
            T *d_matrix = this->_getRawData();
    
            return cuLiNA::set_identity_matrix(d_matrix, rows_, columns_, strm);
            
        };
        
        virtual ~culina_base_matrix() {};
        
    };
    
}

#endif //CGMAPPING_CUBLAS_BASE_MATRIX_H
