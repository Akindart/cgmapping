//
// Created by spades on 02/06/17.
//

#ifndef CGMAPPING_CUBLAS_BASE_MATRIX_H
#define CGMAPPING_CUBLAS_BASE_MATRIX_H

#include <iomanip>

#include <thrust/device_vector.h>

#include <cuLiNA/culina_utils.cuh>
#include <cuLiNA/culina_error_data_types.h>
#include <cuLiNA/culina_matrix_allocator.h>
#include <cuLiNA/culina_template_matrix.h>

#include <cuda_device_runtime_api.h>

#include <cuLiNA/thrust_test.cu>

#define IDX2C(i, j, ld) (((j)*(ld))+(i))

namespace cuLiNA {
    
//    typedef enum {
//
//        NOTHING,
//        IDENTITY,
//        DIAGONAL
//
//    } matrix_advanced_initialization_t;
    
    template<typename T, typename Alloc = culina_matrix_allocator<T> >
    class culina_base_matrix : public culina_tm<T>{
        
        int rows_;
        int columns_;
        int leading_dimension_;
        int number_of_elements_;
        matrix_advanced_initialization_t matrix_type_ = matrix_advanced_initialization_t::NOTHING;
        thrust::device_vector<T, Alloc> data_;
        
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
         * @param mai matrix advanced initialization
         *
         * */
        culina_base_matrix(int rows,
                           int columns,
                           int leading_dimension,
                           matrix_advanced_initialization_t mai = NOTHING) {
            
            rows_ = rows;
            columns_ = columns;
            leading_dimension_ = leading_dimension;
            number_of_elements_ = rows_ * columns_;

            //cudaSetDevice(0);
            
            if (mai == IDENTITY) {
                
                this->_allocateMatrixDataMemory();
                
                this->matrix_type_ = mai;
                
                    for (int i = 0; i < rows_; i++)
                        for (int j = 0; j < columns_; j++)
                            if (i == j)
                                data_[IDX2C(i, j, leading_dimension_)] = 1.0;
                            else data_[IDX2C(i, j, leading_dimension_)] = 0.0;

            } else if (mai == DIAGONAL) {
                
                //diagonal matrices are squared
                if (this->columns_ != this->rows_) {
                    
                    cuLiNA::cuLiNA_error_t culina_stat = cuLiNA_error_t::CULINA_PARAMETERS_MISMATCH;
                    cuLiNA::cuLiNACheckErrors(culina_stat, __FILE__, __FUNCTION__, __LINE__);
                
                }else {
                    
                    this->matrix_type_ = mai;
                
                }
                
                if (!this->data_.empty()) this->data_.clear();
                
                this->data_.reserve((uint) this->rows_);
                
                
                
            } else this->_allocateMatrixDataMemory();
            
            //data_.reserve((uint) number_of_elements_);
            
            
        };
        
        culina_base_matrix(thrust::device_vector<T> &data, int rows, int columns, int leading_dimension) {
            
            //data_.assign(data.cbegin(), data.cend());
            data_.swap(data);
            rows_ = rows;
            columns_ = columns;
            leading_dimension_ = leading_dimension;
            number_of_elements_ = rows_ * columns_;
            
        };
    
        culina_base_matrix(thrust::host_vector<T> &data, int rows, int columns, int leading_dimension) {
        
            data_ = data;
            rows_ = rows;
            columns_ = columns;
            leading_dimension_ = leading_dimension;
            number_of_elements_ = rows_ * columns_;
        
        };
        
        inline int _getRows() const {
            return rows_;
        }
        
        inline void _setRows(int rows_){
            
            if(this->matrix_type_!=DIAGONAL) {
    
                culina_base_matrix::rows_ = rows_;
                culina_base_matrix::number_of_elements_ = culina_base_matrix::rows_ * culina_base_matrix::columns_;
                culina_base_matrix::leading_dimension_ = rows_;
            }
            else {
    
                culina_base_matrix::rows_ = culina_base_matrix::columns_ = rows_;
                culina_base_matrix::number_of_elements_ = culina_base_matrix::rows_ * culina_base_matrix::columns_;
                culina_base_matrix::leading_dimension_ = rows_;
                
            }
        }
        
        inline int _getColumns() const {
            return columns_;
        }
        
        inline void _setColumns(int columns_) {
            
            if(this->matrix_type_!=DIAGONAL) {
             
                this->columns_ = columns_;
                number_of_elements_ = rows_ * this->columns_;
            
            }
        
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
        
        inline T *_getRawData() override {

            return thrust::raw_pointer_cast(data_.data());

        }
    
        inline T _getRawValue(int row, int column) const override {
    
            if(this->matrix_type_ == DIAGONAL)
                return this->data_[IDX2C(row, 0, leading_dimension_)];
    
            return this->data_[IDX2C(row, column, leading_dimension_)];
            
        }
    
        inline thrust::device_vector<T, Alloc> &_getData() {
            
            return data_;
            
        }
        
        /**
         *
         * @data is consumed during the process
         *
         * */
        inline void _setData(thrust::device_vector<T> &data) {
            
            this->data_.swap(data);
            
        }
        
        inline void _setData(thrust::host_vector<T> &data) {
        
            this->data_ = data;
        
        }
    
        inline matrix_advanced_initialization_t _getMatrix_type() const override {
            return matrix_type_;
        }
        
        inline void _setMatrix_type(matrix_advanced_initialization_t matrix_type) {
            culina_base_matrix::matrix_type_ = matrix_type;
        
        }
        
        inline int _allocateMatrixDataMemory() {
            
            if (!this->data_.empty()) this->data_.clear();
    
            if(this->matrix_type_ != matrix_advanced_initialization_t::DIAGONAL) {
            
                this->data_.reserve((uint) this->number_of_elements_);
                //vector_resize<T, Alloc>(this->data_, (uint)this->number_of_elements_, 0);
              
            }
            else this->data_.reserve((uint) this->columns_);
    
            return 1;
            
        };
        
        inline cuLiNA::cuLiNA_error_t _setIdentity(cudaStream_t *strm = NULL) override {
            
            if (this->data_.capacity() == 0)
                this->_allocateMatrixDataMemory();
            
            cuLiNA_error_t stat = CULINA_SUCCESS;
    
            T *d_matrix = this->_getRawData();
            
            stat = cuLiNA::set_Didentity_matrix(d_matrix, rows_, columns_, strm);
            cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
            
            return stat;
            
        };
    
       inline  cuLiNA_error_t _setZero(cudaStream_t *strm = NULL) override {
           
           if (this->data_.capacity() == 0)
               this->_allocateMatrixDataMemory();
    
           cuLiNA_error_t stat = CULINA_SUCCESS;
    
           T *d_matrix = this->_getRawData();
    
           stat = cuLiNA::set_Dzero_matrix(d_matrix, rows_, columns_, strm);
           cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
           return stat;
           
        }
    
        inline cuLiNA::cuLiNA_error_t _setDiagonalValue(T value, cudaStream_t *strm = NULL) override {
    
            if (this->data_.capacity() == 0)
                this->_allocateMatrixDataMemory();
    
            cuLiNA_error_t stat = CULINA_SUCCESS;
    
            T *d_matrix = this->_getRawData();
    
            stat = cuLiNA::set_Ddiagonal_value_matrix(d_matrix, rows_, columns_, value, strm);
            cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
            return stat;
           
       }
        
        inline bool _isEmpty() const override {
           
            return data_.empty();
        
        }
        
        culina_tm<T>& operator=(const culina_tm<T>& rhs) {

           return culina_tm<T>::operator=(rhs);
           
        }
        
        /***
         * Diagonal matrices will return always the diagonal element of the row indicated
         *
         * */
        inline thrust::device_reference<T> operator()(int row, int col = 0) {
            
            if(this->matrix_type_ == DIAGONAL)
                return this->data_[IDX2C(row, 0, leading_dimension_)];
            
            return this->data_[IDX2C(row, col, leading_dimension_)];
            
        };
        
        void _printMatrix(bool print_matrix_content = true, bool print_matrix_info = false, std::ostream &output_stream = std::cout) {
            
            if (this->_getData().capacity() && print_matrix_content) {
    
                for (int i = 0; i < rows_; i++) {
        
                    for (int j = 0; j < columns_; j++) {
            
                        T print_value;
            
//                        if (this->_getMatrix_type() == DIAGONAL && j != i)
//                            print_value = 0;
//                        else
                        if (matrix_type_ == DIAGONAL && j == i) {
                        
                            print_value = this->_getRawData()[IDX2C(i, 0, leading_dimension_)];
    
                            output_stream << std::left << std::setw(15) << print_value;
    
                            continue;
                            
                        }
                        else if(matrix_type_ != DIAGONAL) {
                            print_value = this->_getRawData()[IDX2C(i, j, leading_dimension_)];
    
                            //output_stream << "[" << i << "," << j << "] = " << print_value << "\t";
    
                            output_stream << std::left << std::setw(15) << print_value;
                        }
                    
                    }
    
                    output_stream << std::endl;
        
                }
                
            }
            else if(print_matrix_content)
                output_stream << "Matrix data is not allocated, cant print its content" << std::endl;
    
            if(print_matrix_info){
    
                output_stream << "Rows: " << this->rows_ << std::endl;
                output_stream << "Columns: " << this->columns_ << std::endl;
                output_stream << "Leading dimension: " << this->leading_dimension_ << std::endl;
                output_stream << "Number of elements: " << this->number_of_elements_ << std::endl;
        
            }
            
        }
        
        //virtual int _getTrace(cudaStream_t *strm = NULL)=0;
        
        virtual ~culina_base_matrix() {
        
        
        
        };
        
    };
    
}

#endif //CGMAPPING_CUBLAS_BASE_MATRIX_H
