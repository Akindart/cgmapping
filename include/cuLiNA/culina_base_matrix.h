//
// Created by spades on 02/06/17.
//

#ifndef CGMAPPING_CUBLAS_BASE_MATRIX_H
#define CGMAPPING_CUBLAS_BASE_MATRIX_H

#include <thrust/device_vector.h>
#include <cuLiNA/culina_utils.cuh>
#include <cuLiNA/culina_error_data_types.h>

#define IDX2C(i, j, ld) (((j)*(ld))+(i))

namespace cuLiNA {
    
    template<typename T>
    struct culina_matrix_allocator : thrust::device_malloc_allocator<T>
    {
    
        // shorthand for the name of the base class
        typedef thrust::device_malloc_allocator<T> super_t;
    
        // get access to some of the base class's typedefs
    
        // note that because we inherited from device_malloc_allocator,
        // pointer is actually thrust::device_ptr<T>
        typedef typename super_t::pointer   pointer;
    
        typedef typename super_t::size_type size_type;
    
        pointer allocate(size_type n)
        {
            
            T* dm_pointer;
            
            cudaMallocManaged(&dm_pointer, sizeof(T)*n);
            
            pointer dev_ptr(dm_pointer);
            
            return dev_ptr;
            
        }
    
    };
    
    typedef enum {
        
        NOTHING,
        IDENTITY,
        DIAGONAL
        
    } matrix_advanced_initialization_t;
    
    template<typename T, typename Alloc = culina_matrix_allocator<T> >
    class culina_base_matrix {
        
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

            cudaSetDevice(0);
            
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
            
            culina_base_matrix::rows_ = rows_;
            culina_base_matrix::number_of_elements_ = culina_base_matrix::rows_*culina_base_matrix::columns_;
            culina_base_matrix::leading_dimension_ = rows_;
            
        }
        
        inline int _getColumns() const {
            return columns_;
        }
        
        inline void _setColumns(int columns_) {
            this->columns_ = columns_;
            number_of_elements_ = rows_*columns_;
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
        
        inline matrix_advanced_initialization_t _getMatrix_type() const {
            return matrix_type_;
        }
        
        inline void _setMatrix_type(matrix_advanced_initialization_t matrix_type) {
            culina_base_matrix::matrix_type_ = matrix_type;
        
        }
        inline int _allocateMatrixDataMemory() {
            
            if (!this->data_.empty()) this->data_.clear();
    
            if(this->matrix_type_ != matrix_advanced_initialization_t::DIAGONAL)
                this->data_.reserve((uint) this->number_of_elements_);
            else this->data_.reserve((uint) this->columns_);
    
            return 1;
            
        };
        
        inline cuLiNA::cuLiNA_error_t _setIdentity(cudaStream_t *strm = NULL) {
            
            if (this->data_.capacity() == 0)
                this->_allocateMatrixDataMemory();
            
            cuLiNA_error_t stat = CULINA_SUCCESS;
    
            T *d_matrix = this->_getRawData();
            
            stat = cuLiNA::set_Didentity_matrix(d_matrix, rows_, columns_, strm);
            
            return stat;
            
        };
        
        inline bool _isSquare(){
            
            return (rows_ == columns_);
            
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
    
        
//        inline int _loadData(T *h_data) {
//
//            cublasStatus_t stat;
//
//            //std::cout << __FUNCTION__ << std::endl;
//
//            data_.
//
//            stat = cublasSetMatrix(culina_base_matrix<T>::_getRows(),
//                                   culina_base_matrix<T>::_getColumns(),
//                                   sizeof(*h_data),
//                                   h_data,
//                                   culina_base_matrix<T>::_getLeading_dimension(),
//                                   culina_base_matrix<T>::_getRawData(),
//                                   culina_base_matrix<T>::_getLeading_dimension());
//
//            cuBLAS_wrapper::cublas_wrapper::_cublasCheckErrors(stat, __FILE__, __FUNCTION__);
//
//            return 1;
//
//        };
//
//        /***
//         *
//         * @param [in] h_data must've been pre-allocated outside this function and also be of the same
//         * type of the cuda_matrix it's receiving information from
//         *
//         * */
//        inline int _downloadData(T *h_data) {
//
//            cublasStatus_t stat;
//
//            stat = cublasGetMatrix(culina_base_matrix<T>::_getRows(),
//                                   culina_base_matrix<T>::_getColumns(),
//                                   sizeof(*h_data),
//                                   culina_base_matrix<T>::_getRawData(),
//                                   culina_base_matrix<T>::_getLeading_dimension(),
//                                   h_data,
//                                   culina_base_matrix<T>::_getLeading_dimension());
//
//            cuBLAS_wrapper::cublas_wrapper::_cublasCheckErrors(stat, __FILE__, __FUNCTION__);
//
//            if (stat != CUBLAS_STATUS_SUCCESS)
//                std::cout << "shit happens when downloading a matrix" << std::endl;
//
//            return 1;
//
//        }
        
        void _printMatrix(bool print_matrix_content = true, bool print_matrix_info = false) {
            
            if (data_.capacity() && print_matrix_content) {
    
                for (int i = 0; i < rows_; i++) {
        
                    for (int j = 0; j < columns_; j++) {
            
                        T print_value;
            
                        if (matrix_type_ == DIAGONAL && j != i)
                            print_value = 0;
                        else if (matrix_type_ == DIAGONAL)
                            print_value = this->data_[IDX2C(i, 0, leading_dimension_)];
                        else print_value = this->data_[IDX2C(i, j, leading_dimension_)];
            
                        std::cout << "[" << i << "," << j << "] = " << print_value << "\t";
            
                    }
        
                    std::cout << std::endl;
        
                }
                
            }
            else if(print_matrix_content)
                std::cout << "Matrix data is not allocated, cant print its content" << std::endl;
    
            if(print_matrix_info){
        
                std::cout << "Rows: " << this->rows_ << std::endl;
                std::cout << "Columns: " << this->columns_ << std::endl;
                std::cout << "Leading dimension: " << this->leading_dimension_ << std::endl;
                std::cout << "Number of elements: " << this->number_of_elements_ << std::endl;
        
            }
            
        }
        
        //virtual int _getTrace(cudaStream_t *strm = NULL)=0;
        
        virtual ~culina_base_matrix() {};
        
    };
    
}

#endif //CGMAPPING_CUBLAS_BASE_MATRIX_H
