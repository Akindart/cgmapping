//
// Created by spades on 12/12/17.
//

#ifndef CULINA_TEMPLATE_MATRIX_H
#define CULINA_TEMPLATE_MATRIX_H

#include <cuda_runtime_api.h>

#include <cuLiNA/culina_error_data_types.h>

#include <general_utils.h>

namespace cuLiNA {
    
    typedef enum {
        
        NOTHING,
        IDENTITY,
        DIAGONAL
        
    } matrix_advanced_initialization_t;
    
    template<typename T>
    class culina_tm { //culina_template_matrix
     
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
        
        virtual inline T _getRawValue(int row, int column) const = 0;
        
        virtual inline matrix_advanced_initialization_t _getMatrix_type() const = 0;
        
        virtual inline void _setMatrix_type(matrix_advanced_initialization_t matrix_type) = 0;
        
        virtual inline cuLiNA::cuLiNA_error_t _setIdentity(cudaStream_t *strm = NULL) = 0;
        
        virtual inline cuLiNA::cuLiNA_error_t _setZero(cudaStream_t *strm = NULL) = 0;
    
        virtual inline cuLiNA::cuLiNA_error_t _setDiagonalValue(T value, cudaStream_t *strm = NULL) = 0;
        
        virtual inline int _allocateMatrixDataMemory()  = 0;
        
        inline bool _isSquare(){
            
            return (this->_getColumns() == this->_getRows());
            
        };
        
        virtual inline  bool _isEmpty() const = 0;
        
        inline bool operator==(const culina_tm<T>& rhs){
    
            if(//!this->_isEmpty() && !rhs._isEmpty() &&
                (this->_getRows() == rhs._getRows()) &&
                    (this->_getColumns() == rhs._getColumns())){
        
                if(this->_getMatrix_type() == DIAGONAL && rhs._getMatrix_type() == DIAGONAL){
            
                    for (int i = 0; i < this->_getRows(); ++i) {
                        for (int j = 0; j < this->_getColumns(); ++j) {
                    
                            if (abs(this->_getRawValue(i, j) - rhs._getRawValue(i, j)) > 0.001) return false;
                    
                        }
                    }
            
                }
        
                if(this->_getMatrix_type() != DIAGONAL && rhs._getMatrix_type() != DIAGONAL) {
            
                    for (int i = 0; i < this->_getRows(); ++i) {
                        for (int j = 0; j < this->_getColumns(); ++j) {
                    
                            if (abs(this->_getRawValue(i, j) - rhs._getRawValue(i, j)) > 0.0000000001) return false;
                    
                        }
                    }
            
                }
        
        
            }
            else{
        
                std::cout << "FAILED" << std::endl;
        
                if(this->_getRows() != rhs._getRows()) std::cout << "Number of rows" << std::endl;
                if(this->_getColumns() != rhs._getColumns()) std::cout << "Number of columns" << std::endl;
                if(this->_getMatrix_type() != rhs._getMatrix_type()) std::cout << "Matrix type" << std::endl;
        
                return false;
        
            }
    
            return true;
            
        };
        
        inline culina_tm<T>& operator=(const culina_tm<T>& rhs){
    
            if (this->_getRows() == rhs._getRows() &&
                this->_getColumns() == rhs._getColumns() &&
                this->_getLeading_dimension() == rhs._getLeading_dimension() &&
                this->_getNumber_of_elements() == rhs._getNumber_of_elements()) {
        
                if (this->_getRawData() == const_cast<cuLiNA::culina_tm<T> *>(&rhs)->_getRawData())
                    return *this;
        
                auto stat = cudaMemcpyAsync((void *) this->_getRawData(),
                                            (const void *) const_cast<cuLiNA::culina_tm<T> *>(&rhs)->_getRawData(),
                                            sizeof(T) * this->_getNumber_of_elements(),
                                            cudaMemcpyDeviceToDevice,
                                            NULL);
        
                cudaCheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
        
            }
    
            return *this;
            
        };
    
        inline cudaError_t _loadData(const culina_tm<T>& rhs){
        
            auto stat = cudaError_t::cudaSuccess;
        
            if (this->_getRows() == rhs._getRows() &&
                this->_getColumns() == rhs._getColumns() &&
                this->_getLeading_dimension() == rhs._getLeading_dimension() &&
                this->_getNumber_of_elements() == rhs._getNumber_of_elements()) {
            
                if (this->_getRawData() == const_cast<cuLiNA::culina_tm<T> *>(&rhs)->_getRawData())
                    return stat;
            
                stat = cudaMemcpyAsync((void *) this->_getRawData(),
                                       (const void *) const_cast<cuLiNA::culina_tm<T> *>(&rhs)->_getRawData(),
                                       sizeof(T) * this->_getNumber_of_elements(),
                                       cudaMemcpyDeviceToDevice,
                                       NULL);
            
                cudaCheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
            
            }
        
            return stat;
        }
        
        inline cudaError_t _loadData(const culina_tm<T>& rhs, cudaStream_t &strm){
    
            auto stat = cudaError_t::cudaSuccess;
            
            if (this->_getRows() == rhs._getRows() &&
                this->_getColumns() == rhs._getColumns() &&
                this->_getLeading_dimension() == rhs._getLeading_dimension() &&
                this->_getNumber_of_elements() == rhs._getNumber_of_elements()) {
        
                if (this->_getRawData() == const_cast<cuLiNA::culina_tm<T> *>(&rhs)->_getRawData())
                    return stat;
    
                stat = cudaMemcpyAsync((void *) this->_getRawData(),
                                       (const void *) const_cast<cuLiNA::culina_tm<T> *>(&rhs)->_getRawData(),
                                       sizeof(T) * this->_getNumber_of_elements(),
                                       cudaMemcpyDeviceToDevice,
                                       strm);
        
                cudaCheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
        
            }
    
            return stat;
        }
        
        
        inline cudaError_t _uploadData(T *h_data, int num_of_elements, cudaStream_t *strm = NULL){
    
            cudaError_t stat;
    
            if(num_of_elements == this->_getNumber_of_elements()) {
                
                stat = cudaMemcpyAsync((void *) this->_getRawData(),
                                       (const void *) h_data,
                                       sizeof(T) * num_of_elements,
                                       cudaMemcpyHostToDevice,
                                       ((strm != NULL) ? (*strm) : 0));
                cudaCheckErrors(stat, __FILE__, __FUNCTION__, 0);
        
            }
            else stat = cudaError_t::cudaErrorUnsupportedLimit;
    
            return stat;
            
        };
    
        /***
         *
         * @param [in] h_data must've been pre-allocated outside this function and also be of the same
         * type of the cuda_matrix it's receiving information from
         *
         * */
        inline int _downloadData(T *h_data, int num_of_elements, cudaStream_t *strm = NULL) const {
    
            cudaError_t stat;
    
            if(num_of_elements == this->_getNumber_of_elements()) {
        
                stat = cudaMemcpyAsync((void *) h_data,
                                       (const void *) const_cast<cuLiNA::culina_tm<T> *>(this)->_getRawData(),
                                       sizeof(T) * num_of_elements,
                                       cudaMemcpyDeviceToHost,
                                       ((strm != NULL) ? (*strm) : 0));
                cudaCheckErrors(stat, __FILE__, __FUNCTION__, 0);
        
            }
            else stat = cudaError_t::cudaErrorUnsupportedLimit;
    
            return stat;
            
        };
        
        virtual void _printMatrix(bool print_matrix_content = true, bool print_matrix_info = false, std::ostream &output_stream = std::cout) = 0;
        
        //virtual int _getTrace(cudaStream_t *strm = NULL)=0;
        
        
    };
    
}
#endif //CULINA_TEMPLATE_MATRIX_H
