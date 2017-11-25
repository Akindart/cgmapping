//
// Created by spades on 08/06/17.
//

#ifndef CGMAPPING_CUDA_PROPERTIES_GETTER_H
#define CGMAPPING_CUDA_PROPERTIES_GETTER_H

#include <cuda_runtime.h>
#include <driver_types.h>
#include <general_utils.h>

class cuda_device_properties_getter {
    
    int device_ = -1;
    cudaDeviceProp device_prop_;
    
 public:
    
    cuda_device_properties_getter() {
        
        cudaError_t stat;
        stat = this->_obtain_from_device_its_properties();
        
        cudaCheckErrors(stat, __FILE__, __FUNCTION__);
        
    };
    
    cudaDeviceProp &_consult_device_properties();
    cudaError_t _obtain_from_device_its_properties();
    
    int _getDevice();
    void _setDevice(int device_);
    
};

#endif //CGMAPPING_CUDA_PROPERTIES_H
