//
// Created by spades on 08/06/17.
//

#ifndef CGMAPPING_CUDA_PROPERTIES_H
#define CGMAPPING_CUDA_PROPERTIES_H

#include <cuda_runtime.h>
#include <driver_types.h>
#include <general_utils.h>

class cuda_device_properties {
    
    //TODO implement device_ getter and setter
    static int device_;
    static cudaDeviceProp device_prop_;
 
 public:
    
    cuda_device_properties() {
        
        cudaError_t stat;
        stat = _obtain_from_device_its_properties();
        
        cudaCheckErrors(stat, __FILE__, __FUNCTION__);
        
    };
    
    static cudaDeviceProp &_consult_device_properties();
    static cudaError_t _obtain_from_device_its_properties();
    
};

#endif //CGMAPPING_CUDA_PROPERTIES_H
