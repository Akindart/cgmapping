//
// Created by spades on 10/07/17.
//

#ifndef CGMAPPING_GPU_CUDA_DEVICE_PROPERTIES_H
#define CGMAPPING_GPU_CUDA_DEVICE_PROPERTIES_H

#include <cuda_device_properties/cuda_device_properties_getter.h>

class gpu_cuda_device_properties{
    
    static cuda_device_properties_getter device_properties_getter;
    
 public:
    
    static const cudaDeviceProp& _getProperties(){
        
        return device_properties_getter._consult_device_properties();
        
    }
    
    static const cuda_device_properties_getter &getDevice_properties_getter() {
        return device_properties_getter;
    }
    static void setDevice_properties_getter(const cuda_device_properties_getter &device_properties_getter) {
        gpu_cuda_device_properties::device_properties_getter = device_properties_getter;
    }
    
};

#endif //CGMAPPING_CUDA_DEVICE_PROPERTIES_H
