//
// Created by spades on 08/06/17.
//

#include <cuda_device_properties/cuda_device_properties_getter.h>

cudaError_t cuda_device_properties_getter::_obtain_from_device_its_properties(){
    
    if(device_ == -1){
        
        device_ = 0;
        cudaSetDevice(device_);
        
    }
    
    return cudaGetDeviceProperties(&device_prop_, device_);
    
};

cudaDeviceProp& cuda_device_properties_getter::_consult_device_properties(){
    
    if(device_ ==  -1){
        
        cuda_device_properties_getter::_obtain_from_device_its_properties();
        
    }
    
    return device_prop_;

}
int cuda_device_properties_getter::_getDevice() {
    return device_;
}
void cuda_device_properties_getter::_setDevice(int device_) {
    cuda_device_properties_getter::device_ = device_;
};


