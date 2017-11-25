//
// Created by spades on 12/04/18.
//

#ifndef CGMAPPING_CULINA_MATRIX_ALLOCATOR_H
#define CGMAPPING_CULINA_MATRIX_ALLOCATOR_H

#include <thrust/device_vector.h>

namespace cuLiNA {
    
    template<typename T>
    struct culina_matrix_allocator : thrust::device_malloc_allocator<T> {
        
        // shorthand for the name of the base class
        typedef thrust::device_malloc_allocator <T> super_t;
        
        // get access to some of the base class's typedefs
        
        // note that because we inherited from device_malloc_allocator,
        // pointer is actually thrust::device_ptr<T>
        typedef typename super_t::pointer pointer;
        
        typedef typename super_t::size_type size_type;
        
        pointer allocate(size_type n) {
            
            T *dm_pointer;
            
            cudaMallocManaged((void **)&dm_pointer, sizeof(T) * n);
            
            pointer dev_ptr(dm_pointer);
            
            return dev_ptr;
            
        }
        
    };
    
}

#endif //CGMAPPING_CULINA_MATRIX_ALLOCATOR_H
