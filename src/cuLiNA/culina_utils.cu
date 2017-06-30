//
// Created by spades on 07/06/17.
//

#include <cuLiNA/culina_utils.cuh>

__host__ cuLiNA::cuLiNA_error_t
cuLiNA::set_identity_matrix(double *d_matrix, int n_rows, int n_columns, cudaStream_t *strm) {
    
    int max_block_size = cuda_device_properties::_consult_device_properties().maxThreadsPerBlock / 2;
    int max_grid_size[3] = {cuda_device_properties::_consult_device_properties().maxGridSize[0],
                            cuda_device_properties::_consult_device_properties().maxGridSize[1],
                            cuda_device_properties::_consult_device_properties().maxGridSize[2]};
    
    if (n_rows != n_columns){

#ifndef DEBUG
    
        std::cerr << "File: " << __FILE__ << " - ERROR INFO SUPPRESSED - use SET(CMAKE_CXX_FLAGS_DEBUG \"-DDEBUG\") to see it" << std::endl;
        
#endif
#ifdef DEBUG
        
        std::cerr << std::endl << "###########################################################################" << std::endl << std::endl;
        
        std::cerr << "ERROR HAPPENED FROM WITHIN cuLiNA::" <<  __FUNCTION__  << std::endl;
        std::cerr << "File: \"" << __FILE__ << "\"." <<  std::endl;
        std::cerr << "Error between number of rows and columns, they are different." << std::endl;
        std::cerr << "Number of n_rows: " << n_rows << " - Number of columns: " << n_columns << std::endl;
        std::cerr << "Identity matrix is only defined for squared matrices." << std::endl;
        

        std::cerr << std::endl << "###########################################################################" << std::endl;
        
#endif
        
        return cuLiNA::CULINA_PARAMETERS_MISMATCH;
       
    }
    
    
    
    int   c1, c2;
    int leftover_of_c1, leftover_of_c2;
    
    int block_size_x, block_size_y;
    int grid_size_x, grid_size_y;
    
    block_size_x = n_columns;
    block_size_y = n_rows;
    grid_size_x = 1;
    grid_size_y = 1;
    
    c1 = n_columns / max_block_size;
    leftover_of_c1 = n_columns % max_block_size;
    
    c2 = n_rows / max_block_size;
    leftover_of_c2 = n_rows % max_block_size;
    
    if (c1 > 1) {
        
        if (leftover_of_c1) {
            
            block_size_x = n_columns / (c1 + 1);
            
            if (n_columns % (c1 + 1))
                block_size_x++;
            
            grid_size_x = c1 + 1;
            
        } else {
            
            block_size_x = n_columns / c1;
            
            if (n_columns % c1)
                block_size_x++;
            
            grid_size_x = c1;
            
        }
        
    }
    if (c2 > 1) {
        
        if (leftover_of_c2) {
            
            block_size_y = n_rows / (c2 + 1);
            
            if (n_rows % (c2 + 1))
                block_size_y++;
            
            grid_size_y = c2 + 1;
            
        } else {
            
            block_size_y = n_rows / c1;
            
            if (n_rows % c2)
                block_size_y++;
            
            grid_size_y = c2;
            
        }
        
    }
    
    dim3 block_dim(block_size_x, block_size_y, 1);
    dim3 grid_dim(grid_size_x, grid_size_y, 1);
    
   
    
    if(strm == NULL)
        set_identity_matrix_kernel <<< grid_dim, block_dim, 0, NULL >>> (d_matrix, n_rows, n_columns);
    else set_identity_matrix_kernel <<< grid_dim, block_dim, 0, *strm >>> (d_matrix, n_rows, n_columns);
    
    std::cout << "aushuashaus" << std::endl;
    
    return cuLiNA::CULINA_SUCCESS;
    
}