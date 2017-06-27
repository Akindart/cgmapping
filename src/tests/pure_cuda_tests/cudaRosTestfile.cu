#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <device_launch_parameters.h>


__global__ void kernelTest() {

    __shared__ int shrd[50];

    //shrd = cudaMalloc((void **) &shrd, 50*sizeof(int));

    //int blockX, blockY, blockZ;
    //int threadX, threadY, threadZ;

    shrd[0] = blockIdx.x;
    shrd[1] = blockIdx.y;
    shrd[2] = blockIdx.z;

    shrd[3] = threadIdx.x;
    shrd[4] = threadIdx.y;
    shrd[6] = threadIdx.z;



//    printf("Block x: %d - y: %d - z: %d\n", blockX, blockY, blockZ);
//    printf("Thread x: %d - y: %d - z: %d\n", threadX, threadY, threadZ);
//    __syncthreads();

    return ;

}

void test(){

    dim3 blockSize(3, 3, 3);
    dim3 kernelSize(3, 3, 3);

    kernelTest<<<kernelSize, blockSize>>>();
    cudaDeviceSynchronize();


}