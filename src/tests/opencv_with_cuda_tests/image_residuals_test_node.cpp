//
// Created by spades on 28/08/17.
//

#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cv.hpp>

#include <cuda_device_properties/gpu_cuda_device_properties.h>
#include <cgmapping/cgmapping_utils.cuh>
#include <cgmapping/cgmapping_utils.h>
#include <cgmapping/timer.h>
#include <cuLiNA/culina_matrix.h>
#include <cuLiNA/culina_definition.h>
#include <cuLiNA/culina_operations.h>
#include <opencv2/core/cuda.hpp>


//#include <thrust/device_vector.h>
//#include <thrust/extrema.h>
//#include <thrust/pair.h>
//#include <algorithm>



using namespace cv;

int main(int argc, char **argv) {
    
    cuBLAS_wrapper::cublas_wrapper::_start_cublas_handle_wrapper();
    cuSOLVER_wrapper::cusolver_wrapper::_start_cusolverDn_handle_wrapper();
 
    cv::Mat img_rgb_t_minus_1, img_rgb_t;
    cv::Mat img_depth_t_minus_1, img_depth_t(480, 640, CV_16U);

    img_rgb_t_minus_1 = imread(
        "/home/spades/kinetic_ws/src/cgmapping/datasets/rgbd_dataset_freiburg1_desk/rgb/1305031453.359684.png",
        CV_LOAD_IMAGE_GRAYSCALE);
    img_rgb_t = imread(
        "/home/spades/kinetic_ws/src/cgmapping/datasets/rgbd_dataset_freiburg1_desk/rgb/1305031453.391690.png",
        CV_LOAD_IMAGE_GRAYSCALE);

    img_depth_t_minus_1 = imread(
        "/home/spades/kinetic_ws/src/cgmapping/datasets/rgbd_dataset_freiburg1_desk/depth/1305031453.374112.png",
        CV_16U);
    img_depth_t = imread(
        "/home/spades/kinetic_ws/src/cgmapping/datasets/rgbd_dataset_freiburg1_desk/depth/1305031453.404816.png",
        CV_16U);

    cv::Mat img_diff(480, 640, IMREAD_GRAYSCALE);
    cv::Mat img_diff1(480, 640, IMREAD_GRAYSCALE);
//
//    cv::subtract(img_rgb_t, img_rgb_t_minus_1, img_diff1);
    //img_diff1 = img_rgb_t - img_rgb_t_minus_1;



//    std::cout << "Pixel at por (100,100) = " << int(img_depth_t.at<unsigned short>(100, 100)) << std::endl;
//    std::cout << "Img type = " << img_depth_t.type() << std::endl;
//
//    unsigned char *imgData;
//    unsigned short *imgDepthData;
//
//    imgData = img_rgb_t.data;
//    imgDepthData = (unsigned short *) img_depth_t.data;
//
//    std::cout << "Pixel at por (100,100) = " << (int) (imgData[640 * 100 + 100] - imgData[640 * 330 + 261])
//              << std::endl;
//
//    std::cout << "Pixel value from matrix = " << img_depth_t.at<unsigned short>(100, 100) << std::endl;
//    std::cout << "Pixel value from array = " << imgDepthData[640 * 100 + 100] << std::endl;

//    cv::namedWindow("Subtraction test", IMREAD_GRAYSCALE);
//
//    imshow("Subtraction test", img_diff);
//
//    cv::namedWindow("Subtraction test 2", IMREAD_GRAYSCALE);
//
//    imshow("Subtraction test 2", img_diff1);
//
//    std::cout << "hello dumb bastards!" << std::endl;
    
    cv::cuda::GpuMat d_img1, d_img2, d_depth1, d_depth2;

    d_img1.upload(img_rgb_t_minus_1);
    d_img2.upload(img_rgb_t);
    d_depth1.upload(img_depth_t_minus_1);
    d_depth2.upload(img_depth_t);

    const int size_residuals = 480 * 640;

    cuLiNA::culina_matrix4d homogenic_matrix(cuLiNA::IDENTITY);
    cuLiNA::culina_matrix4d homogenic_matrix2(cuLiNA::IDENTITY);
    cuLiNA::culina_matrix<double, size_residuals, 1> residual_matrix;
    cuLiNA::culina_matrix<double, size_residuals, 6> jacobian_matrix;
    
    
    
    cuLiNA::culina_matrix<double, size_residuals, 1> residual_weight_matrix;

    cuLiNA::culina_matrix4d diag_matrix(cuLiNA::DIAGONAL);
    cuLiNA::culina_matrix4d some_test_matrix(cuLiNA::IDENTITY);

    diag_matrix(0,0) = 0.25;
    diag_matrix(1,1) = 0.5;
    diag_matrix(2,2) = 1;
    diag_matrix(3,3) = 2;

    for (int j = 0; j < 4; ++j) {


        for (int k = 0; k < 4; ++k) {

            if(k!=j)
                some_test_matrix(j,k) = 2;

        }


    }

    cudaStream_t streams[3];

    cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&streams[2], cudaStreamNonBlocking);

    //cgmapping::Timer tmr;

    //tmr.reset();
    
    
    cgmapping::cuda::pixel_residual_calculation(d_img1,
                                                d_img2,
                                                d_depth1,
                                                d_depth2,
                                                homogenic_matrix,
                                                residual_matrix,
                                                525.,
                                                525.,
                                                319.5,
                                                239.5,
                                                5000
        ,&streams[0]
    );
    
    cudaDeviceSynchronize();
    
    double variance_k = cgmapping::calculate_standard_deviation_t_student(residual_matrix, 0.49, 1, 0.00001, &streams[2]);
    cgmapping::cuda::define_data_weight_t_student(residual_matrix, residual_weight_matrix, 0.49, variance_k, &streams[2]);
    //double t1 = tmr.elapsed();

    homogenic_matrix(1, 3) = 0.01;
    homogenic_matrix2(1, 3) = -0.01;

    cgmapping::cuda::residual_jacobian_calculation(d_img1,
                                                   d_img2,
                                                   d_depth1,
                                                   d_depth2,
                                                   homogenic_matrix,
                                                   homogenic_matrix2,
                                                   jacobian_matrix,
                                                   2,
                                                   525.,
                                                   525.,
                                                   319.5,
                                                   239.5,
                                                   0.01,
                                                   5000
        ,&streams[1]
    );

    homogenic_matrix(1, 3) = 0;
    
    
    
    //cv::Mat sub_test_2(480, 640, CV_64FC1);

    //int sum = 0;

//    for (int l = 0; l < jacobian_matrix._getRows(); ++l) {
//
//        if(abs(jacobian_matrix(l, 1)) < 1000)
//            sum++;
//
//    }
//
//    sum = 0;
    
//    for (int j = 0; j < img_rgb_t.rows ; ++j) {
//
//        for (int k = 0; k < img_rgb_t.cols; ++k) {
//
//            if(jacobian_matrix(d_img1.rows * k + j, 0) != 0 && abs(jacobian_matrix(d_img1.rows * k + j, 0)) < 1000)
//                sub_test_2.at<double>(j,k) = jacobian_matrix(d_img1.rows * k + j, 0);
//            else sum++;
//
//        }
//
//    }

    
//    std::cout << "Max: " << max << std::endl;
//    std::cout << "Min: " << min << std::endl;
//
////    thrust::pair<thrust::device_vector<double>::iterator,thrust::device_vector<double>::iterator> tuple;
////    tuple = thrust::minmax_element(residual_matrix._getData().begin(), residual_matrix._getData().end());
//
    //cv::Mat sub_test_2_8u;
//
//    double scale_factor = 255/(max-min);
//    double sum_factor = (-1)*(((min*255)/(max-min)) + 0);
//
//    sub_test_2.convertTo(sub_test_2_8u, CV_8UC1);
//
//    cv::namedWindow("Subtraction gpu test", CV_8UC1);
//
//    imshow("Subtraction gpu test", sub_test_2_8u);
//
//    while (true) {
//
//        int c;
//        c = waitKey(10);
//
//        if ((char) c == 27) break;
//
//    }
//
//    destroyAllWindows();
    
//
//
//    long sum2 = 2;
//
//    sum2 = cgmapping::cuda::count_valid_data(residual_matrix, nullptr);
//
//
//
//    std::cout << "SUM thrust: " << sum2 << std::endl;
    
    
//    std::cout << "variance k thrust: " << variance_k << std::endl;
    
//    std::cout << "time of usage: " << t1*1000000 << "[us]"  << std::endl;


//    for (int j = 0; j < residual_weight_matrix._getNumber_of_elements(); ++j) {
//
//        if(residual_weight_matrix(j, 0) != 0)
//        std::cout << residual_weight_matrix(j, 0) << std::endl;
//
//    }
    
    
//    std::cout << "haha" << std::endl;
//
//

    
    
    //dimensions variables
    int m = 6;
    int n = 160*120;
    
    cuLiNA::culiopD_t m_culiopD, culiopD_1, culiopD_2, culiopD_3;
    
    culiopD_1.workspace = new cuLiNA::culina_matrix<double>();
    culiopD_2.workspace = new cuLiNA::culina_matrix<double>();
    culiopD_3.workspace = new cuLiNA::culina_matrix<double>();
    
    culiopD_2.workspace->_setRows(m);
    culiopD_2.workspace->_setColumns(1);
    culiopD_2.workspace->_allocateMatrixDataMemory();
    
    culiopD_3.workspace->_setRows(m);
    culiopD_3.workspace->_setColumns(m);
    culiopD_3.workspace->_allocateMatrixDataMemory();
    
    cuLiNA::culina_Dcreate_buffer(*(culiopD_3.workspace), *(culiopD_1.workspace), cuLiNA::cuLiNA_buffer_t::GEQRF_BUFFER);
    cudaCheckErrors(cudaMallocManaged(&culiopD_1.d_TAU, sizeof(double) * m), __FILE__, __FUNCTION__, 0);
    cudaCheckErrors(cudaMallocManaged(&culiopD_1.dev_info, sizeof(int)), __FILE__, __FUNCTION__, 0);
    
    cuLiNA::culina_matrix<double> data;
    cuLiNA::culina_matrix<double> jacobian;
    cuLiNA::culina_matrix<double> delta;
    cuLiNA::culina_matrix<double> weight_matrix(cuLiNA::DIAGONAL);
    
    std::stringstream jacobian_ss_file_name;
    jacobian_ss_file_name << "/home/spades/kinetic_ws/src/cgmapping/datasets/matrices_to_test/jacobian_" <<  n << "x" << m << ".matrix";
    std::string jacobian_file_name(jacobian_ss_file_name.str());
    
    jacobian._setRows(n);
    jacobian._setColumns(m);
    cuLiNA::culina_load_matrix_file<double>(jacobian, jacobian_file_name);
    
    if(culiopD_1.workspace->_getNumber_of_elements() < jacobian._getNumber_of_elements()){
        
        std::cout << "test" << std::endl;
        culiopD_1.workspace->_setRows(jacobian._getNumber_of_elements());
        culiopD_1.workspace->_setColumns(1);
        culiopD_1.workspace->_allocateMatrixDataMemory();
        
    }
    
    //jacobian._printMatrix();
    
    weight_matrix._setRows(n);
    weight_matrix._setColumns(n);
    weight_matrix._allocateMatrixDataMemory();
    
    for(int i; i<weight_matrix._getRows(); i++){
        
        weight_matrix(i) = 1;
        
    }
    
    //weight_matrix._printMatrix();
    
    std::stringstream data_ss_file_name;
    data_ss_file_name << "/home/spades/kinetic_ws/src/cgmapping/datasets/matrices_to_test/data_" <<  n << "x" << 1 << ".matrix";
    std::string data_file_name(data_ss_file_name.str());
    
    data._setRows(n);
    data._setColumns(1);
    cuLiNA::culina_load_matrix_file<double>(data, data_file_name);
    
    //std::cout << std::endl << std::endl << std::endl << "data" << std::endl<< std::endl;
    //data._printMatrix();
    
    std::stringstream delta_ss_file_name;
    delta_ss_file_name << "/home/spades/kinetic_ws/src/cgmapping/datasets/matrices_to_test/delta_csi_" <<  m << "x" << 1 << ".matrix";
    std::string delta_file_name(delta_ss_file_name.str());
    
    delta._setRows(m);
    delta._setColumns(1);
    cuLiNA::culina_load_matrix_file<double>(delta, delta_file_name);
    
    //std::cout << std::endl << std::endl << std::endl << "delta" << std::endl<< std::endl;
    //delta._printMatrix();
    
    
    cudaStream_t strm1, strm2, strm3;

    culiopD_1.strm = &strm1;
    culiopD_2.strm = &strm2;
    culiopD_3.strm = &strm3;

    cudaStreamCreateWithFlags(culiopD_1.strm, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(culiopD_2.strm, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(culiopD_3.strm, cudaStreamNonBlocking);

    
    cuLiNA::cuLiNA_error_t stat;
    cudaDeviceSynchronize();
    
    cgmapping::Timer timer;
    stat = cuLiNA::culina_Dsolve_gradient_descent_first_order(&jacobian, &delta, &data, &weight_matrix, culiopD_1, culiopD_2, culiopD_3);
    cudaDeviceSynchronize();
    
    double time = timer.elapsed();
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);

    

    std::cout << "Time solver = " << time*1000000 << std::endl;
    
    //culiopD_2.workspace->_printMatrix();
    //delta._printMatrix();
    
    int count = 0;
    
    for(int i=0; i < m; i++){

        double var_calc = std::trunc(1000000*(*culiopD_2.workspace)(i,0))/1000000;
        double var_rec = std::trunc(1000000*delta(i,0))/1000000;

        
        
        if(var_calc != var_rec){

//            std::cout << "row " << i << std::endl;
//            std::cout << "(*culiopD_2.workspace)(i,0) = " << (*culiopD_2.workspace)(i,0) << std::endl;
//            std::cout << "delta(i,0) = " << delta(i,0) << std::endl;

            count++;
            
        }

        
        
    }
    
    std::cout << "count = " << count << std::endl;
    
    cuLiNA::culina_matrix3d skew_matrix;
    cuLiNA::culina_vector3d skew_vector;
    cuLiNA::culina_vector4d diag_cpy_vector;
    cuLiNA::culiopD_t culiopD_skew;
    
    skew_vector(0) = 1;
    skew_vector(1) = 2;
    skew_vector(2) = 3;
    
    culiopD_skew.strm = streams;
    
    stat = cuLiNA::culina_Dskew_matrix3x3_operator(&skew_vector, &skew_matrix, culiopD_skew);
    
    cuLiNA::cuLiNACheckErrors(stat, __FILE__, __FUNCTION__, __LINE__);
    
    std::cout << "Skew symetric matrix" << std::endl;
    skew_matrix._printMatrix();
    
    cuLiNA::culina_matrix<double, 4, 4> receiver_matrix;
    
    culiopD_skew.op_m1 = CUBLAS_OP_T;
    
    stat = cuLiNA::culina_Dblock_assignment_operation(&skew_vector, &receiver_matrix, 0,0, 1,0, 3,1, culiopD_skew);
    cuLiNA::cuLiNACheckErrors(stat, __FILE__,__FUNCTION__, __LINE__);
    culiopD_skew.op_m1 = CUBLAS_OP_N;
    stat = cuLiNA::culina_Dblock_assignment_operation(&skew_vector, &receiver_matrix, 0,0, 1,0, 3,1, culiopD_skew);
    cuLiNA::cuLiNACheckErrors(stat, __FILE__,__FUNCTION__, __LINE__);
    culiopD_skew.op_m1 = CUBLAS_OP_T;
    stat = cuLiNA::culina_Dblock_assignment_operation(&skew_vector, &receiver_matrix, 0,0, 2,1, 3,1, culiopD_skew);
    cuLiNA::cuLiNACheckErrors(stat, __FILE__,__FUNCTION__, __LINE__);
    
    std::cout << "Receiver Matrix" << std::endl;
    receiver_matrix._printMatrix();
    
    double trace = 0;
    
    culiopD_skew.op_m1 = CUBLAS_OP_N;
    //stat = cuLiNA::culina_Ddiagonal_to_vector_operation(&receiver_matrix, &diag_cpy_vector, culiopD_skew);
    stat = cuLiNA::culina_Dtrace_operation(&receiver_matrix, &diag_cpy_vector, trace, culiopD_skew);
    cuLiNA::cuLiNACheckErrors(stat, __FILE__,__FUNCTION__, __LINE__);
    
    cudaDeviceSynchronize();
    
    std::cout << "Diag vector" << std::endl;
    diag_cpy_vector._printMatrix();
    
    std::cout <<  "Trace result = " << trace << std::endl;
    
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaStreamDestroy(streams[2]);
    
    return 0;
    
}