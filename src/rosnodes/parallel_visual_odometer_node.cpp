//
// Created by spades on 12/04/18.
//


#include <string>

#include <ros/ros.h>

#include <rosnodes/ros_visual_odometer.h>
#include <rosnodes/ros_point_cloud_publisher.h>

int main(int argc, char **argv) {
    
    ros::init(argc, argv, "visual_odometer_node");
    
    ros::NodeHandle n;
    
    auto cublas_stat = cuBLAS_wrapper::cublas_wrapper::_start_cublas_handle_wrapper();
    cuBLAS_wrapper::cublas_wrapper::_cublasCheckErrors(cublas_stat, __FILE__, __FUNCTION__, __LINE__);
    
    std::string pc_frame("/camera/rgb/points");
    
//    ros_point_cloud_publisher visual_odometer(pc_frame);
    
//    ros_visual_odometer_();
    
    ros_visual_odometer ros_vo;
    
    ros::Rate rate(1);
    
    ros::Publisher pub;
    
//    pub = n.advertise<sensor_msgs::PointCloud2>("first_point_cloud", 10);
    
   while(ros::ok()){
//
////        visual_odometer._initPoint_cloud_publisher(pub);
//
        ros::spinOnce();
//        rate.sleep();
//
    }
    
    return 0;
    
    
}