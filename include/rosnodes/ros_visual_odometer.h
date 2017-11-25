//
// Created by spades on 24/04/18.
//

#ifndef CGMAPPING_ROS_VISUAL_ODOMETER_H
#define CGMAPPING_ROS_VISUAL_ODOMETER_H

#include <ros/ros.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Odometry.h>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/transform_datatypes.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Dense>

#include <cgmapping/visual_odometer.h>
#include <cgmapping/rgb_d_camera_model.h>

#include <cuLiNA/culina_definition.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> ApproxTimeSyncPolicy;

/***
 * @brief This object assumes depth image is already aligned with the rgb image
 * therefore we track the optical frame of the rgb camera.
 *
 *
 *
 */
 class ros_visual_odometer {
    
    ros::NodeHandle nh_ = ros::NodeHandle("~");
    
    cgmapping::cuda::visual_odometer<double> *visual_odometer_ = nullptr;
    cgmapping::rgb_d_camera_model camera_model_;
    
    std::string world_frame_name_;
    std::string base_link_frame_name_;
    std::string tracking_frame_name_;
    
    std::string rgb_image_topic_name_;
    std::string depth_image_topic_name_;
    std::string camera_info_topic_name_;
    
    double k_acceptance_error_convergence_delta_;
    double k_degrees_of_freedom_for_t_student_;
    double k_acceptance_std_dev_t_student_;
    int k_max_iterations_;
    
//     this is for when the info in depth image is somehow escalated
    int scale_for_depth_;
    
    Eigen::Matrix4d tracking_frame_wrt_world_frame_;
    Eigen::Matrix4d tracking_frame_wrt_base_link_;
    Eigen::Matrix4d base_link_frame_wrt_tracking_frame_;
    Eigen::Matrix4d base_link_frame_wrt_world_frame_;
    
    Eigen::Matrix4d delta_transformation_wrt_tracking_frame_;
    Eigen::Matrix4d delta_transformation_wrt_base_link_frame_;
    
    Eigen::Vector3d general_linear_vel_;
    Eigen::Vector3d general_angular_vel_;
    
    Eigen::Matrix<double, 6, 6> cur_tracking_frame_estimated_covariance_;
    Eigen::Matrix<double, 6, 6> cur_base_link_frame_estimated_covariance_;
    Eigen::Matrix<double, 6, 6> cur_delta_wrt_base_link_covariance_;
    Eigen::Matrix<double, 6, 6> cur_delta_wrt_tracking_frame_covariance_;
    
    Eigen::Matrix<double, 6, 6> general_adjoint_matrix_;
    
    cuLiNA::culina_matrix<double,6,6> d_cur_tracking_frame_estimated_covariance_;
    cuLiNA::culina_matrix<double,6,6> d_cur_base_link_frame_estimated_covariance_;
    
    cuLiNA::culina_matrix4d d_generical_homogenic_transformation_;
    cuLiNA::culina_matrix3d d_auxiliar_matrix1_;
    cuLiNA::culina_matrix3d d_auxiliar_matrix2_;
    cuLiNA::culina_vector3d d_general_linear_vel_;
    cuLiNA::culina_vector3d d_general_angular_vel_;
    
    bool first_run = false;
    
    int min_img_size;
    int max_img_size;
    
    unsigned int transform_seq = 0;
    unsigned int odometry_seq = 0;
    
    cudaStream_t raw_strm_;
    cv::cuda::Stream strm_;
    
    geometry_msgs::TransformStamped base_link_wrt_world_frame_stamped_transformed_;
    geometry_msgs::TransformStamped tracking_frame_wrt_world_frame_stamped_transformed_;
    
    nav_msgs::Odometry camera_odometry;
    
    tf2_ros::Buffer tf2_buffer_;
    tf2_ros::TransformListener tf2_listener_;
    tf2_ros::TransformBroadcaster tf2_broadcaster_;
    
    message_filters::Subscriber<sensor_msgs::Image> rgb_image_sub_;
    message_filters::Subscriber<sensor_msgs::Image> depth_image_sub_;
    message_filters::Subscriber<sensor_msgs::CameraInfo> camera_info_sub_;
    message_filters::Synchronizer<ApproxTimeSyncPolicy> rgbd_sync_;
    //message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> rgbd_sync_;
    
    ros::Publisher odom_pub;
    
    void broadcast_base_link_wrt_world_transformation();
    
    void correct_frame_name(std::string &frame_name);
    
    public:
    
     ros_visual_odometer();
     
     void _estimateCamera_motion(const sensor_msgs::ImageConstPtr& rgb_image,
                                 const sensor_msgs::ImageConstPtr& depth_image,
                                 const sensor_msgs::CameraInfoConstPtr& camera_info);
    
     void _publishOdometry_msg(const std::string &frame_name,
                               const std::string &child_fame_name,
                               const Matrix4d &child_frame_wrt_frame_transform,
                               const Vector3d &twist_linear_vel,
                               const Vector3d &twist_angular_vel,
                               const ros::Time message_stamp,
                               const Matrix<double, 6, 6> &twist_vel_cov_matrix,
                               const Matrix<double, 6, 6> &odom_cov_matrix);
     
     void _broadcast_tracking_frame_wrt_world_frame_transformation(const std::string &frame_name,
                                                                   const std::string &child_fame_name,
                                                                   const Matrix4d &child_frame_wrt_frame_transform,
                                                                   const ros::Time message_stamp);
     
     static void convertTransform2Eigen4d_homogenic_matrix(geometry_msgs::Transform &geo_msg, Eigen::Matrix4d &homogenic_matrix);
     
     static void convertEigen4d_homogenic_matrix2Transform(const Matrix4d &homogenic_matrix,
                                                           geometry_msgs::Transform &geo_msg);
     
     ~ros_visual_odometer();
    
     
 };

#endif //CGMAPPING_ROS_VISUAL_ODOMETER_H
