//
// Created by spades on 24/04/18.
//

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>

#include "rosnodes/ros_visual_odometer.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>

ros_visual_odometer::ros_visual_odometer() : tf2_listener_(tf2_buffer_),
                                             rgbd_sync_(ApproxTimeSyncPolicy(3), rgb_image_sub_, depth_image_sub_, camera_info_sub_) {
    
    if(!this->nh_.getParam("world_frame_name", this->world_frame_name_))
        this->world_frame_name_  = std::string("world");
    else correct_frame_name(this->world_frame_name_);
    
    if(!this->nh_.getParam("base_link_frame_name", this->base_link_frame_name_))
        this->base_link_frame_name_  = std::string("kinect");
    else correct_frame_name(this->base_link_frame_name_);
    
    if(!this->nh_.getParam("depth_image_topic_name", this->depth_image_topic_name_))
        this->depth_image_topic_name_  = std::string("/camera/depth/image");
    
    if(!this->nh_.getParam("rgb_image_topic_name", this->rgb_image_topic_name_))
        this->rgb_image_topic_name_  = std::string("/camera/rgb/image_color");
    
    if(!this->nh_.getParam("camera_info_topic_name", this->camera_info_topic_name_))
        this->camera_info_topic_name_  = std::string("/camera/rgb/camera_info");
    
    if(!this->nh_.getParam("param_accptance_optimization_error_convergence", this->k_acceptance_error_convergence_delta_))
        this->k_acceptance_error_convergence_delta_  = 0.00001;
    
    if(!this->nh_.getParam("param_accptance_std_dev_convergence", this->k_acceptance_std_dev_t_student_))
        this->k_acceptance_std_dev_t_student_  = 0.00001;
    
    if(!this->nh_.getParam("param_degrees_of_freedom_t_student", this->k_degrees_of_freedom_for_t_student_))
        this->k_degrees_of_freedom_for_t_student_  = 5;
    
    if(!this->nh_.getParam("param_depth_scale_parameter", this->scale_for_depth_))
        this->scale_for_depth_ = 1;
    
    if(!this->nh_.getParam("param_max_iterations", this->k_max_iterations_))
        this->k_max_iterations_ = 1;
    
    if(!this->nh_.getParam("param_min_img_size", this->min_img_size))
        this->min_img_size = 0;
    
    if(!this->nh_.getParam("param_max_img_size", this->max_img_size))
        this->max_img_size = 4;
    
    if(this->min_img_size > this->max_img_size) {
    
        this->min_img_size = 0;
        this->max_img_size = 4;
    
    }
    
    if(this->min_img_size < 0 || this->min_img_size > 4) this->min_img_size = 0;
    if(this->max_img_size < 0 || this->max_img_size > 4) this->max_img_size = 4;
    
    rgb_image_sub_.subscribe(this->nh_, this->rgb_image_topic_name_, 1);
    depth_image_sub_.subscribe(this->nh_, this->depth_image_topic_name_, 1);
    camera_info_sub_.subscribe(this->nh_, this->camera_info_topic_name_, 1);
    
    rgbd_sync_.registerCallback(boost::bind(&ros_visual_odometer::_estimateCamera_motion, this, _1, _2, _3));
    
    this->odom_pub = nh_.advertise<nav_msgs::Odometry>("ros_visual_odometry", 10);
    
    this->d_generical_homogenic_transformation_._allocateMatrixDataMemory();
    this->d_general_linear_vel_._allocateMatrixDataMemory();
    this->d_general_angular_vel_._allocateMatrixDataMemory();
    this->d_auxiliar_matrix1_._allocateMatrixDataMemory();
    this->d_auxiliar_matrix2_._allocateMatrixDataMemory();
    
    cudaStreamCreateWithFlags(&this->raw_strm_, cudaStreamNonBlocking);
    this->strm_ = cv::cuda::StreamAccessor::wrapStream(this->raw_strm_);
    
    this->cur_base_link_frame_estimated_covariance_.setZero();
    this->cur_base_link_frame_estimated_covariance_.diagonal().setConstant(0.00000001);
    
    this->cur_tracking_frame_estimated_covariance_.setZero();
    this->cur_tracking_frame_estimated_covariance_.diagonal().setConstant(0.00000001);
    
    this->cur_delta_wrt_tracking_frame_covariance_.setZero();
    this->cur_delta_wrt_tracking_frame_covariance_.diagonal().setConstant(0.00000001);
    
    this->delta_transformation_wrt_tracking_frame_.setIdentity();
    
    std::cout << "teeheee" << std::endl;
    std::cout << "scale_depth = " << this->scale_for_depth_ << std::endl;
    
    this->first_run = true;
    
}



void ros_visual_odometer::_estimateCamera_motion(const sensor_msgs::ImageConstPtr &rgb_image,
                                                 const sensor_msgs::ImageConstPtr &depth_image,
                                                 const sensor_msgs::CameraInfoConstPtr &camera_info) {
    
    cv::Mat cv_rgb_image, cv_depth_image;
    
//    ROS_INFO("rgb_image_stamp: %d.%d\n", rgb_image->header.stamp.sec,rgb_image->header.stamp.nsec);
//    ROS_INFO("depth_image_stamp: %d.%d\n", depth_image->header.stamp.sec,depth_image->header.stamp.nsec);
//    ROS_INFO("camera_info_stamp: %d.%d\n\n", camera_info->header.stamp.sec,camera_info->header.stamp.nsec);
    
    try {
    
        cv_rgb_image = cv_bridge::toCvShare(rgb_image, rgb_image->encoding)->image;
        cv_depth_image = cv_bridge::toCvShare(depth_image, depth_image->encoding)->image;
        
    }catch(cv_bridge::Exception& e) {
    
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    
    }
    
    cv_rgb_image.convertTo(cv_rgb_image, CV_8UC1);
    
    
    
    if(first_run){
        
        this->tracking_frame_name_ = rgb_image->header.frame_id;
    
        this->correct_frame_name(this->tracking_frame_name_);
        
        geometry_msgs::TransformStamped base_link_wrt_image_frame, tracking_frame_wrt_world_frame;

        try {
            tracking_frame_wrt_world_frame = this->tf2_buffer_.lookupTransform(this->world_frame_name_, this->tracking_frame_name_, ros::Time(0));
        }catch (tf2::TransformException &ex) {
            ROS_WARN("Could NOT transform %s to %s: %s",this->tracking_frame_name_.c_str(), this->world_frame_name_.c_str(), ex.what());
            return;
        }

        ros_visual_odometer::convertTransform2Eigen4d_homogenic_matrix(tracking_frame_wrt_world_frame.transform, this->tracking_frame_wrt_world_frame_);
        
        this->camera_model_ = cgmapping::rgb_d_camera_model(camera_info->K[0],
                                                            camera_info->K[4],
                                                            camera_info->K[2],
                                                            camera_info->K[5],
                                                            (uint)this->scale_for_depth_);
        
        this->visual_odometer_ = new cgmapping::cuda::visual_odometer<double>(cv_rgb_image,
                                                                              cv_depth_image,
                                                                              this->camera_model_,
                                                                              this->k_acceptance_error_convergence_delta_,
                                                                              this->k_max_iterations_,
                                                                              this->k_degrees_of_freedom_for_t_student_,
                                                                              this->k_acceptance_std_dev_t_student_,
                                                                              tracking_frame_wrt_world_frame_.data());

        first_run = false;
        
    }
    else{
   
        this->visual_odometer_->_estimateCamera_motion(cv_rgb_image,
                                                       cv_depth_image,
                                                       static_cast<cgmapping::image_size_t>(this->min_img_size),
                                                       static_cast<cgmapping::image_size_t>(this->max_img_size));
        
        std::cout << "delta calculated" <<std::endl;
    
        this->visual_odometer_->_getCur_estimated_pose()._downloadData(this->tracking_frame_wrt_world_frame_.data(),
                                                                  16,
                                                                  &(this->raw_strm_));
        this->visual_odometer_->_getCur_delta_transform()._downloadData(this->delta_transformation_wrt_tracking_frame_.data(),
                                                                        16,
                                                                        &(this->raw_strm_));
        this->visual_odometer_->_getLower_bound_delta_twist_cov_matrix()._downloadData(this->cur_delta_wrt_tracking_frame_covariance_.data(),
                                                                                       36,
                                                                                       &(this->raw_strm_));
        this->visual_odometer_->_getDelta_linear_vel()._downloadData(this->general_linear_vel_.data(),
                                                                     3,
                                                                     &(this->raw_strm_));
        this->visual_odometer_->_getDelta_angular_vel()._downloadData(this->general_angular_vel_.data(),
                                                                     3,
                                                                     &(this->raw_strm_));
        
        
        strm_.waitForCompletion();
        
        //std::cout << this->tracking_frame_wrt_world_frame_ << std::endl;
        
        
    }
    
    
    auto tracking_frame_name_expanded = this->tracking_frame_name_;
    
    tracking_frame_name_expanded.append("_estimated");
    
    this->_broadcast_tracking_frame_wrt_world_frame_transformation(this->world_frame_name_,
                                                                   tracking_frame_name_expanded,
                                                                   this->tracking_frame_wrt_world_frame_,
                                                                   rgb_image->header.stamp);
    
//    this->_publishOdometry_msg(this->world_frame_name_,
//                               this->tracking_frame_name_,
//                               this->tracking_frame_wrt_world_frame_,
//                               this->general_linear_vel_,
//                               this->general_angular_vel_,
//                               rgb_image->header.stamp,
//                               cur_delta_wrt_tracking_frame_covariance_,
//                               cur_delta_wrt_tracking_frame_covariance_);
    
//    ros_visual_odometer::convertEigen4d_homogenic_matrix2Transform(this->base_link_frame_wrt_world_frame_,
//                                                                   this->base_link_wrt_world_frame_stamped_transformed_.transform);
  
//    this->broadcast_base_link_wrt_world_transformation();
    
}

void ros_visual_odometer::convertTransform2Eigen4d_homogenic_matrix(geometry_msgs::Transform &geo_msg,
                                                                    Eigen::Matrix4d &homogenic_matrix) {
    
    tf2::Matrix3x3 tmp_matrix(tf2::Quaternion(geo_msg.rotation.x,
                                              geo_msg.rotation.y,
                                              geo_msg.rotation.z,
                                              geo_msg.rotation.w));
    
    homogenic_matrix <<
        tmp_matrix.getRow(0).getX(), tmp_matrix.getRow(0).getY(), tmp_matrix.getRow(0).getZ(), geo_msg.translation.x,
        tmp_matrix.getRow(1).getX(), tmp_matrix.getRow(1).getY(), tmp_matrix.getRow(1).getZ(), geo_msg.translation.y,
        tmp_matrix.getRow(2).getX(), tmp_matrix.getRow(2).getY(), tmp_matrix.getRow(2).getZ(), geo_msg.translation.z,
        0,                           0,                           0,                           1;
    
}


void ros_visual_odometer::convertEigen4d_homogenic_matrix2Transform(const Matrix4d &homogenic_matrix,
                                                                    geometry_msgs::Transform &geo_msg) {
    
    geo_msg.translation.x = homogenic_matrix(0,3);
    geo_msg.translation.y = homogenic_matrix(1,3);
    geo_msg.translation.z = homogenic_matrix(2,3);
    
    tf2::Matrix3x3 tmp_matrix;
    
    tmp_matrix.setValue(homogenic_matrix(0,0), homogenic_matrix(0,1), homogenic_matrix(0,2),
                        homogenic_matrix(1,0), homogenic_matrix(1,1), homogenic_matrix(1,2),
                        homogenic_matrix(2,0), homogenic_matrix(2,1), homogenic_matrix(2,2));
    
    tf2::Quaternion tmp_quaternion;
    
    tmp_matrix.getRotation(tmp_quaternion);
    
    geo_msg.rotation.x = tmp_quaternion.getX();
    geo_msg.rotation.y = tmp_quaternion.getY();
    geo_msg.rotation.z = tmp_quaternion.getZ();
    geo_msg.rotation.w = tmp_quaternion.getW();
    
}

void ros_visual_odometer::correct_frame_name(std::string &frame_name) {
    
    if(frame_name.c_str()[0] == '/'){
        
        auto str_size = frame_name.size();
        frame_name = frame_name.substr(1, str_size-1);
        
    }
    
}

void ros_visual_odometer::broadcast_base_link_wrt_world_transformation() {
    
    this->base_link_wrt_world_frame_stamped_transformed_.header.seq = this->transform_seq++;
    
    this->base_link_wrt_world_frame_stamped_transformed_.header.stamp = ros::Time::now();
    
    this->tf2_broadcaster_.sendTransform(this->base_link_wrt_world_frame_stamped_transformed_);

}


void ros_visual_odometer::_publishOdometry_msg(const std::string &frame_name,
                                               const std::string &child_fame_name,
                                               const Matrix4d &child_frame_wrt_frame_transform,
                                               const Vector3d &twist_linear_vel,
                                               const Vector3d &twist_angular_vel,
                                               const ros::Time message_stamp,
                                               const Matrix<double, 6, 6> &twist_vel_cov_matrix,
                                               const Matrix<double, 6, 6> &odom_cov_matrix) {

    this->camera_odometry.header.frame_id = frame_name;
    this->camera_odometry.child_frame_id = child_fame_name;
    this->camera_odometry.header.seq = odometry_seq++;
    this->camera_odometry.header.stamp = message_stamp;
    
    geometry_msgs::Transform child_frame_wrt_frame_ros_transform;
    
    ros_visual_odometer::convertEigen4d_homogenic_matrix2Transform(child_frame_wrt_frame_transform,
                                                                   child_frame_wrt_frame_ros_transform);
    
    this->camera_odometry.pose.pose.position.x = child_frame_wrt_frame_ros_transform.translation.x;
    this->camera_odometry.pose.pose.position.y = child_frame_wrt_frame_ros_transform.translation.y;
    this->camera_odometry.pose.pose.position.z = child_frame_wrt_frame_ros_transform.translation.z;
    
    this->camera_odometry.pose.pose.orientation.x = child_frame_wrt_frame_ros_transform.rotation.x;
    this->camera_odometry.pose.pose.orientation.y = child_frame_wrt_frame_ros_transform.rotation.y;
    this->camera_odometry.pose.pose.orientation.z = child_frame_wrt_frame_ros_transform.rotation.z;
    this->camera_odometry.pose.pose.orientation.w = child_frame_wrt_frame_ros_transform.rotation.w;
    
    this->camera_odometry.twist.twist.linear.x = twist_linear_vel(0);
    this->camera_odometry.twist.twist.linear.y = twist_linear_vel(1);
    this->camera_odometry.twist.twist.linear.z = twist_linear_vel(2);
    
    this->camera_odometry.twist.twist.angular.x = twist_angular_vel(0);
    this->camera_odometry.twist.twist.angular.y = twist_angular_vel(1);
    this->camera_odometry.twist.twist.angular.z = twist_angular_vel(2);
    
    typedef Matrix<double,6,6,RowMajor> RowMajMat6d;

    Eigen::Map<RowMajMat6d>(this->camera_odometry.twist.covariance.data(),
                            twist_vel_cov_matrix.rows(),
                            twist_vel_cov_matrix.cols()) = twist_vel_cov_matrix;
    
    Eigen::Map<RowMajMat6d>(this->camera_odometry.pose.covariance.data(),
                            odom_cov_matrix.rows(),
                            odom_cov_matrix.cols()) = odom_cov_matrix;
    
    this->odom_pub.publish(this->camera_odometry);
    
}


ros_visual_odometer::~ros_visual_odometer() {
    
    if(this->visual_odometer_ != nullptr)
        delete this->visual_odometer_;
    
}
void ros_visual_odometer::_broadcast_tracking_frame_wrt_world_frame_transformation(const std::string &frame_name,
                                                                                   const std::string &child_fame_name,
                                                                                   const Matrix4d &child_frame_wrt_frame_transform,
                                                                                   const ros::Time message_stamp) {
    
    ros_visual_odometer::convertEigen4d_homogenic_matrix2Transform(child_frame_wrt_frame_transform,
                                                                   this->tracking_frame_wrt_world_frame_stamped_transformed_.transform);
    
    this->tracking_frame_wrt_world_frame_stamped_transformed_.header.seq = this->transform_seq++;
    
    this->tracking_frame_wrt_world_frame_stamped_transformed_.header.frame_id = frame_name;
    
    this->tracking_frame_wrt_world_frame_stamped_transformed_.child_frame_id = child_fame_name;
    
    this->tracking_frame_wrt_world_frame_stamped_transformed_.header.stamp = message_stamp;
    
    this->tf2_broadcaster_.sendTransform(this->tracking_frame_wrt_world_frame_stamped_transformed_);
    
}
