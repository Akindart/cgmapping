//
// Created by spades on 12/04/18.
//

#ifndef CGMAPPING_ROS_VISUAL_ODOMETER_H
#define CGMAPPING_ROS_VISUAL_ODOMETER_H

#include <string>
#include <iostream>
#include <sstream>
#include <mutex>

#include <ros/ros.h>
#include <ros/time.h>

#include <tf2/transform_datatypes.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/message_filter.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>


#include <message_filters/subscriber.h>
#include <geometry_msgs/TransformStamped.h>

#include <sensor_msgs/PointCloud2.h>


using namespace std;

class ros_point_cloud_publisher{
 
    //world_base_frame_name_ will be referred simply as wbf in other variables
    string world_base_frame_name_ = string("world");
    
    string caemra_base_frame = string("base_link");
    geometry_msgs::TransformStamped cur_camera_pose_wrt_wbf;
    
    //initial pose of camera_base_frame wrt world_base_frame_name_;
    geometry_msgs::TransformStamped init_pose;
    
    //first point cloud stuff
    sensor_msgs::PointCloud2 first_pc_;
    string first_pc_pose_frame_wrt_wbf_ = string("first_pc_frame");
    string pc_topic_name_;
    geometry_msgs::TransformStamped first_pc_pose_;
    bool first_point_cloud_received_ = false;
    mutex first_pc_mutex_;
    uint first_pc_seq_tf_ = 0;
    uint first_pc_seq_ = 0;
    
    tf2_ros::Buffer tf2_buffer_;
    tf2_ros::TransformListener tf2_listener_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> point_cloud_sub_;
    tf2_ros::MessageFilter<sensor_msgs::PointCloud2> tf2_filter_;
    
    ros::NodeHandle private_nh_ = ros::NodeHandle("~");
    
    //ros::Subscriber point_cloud_sub_;
    
    void prepare_first_pc_to_publish();
    
 public:
    
    ros_point_cloud_publisher(string &pc_topic_name);
    
    void _initialPoint_cloud_callback(const sensor_msgs::PointCloud2ConstPtr& msg);
    
    void _initPoint_cloud_publisher(ros::Publisher &pub);
    
    void _broadcastCurrent_camera_pose(tf2_ros::TransformBroadcaster &tf_br);
    
    void _broadcastFirst_pc_pose_wrt_wbf(tf2_ros::StaticTransformBroadcaster &tf_static_br);
    
    void _visualOdometry_publisher(ros::Publisher &pub);

    
    
};

#endif //CGMAPPING_ROS_VISUAL_ODOMETER_H
