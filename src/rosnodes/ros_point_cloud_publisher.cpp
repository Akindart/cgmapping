//
// Created by spades on 13/04/18.
//

#include <rosnodes/ros_point_cloud_publisher.h>

ros_point_cloud_publisher::ros_point_cloud_publisher(string &pc_topic_name) : tf2_listener_(tf2_buffer_),
                                                                              pc_topic_name_(pc_topic_name),
                                                                              tf2_filter_(point_cloud_sub_, tf2_buffer_, this->world_base_frame_name_, 10, 0) {
    
    point_cloud_sub_.subscribe(this->private_nh_, this->pc_topic_name_, 10);
    tf2_filter_.registerCallback(boost::bind(&ros_point_cloud_publisher::_initialPoint_cloud_callback, this, _1));
    
    
    
}

void ros_point_cloud_publisher::_initialPoint_cloud_callback(const sensor_msgs::PointCloud2ConstPtr &msg) {
    
    if(!this->first_point_cloud_received_) {
        
        this->first_pc_ = *msg;
        
        if(this->first_pc_.header.frame_id.c_str()[0] == '/'){
            
            auto str_size = this->first_pc_.header.frame_id.size();
            
            this->first_pc_.header.frame_id = this->first_pc_.header.frame_id.substr(1, str_size-1);
            
        }
        
        this->tf2_buffer_.transform(this->first_pc_, this->first_pc_, this->world_base_frame_name_);
        
        this->first_point_cloud_received_ = true;
        
    }
    
}

void ros_point_cloud_publisher::_initPoint_cloud_publisher(ros::Publisher &pub) {
    
    if(this->first_point_cloud_received_) {
        
        this->prepare_first_pc_to_publish();
        
        return pub.publish(this->first_pc_);
        
    }
    
}

void ros_point_cloud_publisher::_broadcastCurrent_camera_pose(tf2_ros::TransformBroadcaster &tf_br) {
    
    if(this->first_point_cloud_received_) {
        
        this->first_pc_pose_.header.stamp = ros::Time::now();
        
        this->first_pc_pose_.header.seq = first_pc_seq_tf_++;
        
        tf_br.sendTransform(this->first_pc_pose_);
    }
    
}
void ros_point_cloud_publisher::_broadcastFirst_pc_pose_wrt_wbf(tf2_ros::StaticTransformBroadcaster &tf_static_br) {
    
    if(this->first_point_cloud_received_) {
        
        this->first_pc_pose_.header.stamp = ros::Time::now();
        
        this->first_pc_pose_.header.seq = first_pc_seq_tf_++;
        
        tf_static_br.sendTransform(this->first_pc_pose_);
    }
    
}

void ros_point_cloud_publisher::prepare_first_pc_to_publish() {
    
    this->first_pc_.header.stamp = ros::Time::now();
    
    this->first_pc_.header.seq = first_pc_seq_++;
    
}