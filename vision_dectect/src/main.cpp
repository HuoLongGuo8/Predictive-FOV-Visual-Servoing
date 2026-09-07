//
// Created by verse on 24-10-9.
//
#include <ros/ros.h>
#include "uav_topic_subscrib.hpp"

int main(int argc, char* argv[])
{
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
    ros::init(argc, argv, "uav_vision_dectect");

    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    //图像检测节点
    UavTopicSubscrib node(nh, nh_private);

    ros::spin();
    return 0;
}
