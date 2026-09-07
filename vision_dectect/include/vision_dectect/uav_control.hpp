#pragma once

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <stdint.h>
#include <iostream>

class UavControl
{
public:
	UavControl(ros::NodeHandle& nh, ros::NodeHandle& nh_private);

	void arm();
	void disarm();

private:
	ros::NodeHandle nh_;
	ros::NodeHandle nh_private_;

	ros::Publisher  pos_pub_;
	ros::ServiceClient arming_client_;
	ros::ServiceClient set_mode_client_;
	ros::Timer timer_;

	std::string self_ns_;
	uint64_t offboard_setpoint_counter_;

	void timer_callback(const ros::TimerEvent&);
	void publish_position_setpoint();
};
