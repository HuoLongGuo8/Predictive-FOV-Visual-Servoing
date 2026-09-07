#include "uav_control.hpp"

UavControl::UavControl(ros::NodeHandle& nh, ros::NodeHandle& nh_private)
    : nh_(nh), nh_private_(nh_private)
{
    nh_private_.param<std::string>("self_ns", self_ns_, "/uav0/mavros");

    pos_pub_        = nh_.advertise<geometry_msgs::PoseStamped>(
                          self_ns_ + "/setpoint_position/local", 10);
    arming_client_  = nh_.serviceClient<mavros_msgs::CommandBool>(
                          self_ns_ + "/cmd/arming");
    set_mode_client_= nh_.serviceClient<mavros_msgs::SetMode>(
                          self_ns_ + "/set_mode");

    offboard_setpoint_counter_ = 0;

    timer_ = nh_.createTimer(ros::Duration(0.1),
        &UavControl::timer_callback, this);
}

void UavControl::timer_callback(const ros::TimerEvent&)
{
    publish_position_setpoint();

    if (offboard_setpoint_counter_ == 10) {
        mavros_msgs::SetMode mode_cmd;
        mode_cmd.request.custom_mode = "OFFBOARD";
        set_mode_client_.call(mode_cmd);
        arm();
    }
    if (offboard_setpoint_counter_ < 11) {
        offboard_setpoint_counter_++;
    }
}

/**
 * @brief Send a command to Arm the vehicle
 */
void UavControl::arm()
{
    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = true;
    if (arming_client_.call(arm_cmd) && arm_cmd.response.success) {
        ROS_INFO("Arm command sent");
    }
}

/**
 * @brief Send a command to Disarm the vehicle
 */
void UavControl::disarm()
{
    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = false;
    if (arming_client_.call(arm_cmd) && arm_cmd.response.success) {
        ROS_INFO("Disarm command sent");
    }
}

/**
 * @brief Publish a position setpoint (NED → ENU converted)
 *        Hover at x=5, y=5, z=5m altitude
 */
void UavControl::publish_position_setpoint()
{
    geometry_msgs::PoseStamped msg;
    msg.header.stamp    = ros::Time::now();
    msg.header.frame_id = "map";
    msg.pose.position.x = 5.0;   // East  (NED y=5)
    msg.pose.position.y = 5.0;   // North (NED x=5)
    msg.pose.position.z = 5.0;   // Up    (NED z=-5)
    msg.pose.orientation.w = 1.0;
    pos_pub_.publish(msg);
}


