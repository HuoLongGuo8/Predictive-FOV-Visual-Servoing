#pragma once

#include <opencv2/core/mat.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include "LightTrack.h"

#include <uav_common_msg/RectMsg.h>

#include "yolo_detector.hpp"

#include <thread>
#include <string>
#include <vector>


class UavTopicSubscrib
{

public:
    cv::Mat uav_camera_frame;
    cv::Rect uav_result_rect;
    UavTopicSubscrib(ros::NodeHandle& nh, ros::NodeHandle& nh_private);
    ~UavTopicSubscrib();


private:
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    ros::Subscriber uav_image_sub_;

    ros::Publisher uav_detect_result_publisher_;

    uav_common_msg::RectMsg pub_uav_result_rect;

    cv_bridge::CvImagePtr orig_cv_ptr;


    /************************LightTrack跟踪部分************************/
    cv::Rect trackWindow;
    cv::Mat init_window;

    LightTrack *siam_tracker;
    int light_track_flag = 0;

    /************************YOLO Detector 资源成员变量************************/
    std::string yolo_model_path_;
    std::string image_topic_;
    std::string detect_topic_;
    std::vector<int> allowed_class_ids_;
    double accept_score_threshold_ = 0.5;
    bool enable_tracking_ = true;
    bool force_fp16_input_ = true;
    bool force_fp16_output_ = true;

    void initTensorRT();
    std::vector<int> parseClassIds(const std::string& text) const;

    std::unique_ptr<YoloDetector> yolo_detector_;


    /**************************************************/
    std::thread uav_detect_result_thread_;

    void uav_detect_result_loop();
    void image_callback(const sensor_msgs::ImageConstPtr& msg);
    void cxy_wh_2_rect(const cv::Point& pos, const cv::Point2f& sz, cv::Rect &rect);
};



