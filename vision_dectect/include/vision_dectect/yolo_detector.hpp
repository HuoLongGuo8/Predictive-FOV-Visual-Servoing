#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct Detection {
    cv::Rect box;
    float conf;
    int classId;
};

struct DetectorConfig {
    float confThreshold = 0.4f;
    float iouThreshold = 0.45f;
    std::string modelPath;
    int inputWidth = 640;
    int inputHeight = 640;

    // COCO 类别过滤。为空表示不过滤。
    // 0=person, 2=car, 4=airplane
    std::vector<int> allowedClassIds = {0, 2, 4};

    // 当前 yolov5n.onnx 运行时表现为 FP16 输入，部分模型元数据不可靠，因此允许强制指定。
    bool forceFp16Input = true;
    bool forceFp16Output = true;
};

class YoloDetector {
public:
    explicit YoloDetector(const DetectorConfig& config);
    ~YoloDetector() = default;

    YoloDetector(const YoloDetector&) = delete;
    YoloDetector& operator=(const YoloDetector&) = delete;

    std::vector<Detection> detect(const cv::Mat& img);
    static void draw(cv::Mat& img, const std::vector<Detection>& objects);

private:
    cv::Mat letterbox(const cv::Mat& img, float& ratio, int& pad_x, int& pad_y);
    void nms(std::vector<Detection>& dets, float nms_thresh);
    static float iou(const Detection& a, const Detection& b);
    bool isClassAllowed(int class_id) const;
    static std::string cocoClassName(int class_id);

    DetectorConfig config_;
    Ort::Env env_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::string input_name_;
    std::string output_name_;

    bool input_is_fp16_ = false;
    bool output_is_fp16_ = false;
};
