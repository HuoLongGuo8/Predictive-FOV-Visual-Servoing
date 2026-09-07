#include "yolo_detector.hpp"

#include <iostream>
#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>

YoloDetector::YoloDetector(const DetectorConfig& config)
    : config_(config)
    , env_(ORT_LOGGING_LEVEL_WARNING, "YoloDetector")
    , session_(nullptr)
{
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        OrtCUDAProviderOptions cuda_opts;
        cuda_opts.device_id = 0;
        opts.AppendExecutionProvider_CUDA(cuda_opts);
        std::cout << "YoloDetector: CUDA provider enabled" << std::endl;
    } catch (const Ort::Exception&) {
        std::cerr << "YoloDetector: CUDA unavailable, using CPU" << std::endl;
    }

    session_ = Ort::Session(env_, config.modelPath.c_str(), opts);

    auto input_name = session_.GetInputNameAllocated(0, allocator_);
    input_name_ = input_name.get();

    auto output_name = session_.GetOutputNameAllocated(0, allocator_);
    output_name_ = output_name.get();

    try {
        auto input_type_info = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        auto output_type_info = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();

        const auto input_type = input_type_info.GetElementType();
        const auto output_type = output_type_info.GetElementType();

        input_is_fp16_ = (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
        output_is_fp16_ = (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    } catch (const std::exception& e) {
        std::cerr << "YoloDetector: failed to query tensor type, fallback to config. reason: "
                  << e.what() << std::endl;
    }

    // 强制参数优先，用于处理模型元数据与实际运行类型不一致的情况。
    if (config_.forceFp16Input) {
        input_is_fp16_ = true;
    }
    if (config_.forceFp16Output) {
        output_is_fp16_ = true;
    }

    std::cout << "ONNX model loaded: " << config.modelPath << std::endl;
    std::cout << "  Input:  " << input_name_ << " [1, 3, " << config.inputHeight << ", " << config.inputWidth << "]"
              << " feed_type=" << (input_is_fp16_ ? "float16" : "float32")
              << " force_fp16_input=" << (config_.forceFp16Input ? "true" : "false") << std::endl;
    std::cout << "  Output: " << output_name_
              << " read_type=" << (output_is_fp16_ ? "float16" : "float32")
              << " force_fp16_output=" << (config_.forceFp16Output ? "true" : "false") << std::endl;

    std::cout << "  Allowed COCO class IDs: ";
    if (config.allowedClassIds.empty()) {
        std::cout << "ALL";
    } else {
        for (size_t i = 0; i < config.allowedClassIds.size(); ++i) {
            int cid = config.allowedClassIds[i];
            std::cout << cid << "(" << cocoClassName(cid) << ")";
            if (i + 1 < config.allowedClassIds.size()) std::cout << ", ";
        }
    }
    std::cout << std::endl;
}

cv::Mat YoloDetector::letterbox(const cv::Mat& img, float& ratio, int& pad_x, int& pad_y) {
    float r_w = static_cast<float>(config_.inputWidth) / img.cols;
    float r_h = static_cast<float>(config_.inputHeight) / img.rows;
    ratio = std::min(r_w, r_h);

    int new_w = static_cast<int>(img.cols * ratio);
    int new_h = static_cast<int>(img.rows * ratio);
    pad_x = (config_.inputWidth - new_w) / 2;
    pad_y = (config_.inputHeight - new_h) / 2;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat out(config_.inputHeight, config_.inputWidth, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(pad_x, pad_y, new_w, new_h)));
    return out;
}

float YoloDetector::iou(const Detection& a, const Detection& b) {
    int x1 = std::max(a.box.x, b.box.x);
    int y1 = std::max(a.box.y, b.box.y);
    int x2 = std::min(a.box.x + a.box.width, b.box.x + b.box.width);
    int y2 = std::min(a.box.y + a.box.height, b.box.y + b.box.height);

    if (x1 >= x2 || y1 >= y2) return 0.0f;

    float inter_area = static_cast<float>((x2 - x1) * (y2 - y1));
    float union_area = static_cast<float>(a.box.area() + b.box.area()) - inter_area;
    if (union_area <= 1e-6f) return 0.0f;
    return inter_area / union_area;
}

bool YoloDetector::isClassAllowed(int class_id) const {
    if (config_.allowedClassIds.empty()) return true;
    return std::find(config_.allowedClassIds.begin(),
                     config_.allowedClassIds.end(),
                     class_id) != config_.allowedClassIds.end();
}

std::string YoloDetector::cocoClassName(int class_id) {
    switch (class_id) {
        case 0: return "person";
        case 2: return "car";
        case 4: return "airplane";
        case 5: return "bus";
        case 7: return "truck";
        default: return "cls_" + std::to_string(class_id);
    }
}

void YoloDetector::nms(std::vector<Detection>& dets, float nms_thresh) {
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
        return a.conf > b.conf;
    });

    std::vector<Detection> result;
    std::vector<bool> suppressed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!suppressed[j] && dets[i].classId == dets[j].classId &&
                iou(dets[i], dets[j]) > nms_thresh) {
                suppressed[j] = true;
            }
        }
    }
    dets = std::move(result);
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& img_raw) {
    if (img_raw.empty()) return {};

    float ratio;
    int pad_x, pad_y;
    cv::Mat input_img = letterbox(img_raw, ratio, pad_x, pad_y);

    cv::Mat rgb;
    cv::cvtColor(input_img, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);

    std::vector<float> input_data(3 * config_.inputHeight * config_.inputWidth);
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(input_data.data() + c * config_.inputHeight * config_.inputWidth,
                    channels[c].data,
                    config_.inputHeight * config_.inputWidth * sizeof(float));
    }

    std::array<int64_t, 4> input_shape = {1, 3, config_.inputHeight, config_.inputWidth};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};

    std::vector<Ort::Value> output_tensors;

    if (input_is_fp16_) {
        std::vector<Ort::Float16_t> input_data_fp16(input_data.size());
        for (size_t i = 0; i < input_data.size(); ++i) {
            input_data_fp16[i] = Ort::Float16_t(input_data[i]);
        }

        Ort::Value input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
            memory_info, input_data_fp16.data(), input_data_fp16.size(),
            input_shape.data(), input_shape.size());

        output_tensors = session_.Run(Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1, output_names, 1);
    } else {
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());

        output_tensors = session_.Run(Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1, output_names, 1);
    }

    auto& out_tensor = output_tensors[0];
    auto out_shape = out_tensor.GetTensorTypeAndShapeInfo().GetShape();

    if (out_shape.size() < 3) {
        std::cerr << "YOLO output shape is not [1, anchors, data_len]" << std::endl;
        return {};
    }

    int num_anchors = static_cast<int>(out_shape[1]);
    int data_len = static_cast<int>(out_shape[2]);

    std::vector<float> out_float(static_cast<size_t>(num_anchors) * static_cast<size_t>(data_len), 0.0f);

    if (output_is_fp16_) {
        const Ort::Float16_t* out_data = out_tensor.GetTensorData<Ort::Float16_t>();
        for (size_t i = 0; i < out_float.size(); ++i) {
            out_float[i] = static_cast<float>(out_data[i]);
        }
    } else {
        const float* out_data = out_tensor.GetTensorData<float>();
        std::memcpy(out_float.data(), out_data, out_float.size() * sizeof(float));
    }

    std::vector<Detection> detections;
    for (int i = 0; i < num_anchors; ++i) {
        const float* row = out_float.data() + static_cast<size_t>(i) * data_len;
        if (data_len < 5) continue;

        float obj_conf = row[4];
        if (obj_conf < config_.confThreshold) continue;

        float max_cls_score = 0.0f;
        int class_id = 0;
        if (data_len > 5) {
            for (int c = 5; c < data_len; ++c) {
                if (row[c] > max_cls_score) {
                    max_cls_score = row[c];
                    class_id = c - 5;
                }
            }
        } else {
            max_cls_score = 1.0f;
        }

        float score = obj_conf * max_cls_score;
        if (score < config_.confThreshold) continue;
        if (!isClassAllowed(class_id)) continue;

        float cx = (row[0] - pad_x) / ratio;
        float cy = (row[1] - pad_y) / ratio;
        float w = row[2] / ratio;
        float h = row[3] / ratio;

        cv::Rect box(static_cast<int>(cx - w / 2.0f),
                     static_cast<int>(cy - h / 2.0f),
                     static_cast<int>(w),
                     static_cast<int>(h));
        box = box & cv::Rect(0, 0, img_raw.cols, img_raw.rows);
        if (box.width <= 0 || box.height <= 0 || box.area() < 10) continue;

        detections.push_back({box, score, class_id});
    }

    nms(detections, config_.iouThreshold);
    return detections;
}

void YoloDetector::draw(cv::Mat& img, const std::vector<Detection>& objects) {
    for (const auto& obj : objects) {
        cv::rectangle(img, obj.box, cv::Scalar(255, 0, 0), 2);
        std::string label = cocoClassName(obj.classId) + cv::format(" %.2f", obj.conf);
        cv::putText(img, label, cv::Point(obj.box.x, std::max(20, obj.box.y - 10)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
    }
}
