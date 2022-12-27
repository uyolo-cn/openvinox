/**
 * @file Yolov5Detector.hpp
 * @author Kris Wang (jiujiangluck@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-12-24
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef YOLOV5DETECTOR_HPP
#define YOLOV5DETECTOR_HPP
#include <iostream>
#include <string>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

#define STATUS int

/**
 * 使用yolov5实现的目标分类器
 */

class Yolov5Detector {

public:
    typedef struct {
        float prob;
        std::string name;
        cv::Rect rect;
    } Object;

    Yolov5Detector();

    STATUS init(std::string engineName, double thresh, double nmsthresh);

    void unInit();

    STATUS processImage(const cv::Mat &image, std::vector<Object> &detectResults);
    
    void postprocess(const cv::Size& resizedImageShape, const cv::Size& originalImageShape, const ov::Tensor& outputTensors, std::vector<Object>& detections);

    bool setThresh(double thresh);

    bool setNmsThresh(double nmsthresh);

public:
    static const int ERROR_BASE = 0x0100;
    static const int ERROR_INVALID_INPUT = 0x0101;
    static const int ERROR_INVALID_INIT_ARGS = 0x0102;
    static const int PROCESS_OK = 0x1001;
    static const int INIT_OK = 0x1002;

private:
    ov::Core ie;
    std::shared_ptr<ov::Model> model{nullptr};
    ov::CompiledModel compiled_model;
    
private:
    std::vector<std::string> mLabels{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"};
    float mThresh = 0.25;
    float mNms = 0.45;
    int mNumClasses = 80;
    int inW, inH;
};

#endif //YOLOV5DETECTOR_HPP

