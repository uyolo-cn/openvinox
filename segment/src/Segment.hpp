/**
 * @file Segment.hpp
 * @author Kris Wang (jiujiangluck@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-12-24
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef SEGMENT_HPP
#define SEGMENT_HPP
#include <iostream>
#include <string>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

#define STATUS int

class Segment {

public:

    Segment();

    STATUS init(std::string engineName);

    void unInit();

    STATUS processImage(const cv::Mat &image, cv::Mat &mask);

public:
    static const int ERROR_BASE = 0x0100;
    static const int ERROR_INVALID_INPUT = 0x0101;
    static const int ERROR_INVALID_INIT_ARGS = 0x0102;
    static const int PROCESS_OK = 0x1001;
    static const int INIT_OK = 0x1002;

public:
    int inH, inW, outH, outW;

private:
    ov::Core ie;
    std::shared_ptr<ov::Model> model{nullptr};
    ov::CompiledModel compiled_model;
};

#endif //SEGMENT_HPP

