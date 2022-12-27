/**
 * @file main.cpp
 * @author Kris Wang (jiujiangluck@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-12-24
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <iostream>
#include <chrono>
#include "Yolov5Detector.hpp"

int main() {
    // Create a yolov5 detector
    auto mDetector = std::make_shared<Yolov5Detector>();
    mDetector->init("../model/yolov5s.xml", 0.25, 0.45);

    // Read image and Infer
    cv::Mat image = cv::imread("../samples/dog.jpg");
    std::vector<Yolov5Detector::Object> detectResults;
    auto start = std::chrono::system_clock::now();
    mDetector->processImage(image, detectResults);
    std::cout << "infer time cost: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count()
            << "ms"
            << std::endl;

    // Image visualization
    for (auto obj : detectResults) {
        cv::rectangle(image, obj.rect, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::putText(image, obj.name, cv::Point(obj.rect.x, obj.rect.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 0, 255), 2);
    }
    cv::imwrite("out.jpg", image);
    
    return 0;
}