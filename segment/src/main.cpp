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
#include "Segment.hpp"

int main() {
    // Create a segment
    auto mDetector = std::make_shared<Segment>();
    mDetector->init("../model/segment.xml");

    cv::Mat image = cv::imread("../samples/face.jpeg");
    cv::Mat mask;
    auto start = std::chrono::system_clock::now();
    mDetector->processImage(image, mask);
    std::cout << "infer time cost: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count()
            << "ms"
            << std::endl;
    
    cv::imwrite("out.jpg", mask);

    return 0;
}