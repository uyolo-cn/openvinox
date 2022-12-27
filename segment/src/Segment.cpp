/**
 * @file Segment.cpp
 * @author Kris Wang (jiujiangluck@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-12-24
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "Segment.hpp"

Segment::Segment() {
}

STATUS Segment::init(std::string engineName) {
    if (engineName.empty()) return ERROR_INVALID_INIT_ARGS;
    
    // Read a model
    model = ie.read_model(engineName);

    // Inizialize Preprocessing for the model
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    // Specify input image format
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
    // Specify preprocess pipeline to input image without resizing
    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).mean({123.675, 116.28, 103.53}).scale({58.395, 57.12, 57.375});
    //  Specify model's input layout
    ppp.input().model().set_layout("NCHW");
    // Specify output results format
    ppp.output().tensor().set_element_type(ov::element::u8);
    // Embed above steps in the graph
    model = ppp.build();
    
    compiled_model = ie.compile_model(model, "AUTO");

    inH = compiled_model.input().get_shape()[1];

    inW = compiled_model.input().get_shape()[2];

    outH = compiled_model.output().get_shape()[1];

    outW = compiled_model.output().get_shape()[2];
    
    return Segment::INIT_OK;
}

void Segment::unInit() {
}

STATUS Segment::processImage(const cv::Mat &cv_image, cv::Mat &mask) {
    if (cv_image.empty()) return ERROR_INVALID_INPUT;

    cv::Mat resizeImg;
    cv::resize(cv_image, resizeImg, cv::Size(outW, outH), 0, 0, cv::INTER_LINEAR);

    ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), resizeImg.data);
    
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    
    const ov::Tensor& outputTensors = infer_request.get_output_tensor(0);
    mask = cv::Mat(cv::Size(outW, outH), CV_8UC1, outputTensors.data<void>());
    
    return Segment::PROCESS_OK;
}