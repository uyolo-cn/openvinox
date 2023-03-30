/**
 * @file Yolov5Detector.cpp
 * @author Kris Wang (jiujiangluck@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-12-24
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "Yolov5Detector.hpp"

static inline float BoxIOU(const cv::Rect2d &b1, const cv::Rect2d &b2) {
    cv::Rect2d inter = b1 & b2;    
    return inter.area() / (b1.area() + b2.area() - inter.area());
}

static inline void nms(std::vector<Yolov5Detector::Object> &vecBoxObjs, float nmsThresh) {
    std::sort(vecBoxObjs.begin(), vecBoxObjs.end(), [](const Yolov5Detector::Object &b1, const Yolov5Detector::Object &b2){return b1.prob > b2.prob;});
    for (int i = 0; i < vecBoxObjs.size(); ++i) {
        if (vecBoxObjs[i].prob == 0) {
            continue;
        }
        for (int j = i + 1; j < vecBoxObjs.size(); ++j) {
            if (vecBoxObjs[j].prob == 0) {
                continue;
            }
            if (BoxIOU(vecBoxObjs[i].rect, vecBoxObjs[j].rect) >= nmsThresh) {
                vecBoxObjs[j].prob = 0;
            }            
        }
    }
    for (auto iter = vecBoxObjs.begin(); iter != vecBoxObjs.end(); ++iter) {
        if (iter->prob < 0.01) {
            vecBoxObjs.erase(iter);
            --iter;
        }
    }
}

static inline int clamp(int a, int b, int c) {
    if (a < b) return b;
    else if (a > c) return c;
    else return a;
}

static inline void scaleCoords(const cv::Size& imageShape, cv::Rect& coords, const cv::Size& imageOriginalShape) {
    float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height,
                          (float)imageShape.width / (float)imageOriginalShape.width);

    int pad[2] = {(int) (( (float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f),
                  (int) (( (float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f)};

    coords.x = clamp((int) std::round(((float)(coords.x - pad[0]) / gain)), 0, imageOriginalShape.width);
    coords.y = clamp((int) std::round(((float)(coords.y - pad[1]) / gain)), 0, imageOriginalShape.height);

    coords.width = clamp((int) std::round(((float)coords.width / gain)), 0, imageOriginalShape.width);
    coords.height = clamp((int) std::round(((float)coords.height / gain)), 0, imageOriginalShape.height);
}

static inline cv::Mat letterbox_image(const cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

Yolov5Detector::Yolov5Detector() {
}

STATUS Yolov5Detector::init(std::string engineName, double thresh, double nmsthresh) {
    if (engineName.empty()) return ERROR_INVALID_INIT_ARGS;
    
    setThresh(thresh);
    setNmsThresh(nmsthresh);
    
    // Read a model
    model = ie.read_model(engineName);
    // Inizialize Preprocessing for the model
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    // Specify input image format
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
    // Specify preprocess pipeline to input image without resizing
    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255., 255., 255.});
    //  Specify model's input layout
    ppp.input().model().set_layout("NCHW");
    // Specify output results format
    ppp.output().tensor().set_element_type(ov::element::f32);
    // Embed above steps in the graph
    model = ppp.build();
    
    compiled_model = ie.compile_model(model, "AUTO");

    std::cout << compiled_model.input().get_shape() << std::endl; // NHWC BGR {1, 640, 640, 3} 

    inH = compiled_model.input().get_shape()[1];

    inW = compiled_model.input().get_shape()[2];
    
    return Yolov5Detector::INIT_OK;
}

bool Yolov5Detector::setThresh(double thresh) {
    mThresh = thresh;
    return true;
}

bool Yolov5Detector::setNmsThresh(double nmsthresh) {
    mNms = nmsthresh;
    return true;
}

void Yolov5Detector::unInit() {
    if (!mLabels.empty()) {
        std::vector<std::string>().swap(mLabels);
    }
}

void Yolov5Detector::postprocess(const cv::Size& resizedImageShape, const cv::Size& originalImageShape, const ov::Tensor& outputTensors, std::vector<Object>& detections) {
    const ov::Shape outputShape = outputTensors.get_shape();
    auto* batchData = outputTensors.data<const float>();

    size_t count = outputTensors.get_size();
    std::vector<float> output(batchData, batchData + count);
    
    int elementsInBatch = (int)(outputShape[1] * outputShape[2]);

    for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[2]) {
        Object oneResult;

        oneResult.prob = it[4];

        auto idx = std::max_element(it + 5, it + 5 + mNumClasses) - (it + 5);
        
        // filter out the bbox unter prob
        if (oneResult.prob < mThresh) continue;

        oneResult.prob *= it[5 + idx];

        // filter out the bbox unter prob
        if (oneResult.prob < mThresh) continue;

        oneResult.name = mLabels[idx];

        oneResult.rect = cv::Rect((int)(it[0] - it[2] / 2), (int)(it[1] - it[3] / 2), (int)(it[2]), (int)(it[3]));
        
        scaleCoords(resizedImageShape, oneResult.rect, originalImageShape);

        detections.emplace_back(oneResult);
    }
    
    //nms
    nms(detections, mNms);
}

STATUS Yolov5Detector::processImage(const cv::Mat &cv_image, std::vector<Object> &result) {
    if (cv_image.empty()) return ERROR_INVALID_INPUT;
    
    cv::Mat resizedImage = letterbox_image(cv_image, inW, inH);
    
    ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), resizedImage.data);
    
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    
    postprocess(cv::Size(inW, inH), cv_image.size(), infer_request.get_output_tensor(0), result);
    
    return Yolov5Detector::PROCESS_OK;
}
