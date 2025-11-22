// FaceSwap.h
#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include "onnx_session.h"

class Face {
  public:
    float x1;
    float y1;
    float x2;
    float y2;
    float kps[10];
    std::vector<float> embedding;
};

class FaceSwap {
  public:
    FaceSwap();
    ~FaceSwap();

    bool Initialize(const std::string &model_path);
    void Process(cv::Mat target_img, Face &src_face, Face &target_face);

  private:
    std::vector<Ort::Value> RunModel(cv::Mat &blob, cv::Mat &latent_norm);

    void setup_emap();

    OnnxSession onnx_;

    cv::Mat emap_;
};