// FaceSwap.h
#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

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
    FaceSwap(const std::string &swap_model_path);
    ~FaceSwap();

    bool Initialize();
    void Process(cv::Mat target_img, Face &src_face, Face &target_face);

  private:
    std::vector<Ort::Value> RunSwapModel(cv::Mat &blob, cv::Mat &latent_norm);

    void setup_emap();

    std::string swap_model_path_;

    // ONNX环境
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> swap_session_;

    std::vector<const char *> swap_input_names_;
    std::vector<const char *> swap_output_names_;

    std::vector<Ort::AllocatedStringPtr> swap_input_names__;
    std::vector<Ort::AllocatedStringPtr> swap_output_names__;

    cv::Mat emap_;
};