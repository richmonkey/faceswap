#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/io/xnpy.hpp>
#include <xtensor/views/xview.hpp>
#include "face_detect.h"
#include "onnx_session.h"
#include "face_align.h"

typedef std::vector<float> Embedded;

template <class T> std::vector<T> onnx_value_to_vector(Ort::Value &value);

class FaceExtract {
  public:
    FaceExtract();

    bool Initialize(const std::string &model_path);
    Embedded Process(cv::Mat &bgr_affine, float kps[10], float &norm, bool normalize);

  private:
    OnnxSession onnx_;
    int input_size_;
};