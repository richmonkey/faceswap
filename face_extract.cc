
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

FaceExtract::FaceExtract() : onnx_("FaceExtract"), input_size_(112) {}

bool FaceExtract::Initialize(const std::string &model_path) { return onnx_.Initialize(model_path); }

Embedded FaceExtract::Process(cv::Mat &img, float kps[10], float &norm, bool normalize) {
    cv::Mat aimg;
    norm_crop(img, kps, input_size_, aimg);

    const float input_mean = 127.5;
    const float input_std = 127.5;
    // Define parameters for blobFromImage
    double scalefactor = 1.0 / input_std; // Scale pixel values to [0, 1]
    cv::Size size(input_size_, input_size_);
    cv::Scalar mean =
        cv::Scalar(input_mean, input_mean, input_mean); // Mean subtraction (can be model-specific)
    bool swapRB = true; // Swap R and B channels (common for models trained on BGR images)

    cv::Mat blob = cv::dnn::blobFromImage(aimg, scalefactor, size, mean, swapRB);

    // 准备输入tensor
    std::vector<int64_t> input_shape = {1, 3, input_size_, input_size_}; // 根据模型调整

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                             OrtMemType::OrtMemTypeDefault);

    // 创建输入tensors
    std::vector<Ort::Value> input_tensors;

    // 源人脸输入
    Ort::Value source_input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, (float *)(blob.data), 3 * input_size_ * input_size_, input_shape.data(),
        input_shape.size());

    input_tensors.push_back(std::move(source_input_tensor));
    auto outputs = onnx_.RunModel(input_tensors);

    auto &output = outputs.front();

    std::vector<float> embedded = onnx_value_to_vector<float>(output);

    float mse = 0.0f;
    for (const auto &one : embedded) {
        mse += one * one;
    }
    mse = sqrt(mse);
    norm = mse;

    if (normalize) {
        for (float &one : embedded) {
            one /= mse;
        }
    }

    return embedded;
}

template <class T> std::vector<T> onnx_value_to_vector(Ort::Value &output) {
    int size = 1;
    std::vector<std::size_t> xtensor_shape;

    Ort::TensorTypeAndShapeInfo shape_info = output.GetTensorTypeAndShapeInfo();
    for (int64_t dim : shape_info.GetShape()) {
        xtensor_shape.push_back(static_cast<std::size_t>(dim));
        size *= dim;
    }
    T *output_data = output.GetTensorMutableData<T>();

    return std::vector<T>(output_data, output_data + size);
}