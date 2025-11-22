#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/io/xnpy.hpp>
#include <xtensor/views/xview.hpp>
#include "face_detect.h"

bool SortBoxSizeAdapt(const FaceLoc &a, const FaceLoc &b);

FaceDetect::FaceDetect(int input_size, float nms_threshold, float cls_threshold)
    : m_nms_threshold_(nms_threshold), m_cls_threshold_(cls_threshold), m_input_size_(input_size),
      onnx_("FaceDetect") {}

bool FaceDetect::Initialize(const std::string &model_path) { return onnx_.Initialize(model_path); }

template <class T> xt::xarray<T> value_to_xtensor(Ort::Value &output_tensor) {
    Ort::TensorTypeAndShapeInfo shape_info = output_tensor.GetTensorTypeAndShapeInfo();

    auto elem_type = shape_info.GetElementType();
    int size = 1;
    std::vector<std::size_t> xtensor_shape;
    for (int64_t dim : shape_info.GetShape()) {
        xtensor_shape.push_back(static_cast<std::size_t>(dim));
        size *= dim;
    }

    // Get data pointer
    T *output_data = output_tensor.GetTensorMutableData<T>();
    xt::xarray<T> xt_array = xt::adapt(output_data, size, xt::no_ownership(), xtensor_shape);
    return xt_array;
}

FaceLocList FaceDetect::Process(cv::Mat &img) {
    int ori_w = img.cols;
    int ori_h = img.rows;

    cv::Mat resized_img;
    float scale;
    if (ori_w == m_input_size_ && ori_h == m_input_size_) {
        scale = 1.0f;
        resized_img = img;
    } else {
        float img_ratio = (1.0 * img.rows) / img.cols;
        float model_ratio = 1.0;

        int new_height;
        int new_width;
        if (img_ratio > model_ratio) {
            new_height = m_input_size_;
            new_width = int(new_height / img_ratio);
        } else {
            new_width = m_input_size_;
            new_height = int(new_width * img_ratio);
        }
        cv::Mat m;
        cv::resize(img, m, cv::Size(new_width, new_height));
        scale = float(new_height) / img.rows;
        cv::copyMakeBorder(m, resized_img, 0, m_input_size_ - new_height, 0,
                           m_input_size_ - new_width, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }

    const float input_mean = 127.5;
    const float input_std = 128.0;
    // Define parameters for blobFromImage
    double scalefactor = 1.0 / input_std; // Scale pixel values to [0, 1]
    cv::Size size = resized_img.size();
    cv::Scalar mean =
        cv::Scalar(input_mean, input_mean, input_mean); // Mean subtraction (can be model-specific)
    bool swapRB = true; // Swap R and B channels (common for models trained on BGR images)

    cv::Mat blob = cv::dnn::blobFromImage(resized_img, scalefactor, size, mean, swapRB);

    std::vector<Ort::Value> outputs = RunModel(blob);

    std::vector<std::vector<float>> output_vecs;

    for (Ort::Value &output : outputs) {
        int size = 1;
        std::vector<std::size_t> xtensor_shape;

        Ort::TensorTypeAndShapeInfo shape_info = output.GetTensorTypeAndShapeInfo();
        for (int64_t dim : shape_info.GetShape()) {
            xtensor_shape.push_back(static_cast<std::size_t>(dim));
            size *= dim;
        }
        float *output_data = output.GetTensorMutableData<float>();
        std::vector<float> o(output_data, output_data + size);
        output_vecs.push_back(o);
    }

    std::vector<FaceLoc> results;
    std::vector<int> strides = {8, 16, 32};
    for (int i = 0; i < strides.size(); ++i) {
        const std::vector<float> &tensor_cls = output_vecs[i];
        const std::vector<float> &tensor_box = output_vecs[i + 3];
        const std::vector<float> &tensor_lmk = output_vecs[i + 6];
        _decode(tensor_cls, tensor_box, tensor_lmk, strides[i], results);
    }

    _nms(results, m_nms_threshold_);
    std::sort(results.begin(), results.end(), [](FaceLoc a, FaceLoc b) {
        return (a.y2 - a.y1) * (a.x2 - a.x1) > (b.y2 - b.y1) * (b.x2 - b.x1);
    });
    for (auto &face : results) {
        face.x1 = face.x1 / scale;
        face.y1 = face.y1 / scale;
        face.x2 = face.x2 / scale;
        face.y2 = face.y2 / scale;
        for (int i = 0; i < 5; ++i) {
            face.lmk[i * 2 + 0] = face.lmk[i * 2 + 0] / scale;
            face.lmk[i * 2 + 1] = face.lmk[i * 2 + 1] / scale;
        }
    }

    return results;
}

std::vector<Ort::Value> FaceDetect::RunModel(cv::Mat &image) {

    // 准备输入tensor
    std::vector<int64_t> input_shape = {1, 3, m_input_size_, m_input_size_}; // 根据模型调整

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                             OrtMemType::OrtMemTypeDefault);

    // 创建输入tensors
    std::vector<Ort::Value> input_tensors;

    // 源人脸输入
    Ort::Value source_input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, (float *)(image.data), 3 * m_input_size_ * m_input_size_, input_shape.data(),
        input_shape.size());

    input_tensors.push_back(std::move(source_input_tensor));

    return onnx_.RunModel(input_tensors);
}

void FaceDetect::_nms(std::vector<FaceLoc> &input_faces, float nms_threshold) {
    std::sort(input_faces.begin(), input_faces.end(),
              [](FaceLoc a, FaceLoc b) { return a.score > b.score; });
    std::vector<float> area(input_faces.size());
    for (int i = 0; i < int(input_faces.size()); ++i) {
        area[i] = (input_faces.at(i).x2 - input_faces.at(i).x1 + 1) *
                  (input_faces.at(i).y2 - input_faces.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_faces.size()); ++i) {
        for (int j = i + 1; j < int(input_faces.size());) {
            float xx1 = (std::max)(input_faces[i].x1, input_faces[j].x1);
            float yy1 = (std::max)(input_faces[i].y1, input_faces[j].y1);
            float xx2 = (std::min)(input_faces[i].x2, input_faces[j].x2);
            float yy2 = (std::min)(input_faces[i].y2, input_faces[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (area[i] + area[j] - inter);
            if (ovr >= nms_threshold) {
                input_faces.erase(input_faces.begin() + j);
                area.erase(area.begin() + j);
            } else {
                j++;
            }
        }
    }
}

void FaceDetect::_generate_anchors(int stride, int input_size, int num_anchors,
                                   std::vector<float> &anchors) {
    int height = ceil(input_size / stride);
    int width = ceil(input_size / stride);
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            for (int k = 0; k < num_anchors; ++k) {
                anchors.push_back(i * stride);
                anchors.push_back(j * stride);
            }
        }
    }
}

void FaceDetect::_decode(const std::vector<float> &cls_pred, const std::vector<float> &box_pred,
                         const std::vector<float> &lmk_pred, int stride,
                         std::vector<FaceLoc> &results) {
    std::vector<float> anchors_center;
    _generate_anchors(stride, m_input_size_, 2, anchors_center);

    for (int i = 0; i < anchors_center.size() / 2; ++i) {
        if (cls_pred[i] > m_cls_threshold_) {
            FaceLoc faceInfo;
            float cx = anchors_center[i * 2 + 0];
            float cy = anchors_center[i * 2 + 1];
            float x1 = cx - box_pred[i * 4 + 0] * stride;
            float y1 = cy - box_pred[i * 4 + 1] * stride;
            float x2 = cx + box_pred[i * 4 + 2] * stride;
            float y2 = cy + box_pred[i * 4 + 3] * stride;
            faceInfo.x1 = x1;
            faceInfo.y1 = y1;
            faceInfo.x2 = x2;
            faceInfo.y2 = y2;
            faceInfo.score = cls_pred[i];
            //            if (use_kps_) {
            for (int j = 0; j < 5; ++j) {
                float px = cx + lmk_pred[i * 10 + j * 2 + 0] * stride;
                float py = cy + lmk_pred[i * 10 + j * 2 + 1] * stride;
                faceInfo.lmk[j * 2 + 0] = px;
                faceInfo.lmk[j * 2 + 1] = py;
            }
            //            }
            results.push_back(faceInfo);
        }
        std::sort(results.begin(), results.end(), SortBoxSizeAdapt);
    }
}

void FaceDetect::SetNmsThreshold(float mNmsThreshold) { m_nms_threshold_ = mNmsThreshold; }

void FaceDetect::SetClsThreshold(float mClsThreshold) { m_cls_threshold_ = mClsThreshold; }

bool SortBoxSizeAdapt(const FaceLoc &a, const FaceLoc &b) {
    int sq_a = (a.y2 - a.y1) * (a.x2 - a.x1);
    int sq_b = (b.y2 - b.y1) * (b.x2 - b.x1);
    return sq_a > sq_b;
}

int FaceDetect::GetInputSize() const { return m_input_size_; }
