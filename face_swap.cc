#include "face_swap.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/io/xnpy.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor.hpp>
#include "emap_loader.h"
#include "face_align.h"

FaceSwap::FaceSwap() : onnx_("FaceSwap") {}

FaceSwap::~FaceSwap() {}

void FaceSwap::setup_emap() {
    emap_ = EmapLoader::loadFromBinary("emap.bin");
    std::cout << "emap size:" << emap_.size << emap_.at<float>(0, 0) << std::endl;
}

bool FaceSwap::Initialize(const std::string &model_path) {
    if (!onnx_.Initialize(model_path)) {
        return false;
    }

    setup_emap();
    return true;
}

float embedding_norm(std::vector<float> &embedded) {
    float mse = 0.0f;
    for (const auto &one : embedded) {
        mse += one * one;
    }
    mse = std::sqrt(mse);
    return mse;
}

void normed_embedding(std::vector<float> &embedded) {
    float norm = embedding_norm(embedded);
    for (float &one : embedded) {
        one /= norm;
    }
}

cv::Mat tensor_to_mat(Ort::Value &tensor) {
    auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();

    std::vector<int> cv_shape(shape.begin(), shape.end());
    cv::Mat mat(cv_shape, CV_32F, tensor.GetTensorMutableData<float>());

    return mat.clone(); // 返回拷贝以确保数据安全
}

void clipMat(cv::Mat &mat, double lowerBound, double upperBound) {
    cv::min(mat, upperBound, mat); // Clamp upper bound
    cv::max(mat, lowerBound, mat); // Clamp lower bound
}

cv::Mat paste_back_face(const cv::Mat &target_img, const cv::Mat &bgr_fake1, const cv::Mat &aimg1,
                        const cv::Mat &M) {
    cv::Mat bgr_fake;
    bgr_fake1.convertTo(bgr_fake, CV_32FC3);
    cv::Mat aimg;
    aimg1.convertTo(aimg, CV_32FC3);

    cv::Mat fake_diff;
    cv::absdiff(bgr_fake, aimg, fake_diff);

    // numpy reduce(axis=2)
    std::vector<cv::Mat> channels;
    cv::split(fake_diff, channels);
    fake_diff = cv::Mat::zeros(fake_diff.rows, fake_diff.cols, CV_32F);
    for (auto &c : channels) {
        fake_diff += c;
    }
    fake_diff /= channels.size();

    // 边界置0
    fake_diff.rowRange(0, 2).setTo(0);
    fake_diff.rowRange(fake_diff.rows - 2, fake_diff.rows).setTo(0);
    fake_diff.colRange(0, 2).setTo(0);
    fake_diff.colRange(fake_diff.cols - 2, fake_diff.cols).setTo(0);

    // 计算逆变换
    cv::Mat IM;
    cv::invertAffineTransform(M, IM);

    // 变换回原图尺寸
    cv::Mat warped_fake, img_white, warped_diff;
    cv::warpAffine(bgr_fake, warped_fake, IM, target_img.size(), 0, 0);

    img_white = cv::Mat::ones(aimg.size(), CV_32F) * 255;
    cv::warpAffine(img_white, img_white, IM, target_img.size(), 0, 0);

    cv::warpAffine(fake_diff, warped_diff, IM, target_img.size(), 0, 0);

    // 阈值处理
    img_white.setTo(255, img_white > 20);

    float fthresh = 10;
    warped_diff.setTo(0, warped_diff < fthresh);
    warped_diff.setTo(255, warped_diff >= fthresh);

    cv::Mat img_mask = img_white;
    // 计算掩码尺寸
    cv::Mat mask_locations;
    cv::compare(img_mask, 255, mask_locations, cv::CMP_EQ);

    cv::Rect mask_rect = cv::boundingRect(mask_locations);
    int mask_size = std::sqrt(mask_rect.width * mask_rect.height);

    // 形态学操作
    int k = std::max(mask_size / 10, 10);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
    cv::erode(img_mask, img_mask, kernel);

    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(warped_diff, warped_diff, kernel);

    k = std::max(mask_size / 20, 5);
    cv::GaussianBlur(img_mask, img_mask, cv::Size(2 * k + 1, 2 * k + 1), 0);
    cv::GaussianBlur(warped_diff, warped_diff, cv::Size(11, 11), 0);

    // 归一化
    img_mask /= 255.0;
    warped_diff /= 255.0;

    // 合并图像
    cv::Mat img_mask_3ch(img_mask.rows, img_mask.cols, CV_32FC3);
    cv::merge(std::vector<cv::Mat>{img_mask, img_mask, img_mask}, img_mask_3ch);

    cv::Mat target_img_float;
    target_img.convertTo(target_img_float, CV_32FC3);
    auto mat_one =
        cv::Mat(img_mask_3ch.rows, img_mask_3ch.cols, CV_32FC3, cv::Scalar(1.0, 1.0, 1.0));
    cv::Mat fake_merged =
        img_mask_3ch.mul(warped_fake) + (mat_one - img_mask_3ch).mul(target_img_float);

    cv::Mat result;
    fake_merged.convertTo(result, CV_8UC3);
    cv::imwrite("merge.jpg", result);
    return result;
}

std::vector<Ort::Value> FaceSwap::RunModel(cv::Mat &blob, cv::Mat &latent_norm) {
    // 准备输入tensor
    std::vector<int64_t> input_shape = {1, 3, 128, 128}; // 根据模型调整

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                             OrtMemType::OrtMemTypeDefault);

    // 创建输入tensors
    std::vector<Ort::Value> input_tensors;

    // 源人脸输入
    Ort::Value source_input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, (float *)(blob.data), 3 * 128 * 128, input_shape.data(), input_shape.size());

    std::vector<int64_t> latent_shape = {1, 512};

    // 目标人脸输入
    Ort::Value target_input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, (float *)(latent_norm.data), 512, latent_shape.data(), latent_shape.size());

    input_tensors.push_back(std::move(source_input_tensor));
    input_tensors.push_back(std::move(target_input_tensor));

    return onnx_.RunModel(input_tensors);
}

void FaceSwap::Process(cv::Mat target_img, Face &src_face, Face &target_face) {
    cv::Mat cvaimg;
    cv::Mat M = norm_crop(target_img, target_face.kps, 128, cvaimg);

    // Define parameters for blobFromImage
    double scalefactor = 1.0 / 255.0;      // Scale pixel values to [0, 1]
    cv::Size size = cv::Size(128, 128);    // Target size for the input blob
    cv::Scalar mean = cv::Scalar(0, 0, 0); // Mean subtraction (can be model-specific)
    bool swapRB = true; // Swap R and B channels (common for models trained on BGR images)

    // Create the 4D blob
    cv::Mat blob = cv::dnn::blobFromImage(cvaimg, scalefactor, size, mean, swapRB);

    std::vector<float> source_normed_embedding = src_face.embedding;
    normed_embedding(source_normed_embedding);
    cv::Mat latent = cv::Mat(1, 512, CV_32FC1, (void *)source_normed_embedding.data());
    cv::Mat elatent = latent * emap_;
    cv::Mat latent_norm = elatent / cv::norm(elatent);

    auto output_tensors = RunModel(blob, latent_norm);

    auto &output_tensor = output_tensors.front();
    Ort::TensorTypeAndShapeInfo shape_info = output_tensor.GetTensorTypeAndShapeInfo();

    std::vector<std::size_t> xtensor_shape;
    for (int64_t dim : shape_info.GetShape()) {
        xtensor_shape.push_back(static_cast<std::size_t>(dim));
    }

    // Get data pointer
    float *output_data = output_tensor.GetTensorMutableData<float>();
    xt::xarray<float> xt_array =
        xt::adapt(output_data, 3 * 128 * 128, xt::no_ownership(), xtensor_shape);

    auto transposed_arr = xt::transpose(xt_array, {0, 2, 3, 1});
    xt::xarray<float> out_tensor =
        xt::eval(xt::flip(xt::view(transposed_arr, 0, xt::all(), xt::all(), xt::all()), 2));

    cv::Mat image_mat(128, 128, CV_32FC(3), out_tensor.data());
    cv::Mat new_image_mat = image_mat * 255;
    clipMat(new_image_mat, 0, 255);

    cv::Mat bgr_fake;
    new_image_mat.convertTo(bgr_fake, CV_8UC3);

    paste_back_face(target_img, bgr_fake, cvaimg, M);
}
