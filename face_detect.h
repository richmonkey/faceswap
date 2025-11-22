#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <inspireface.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "onnx_session.h"

/** @struct FaceLoc
 *  @brief Struct representing standardized face landmarks for detection.
 *
 *  Contains coordinates for the face, detection score, and landmarks.
 */
typedef struct FaceLoc {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    float lmk[10];
} FaceLoc;

/** @typedef FaceLocList
 *  @brief List of FaceLoc structures.
 */
typedef std::vector<FaceLoc> FaceLocList;

class FaceDetect {
  public:
    /**
     * @brief Constructor for the FaceDetect class.
     * @param input_size The size of the input image for the neural network.
     * @param nms_threshold The threshold for non-maximum suppression.
     * @param cls_threshold The threshold for classification score.
     */
    FaceDetect(int input_size = 160, float nms_threshold = 0.4f, float cls_threshold = 0.5f);

    bool Initialize(const std::string &model_path);

    /** @brief Set non-maximum suppression threshold */
    void SetNmsThreshold(float mNmsThreshold);

    /** @brief Set face classification threshold */
    void SetClsThreshold(float mClsThreshold);

    /**
     * @brief Get the input size
     * @return int The input size
     */
    int GetInputSize() const;

    FaceLocList Process(cv::Mat &image);

  private:
    std::vector<Ort::Value> RunModel(cv::Mat &image);
    /**
     * @brief Applies non-maximum suppression to reduce overlapping detected faces.
     * @param input_faces List of detected faces to be filtered.
     * @param nms_threshold The threshold for non-maximum suppression.
     */
    static void _nms(FaceLocList &input_faces, float nms_threshold);

    /**
     * @brief Generates detection anchors based on stride.
     * @param stride The stride of the detection.
     * @param input_size The size of the input image.
     * @param num_anchors The number of anchors.
     * @param anchors The generated anchors.
     */
    void _generate_anchors(int stride, int input_size, int num_anchors,
                           std::vector<float> &anchors);

    /**
     * @brief Decodes network outputs to face locations.
     * @param cls_pred Classification predictions.
     * @param box_pred Bounding box predictions.
     * @param lmk_pred Landmark predictions.
     * @param stride The stride of the detection.
     * @param results Decoded face locations.
     */
    void _decode(const std::vector<float> &cls_pred, const std::vector<float> &box_pred,
                 const std::vector<float> &lmk_pred, int stride, std::vector<FaceLoc> &results);

  private:
    float m_nms_threshold_; ///< Threshold for non-maximum suppression.
    float m_cls_threshold_; ///< Threshold for classification score.
    int m_input_size_;      ///< Input size for the neural network model.

    OnnxSession onnx_;
};
