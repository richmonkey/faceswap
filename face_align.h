#ifndef FACE_ALIGN_H
#define FACE_ALIGN_H
#include <opencv2/opencv.hpp>

cv::Mat norm_crop(cv::Mat &img, float kps[10], int size, cv::Mat &dst_mat);
#endif