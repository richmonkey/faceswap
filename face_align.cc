#include <iostream>
#include <fstream>
#include <iomanip>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/io/xnpy.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor.hpp>
#include "emap_loader.h"

// similarity transform destination points
const std::vector<cv::Point2f> SIMILARITY_TRANSFORM_DEST = {{38.2946, 51.6963},
                                                            {73.5318, 51.5014},
                                                            {56.0252, 71.7366},
                                                            {41.5493, 92.3655},
                                                            {70.7299, 92.2041}};
cv::Mat norm_crop(cv::Mat &img, float kps[10], int size, cv::Mat &dst_mat) {
    std::vector<cv::Point2f> pointsFive;

    for (size_t i = 0; i < 5; i++) {
        pointsFive.push_back(cv::Point2f(kps[i * 2], kps[i * 2 + 1]));
    }

    std::vector<cv::Point2f> dst(SIMILARITY_TRANSFORM_DEST);

    float ratio;
    float diff_x;
    if (size % 112 == 0) {
        ratio = float(size) / 112.0;
        diff_x = 0;
    } else {
        ratio = float(size) / 128.0;
        diff_x = 8.0 * ratio;
    }

    for (auto &d : dst) {
        d.x = d.x * ratio;
        d.y = d.y * ratio;
    }

    for (auto &d : dst) {
        d.x = d.x + diff_x;
    }

    cv::Mat inliers;
    cv::Mat M = cv::estimateAffinePartial2D(pointsFive, dst, inliers, cv::LMEDS, 3.0);
    cv::warpAffine(img, dst_mat, M, cv::Size(size, size), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                   cv::Scalar(0, 0, 0));
    return M;
}