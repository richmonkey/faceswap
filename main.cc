#include <iostream>
#include <inspireface.h>
#include <herror.h>
#include <format>
#include "face_swap.h"
#include "face_detect.h"
#include "face_extract.h"

template <typename T> std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
    const char *sep = "";
    for (const auto &element : v) {
        os << sep << element;
        sep = ", ";
    }
    return os;
}

int main() {
    FaceDetect detect(640);
    if (!detect.Initialize("models/buffalo_l/det_10g.onnx")) {
        std::cout << "initialize model fail" << std::endl;
        return 1;
    };

    FaceExtract extract;
    if (!extract.Initialize("models/buffalo_l/w600k_r50.onnx")) {
        std::cout << "initialize model fail" << std::endl;
        return 1;
    }

    FaceSwap face_swapper;
    if (!face_swapper.Initialize("models/inswapper_128.onnx")) {
        std::cerr << "Failed to initialize face swapper!" << std::endl;
        return -1;
    }

    Face src_face;
    Face dst_face;
    {
        std::string source_path = "source.jpg";
        cv::Mat source_image = cv::imread(source_path);

        auto results = detect.Process(source_image);
        for (auto &result : results) {
            std::cout << std::format("x1:{} y1:{}, x2:{}, y2:{}", result.x1, result.y1, result.x2,
                                     result.y2)
                      << std::endl;
            std::cout << "lmk:" << result.lmk[0] << " " << result.lmk[1] << " " << result.lmk[2]
                      << " " << result.lmk[3] << std::endl;
        }
        if (results.size() == 0) {
            return 0;
        }
        auto &res = results[0];

        float norm = 0;
        auto embedded = extract.Process(source_image, res.lmk, norm, false);
        std::cout << "embedded:" << embedded << std::endl;

        src_face.x1 = res.x1;
        src_face.y1 = res.y1;
        src_face.x2 = res.x2;
        src_face.y2 = res.y2;
        memcpy(src_face.kps, res.lmk, sizeof(float) * 10);
        src_face.embedding = std::move(embedded);
    }
    std::string target_path = "target.jpg";
    cv::Mat target_image = cv::imread(target_path);
    {

        auto results = detect.Process(target_image);
        for (auto &result : results) {
            std::cout << std::format("x1:{} y1:{}, x2:{}, y2:{}", result.x1, result.y1, result.x2,
                                     result.y2)
                      << std::endl;
            std::cout << "lmk:" << result.lmk[0] << " " << result.lmk[1] << " " << result.lmk[2]
                      << " " << result.lmk[3] << std::endl;
        }
        if (results.size() == 0) {
            return 0;
        }
        auto &res = results[0];

        float norm = 0;
        auto embedded = extract.Process(target_image, res.lmk, norm, false);
        std::cout << "embedded:" << embedded << std::endl;

        dst_face.x1 = res.x1;
        dst_face.y1 = res.y1;
        dst_face.x2 = res.x2;
        dst_face.y2 = res.y2;
        memcpy(dst_face.kps, res.lmk, sizeof(float) * 10);
        dst_face.embedding = std::move(embedded);
    }

    face_swapper.Process(target_image, src_face, dst_face);
    return 0;
}