#include <iostream>
#include <inspireface.h>
#include <herror.h>
#include <format>
#include <chrono>
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

bool detect_face(FaceDetect &detect, FaceExtract &extract, cv::Mat &image, Face &src_face) {
    auto results = detect.Process(image);
    // for (auto &result : results) {
    //     std::cout << std::format("x1:{} y1:{}, x2:{}, y2:{}", result.x1, result.y1,
    //                              result.x2, result.y2)
    //               << std::endl;
    //     std::cout << "lmk:" << result.lmk[0] << " " << result.lmk[1] << " " <<
    //     result.lmk[2]
    //               << " " << result.lmk[3] << std::endl;
    // }
    if (results.size() == 0) {
        return false;
    }
    auto &res = results[0];

    float norm = 0;
    auto embedded = extract.Process(image, res.lmk, norm, false);
    // std::cout << "embedded:" << embedded << std::endl;

    src_face.x1 = res.x1;
    src_face.y1 = res.y1;
    src_face.x2 = res.x2;
    src_face.y2 = res.y2;
    memcpy(src_face.kps, res.lmk, sizeof(float) * 10);
    src_face.embedding = std::move(embedded);
    return true;
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

    std::string source_path = "source.jpg";
    cv::Mat source_image = cv::imread(source_path);

    std::string target_path = "target.jpg";
    cv::Mat target_image = cv::imread(target_path);

    auto start = std::chrono::steady_clock::now();

    const int COUNT = 1;
    for (int i = 0; i < COUNT; i++) {
        Face src_face;
        Face dst_face;
        if (!detect_face(detect, extract, source_image, src_face)) {
            return 0;
        }

        if (!detect_face(detect, extract, target_image, dst_face)) {
            return 0;
        }
        face_swapper.Process(target_image, src_face, dst_face);
    }

    auto end = std::chrono::steady_clock::now();

    // 4. Calculate the duration and count the ticks
    std::chrono::duration<double, std::milli> duration =
        end - start; // Duration in milliseconds (floating point)

    // Get the actual number of ticks (milliseconds in this case)
    double milliseconds = duration.count();

    std::cout << "fps:" << COUNT / (milliseconds / 1000) << std::endl;
    return 0;
}