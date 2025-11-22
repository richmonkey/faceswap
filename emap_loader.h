// emap_loader.h
#pragma once

#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

class EmapLoader {
  public:
    // 方法1: 加载带头的二进制文件
    static cv::Mat loadFromBinary(const std::string &filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open binary file: " + filename);
        }

        // 读取魔数
        char magic[4];
        file.read(magic, 4);
        if (std::string(magic, 4) != "EMAP") {
            throw std::runtime_error("Invalid binary file format");
        }

        // 读取数据类型
        int data_type;
        file.read(reinterpret_cast<char *>(&data_type), sizeof(int));
        if (data_type != 1) { // 1 表示 float32
            throw std::runtime_error("Unsupported data type, expected float32");
        }

        // 读取维度数量
        int num_dims;
        file.read(reinterpret_cast<char *>(&num_dims), sizeof(int));

        // 读取每个维度的大小
        std::vector<int> dims(num_dims);
        for (int i = 0; i < num_dims; ++i) {
            file.read(reinterpret_cast<char *>(&dims[i]), sizeof(int));
        }

        // 计算总元素数量
        int total_elements = 1;
        for (int dim : dims) {
            total_elements *= dim;
        }

        // 创建cv::Mat
        cv::Mat mat;
        if (dims.size() == 1) {
            mat = cv::Mat(dims[0], 1, CV_32F);
        } else if (dims.size() == 2) {
            mat = cv::Mat(dims[0], dims[1], CV_32F);
        } else {
            // 对于高维数组，展平为2D
            mat = cv::Mat(1, total_elements, CV_32F);
        }

        // 读取数据
        file.read(reinterpret_cast<char *>(mat.data), total_elements * sizeof(float));

        std::cout << "Loaded emap from binary: [";
        for (size_t i = 0; i < dims.size(); ++i) {
            std::cout << dims[i] << (i < dims.size() - 1 ? " x " : "");
        }
        std::cout << "]" << std::endl;

        return mat;
    }

    // 方法2: 加载原始二进制文件（需要提前知道形状）
    static cv::Mat loadFromRawBinary(const std::string &filename, int rows, int cols) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file) {
            throw std::runtime_error("Cannot open raw binary file: " + filename);
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        int expected_size = rows * cols * sizeof(float);
        if (size != expected_size) {
            throw std::runtime_error("File size doesn't match expected dimensions");
        }

        cv::Mat mat(rows, cols, CV_32F);
        file.read(reinterpret_cast<char *>(mat.data), size);

        std::cout << "Loaded emap from raw binary: " << mat.size() << std::endl;
        return mat;
    }

    // 方法3: 加载文本文件
    static cv::Mat loadFromText(const std::string &filename) {
        std::ifstream file(filename);
        if (!file) {
            throw std::runtime_error("Cannot open text file: " + filename);
        }

        // 读取第一行获取形状
        std::string line;
        std::getline(file, line);
        std::istringstream shape_stream(line);

        std::vector<int> dims;
        int dim;
        while (shape_stream >> dim) {
            dims.push_back(dim);
        }

        if (dims.empty()) {
            throw std::runtime_error("No shape information found in text file");
        }

        // 计算总元素数量
        int total_elements = 1;
        for (int d : dims) {
            total_elements *= d;
        }

        // 创建cv::Mat
        cv::Mat mat;
        if (dims.size() == 1) {
            mat = cv::Mat(dims[0], 1, CV_32F);
        } else if (dims.size() == 2) {
            mat = cv::Mat(dims[0], dims[1], CV_32F);
        } else {
            mat = cv::Mat(1, total_elements, CV_32F);
        }

        // 读取数据
        float *data = mat.ptr<float>();
        int index = 0;
        while (file >> data[index] && index < total_elements) {
            ++index;
        }

        if (index != total_elements) {
            throw std::runtime_error("Data count doesn't match shape information");
        }

        std::cout << "Loaded emap from text: [";
        for (size_t i = 0; i < dims.size(); ++i) {
            std::cout << dims[i] << (i < dims.size() - 1 ? " x " : "");
        }
        std::cout << "]" << std::endl;

        return mat;
    }

    // 自动检测并加载（推荐使用）
    static cv::Mat autoLoad(const std::string &base_path, int known_rows = -1,
                            int known_cols = -1) {
        // 尝试按优先级加载不同格式
        std::vector<std::pair<std::string, std::function<cv::Mat()>>> loaders = {
            {base_path + ".bin", [&]() { return loadFromBinary(base_path + ".bin"); }},
            {base_path + ".npy",
             [&]() {
                 // 简化版的NPY加载，实际使用时可以实现完整的NPY解析
                 if (known_rows > 0 && known_cols > 0) {
                     return loadFromRawBinary(base_path + ".raw", known_rows, known_cols);
                 }
                 throw std::runtime_error("Need known dimensions for NPY/RAW loading");
             }},
            {base_path + ".txt", [&]() { return loadFromText(base_path + ".txt"); }}};

        for (const auto &loader : loaders) {
            std::ifstream test_file(loader.first);
            if (test_file) {
                std::cout << "Loading from: " << loader.first << std::endl;
                return loader.second();
            }
        }

        throw std::runtime_error("No emap file found with base path: " + base_path);
    }
};