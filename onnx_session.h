#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <inspireface.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

class OnnxSession {
  public:
    OnnxSession(const char *logid) : env_(ORT_LOGGING_LEVEL_WARNING, logid) {}
    bool Initialize(const std::string &model_path) {
        try {
            // 初始化换脸模型
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

            // 获取输入输出节点名称
            Ort::AllocatorWithDefaultOptions allocator;

            // 换脸模型节点
            size_t num_input_nodes = session_->GetInputCount();
            for (size_t i = 0; i < num_input_nodes; i++) {
                input_names__.push_back(session_->GetInputNameAllocated(i, allocator));
                input_names_.push_back(input_names__.back().get());
            }

            size_t num_output_nodes = session_->GetOutputCount();
            for (size_t i = 0; i < num_output_nodes; i++) {
                output_names__.push_back(session_->GetOutputNameAllocated(i, allocator));
                output_names_.push_back(output_names__.back().get());
            }

            // setup_emap();
            std::cout << "Models initialized successfully!" << std::endl;

            return true;

        } catch (const std::exception &e) {
            std::cerr << "Failed to initialize models: " << e.what() << std::endl;
            return false;
        }
    }

    std::vector<Ort::Value> RunModel(std::vector<Ort::Value> &input_tensors) {
        auto output_tensors =
            session_->Run(Ort::RunOptions{nullptr}, input_names_.data(), input_tensors.data(),
                          input_tensors.size(), output_names_.data(), output_names_.size());
        return output_tensors;
    }

  private:
    // ONNX环境
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    std::vector<const char *> input_names_;
    std::vector<const char *> output_names_;

    std::vector<Ort::AllocatedStringPtr> input_names__;
    std::vector<Ort::AllocatedStringPtr> output_names__;
};