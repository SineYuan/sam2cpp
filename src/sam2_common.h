#pragma once

#include <string>
#include <memory>
#include "expected.hpp"


namespace sam2 {

struct Point {
  size_t x;
  size_t y;
  int label; // 正点1，负点0
};

struct BBox {
  // 左上角坐标，右下角坐标( 以左上角为坐标原点的坐标系)
  size_t x_min;
  size_t y_min;
  size_t x_max;
  size_t y_max;
};

struct Params {
    std::string model_path;
};
  
// rgb image
struct Mask {
    std::vector<uint8_t> data;
    int height;
    int width;

    float score;
};

enum class Backend {
  AUTO,
  CPU,
  CUDA,
  OPENCL,
};

struct ErrorCode {
    int code;
    std::string message;
};

template<typename T>
using Result = tl::expected<T, ErrorCode>;

// Helper functions for creating results
template<typename T>
Result<T> Ok(T&& value) {
    return tl::expected<T, ErrorCode>(std::forward<T>(value));
}

template<typename T>
Result<T> Err(ErrorCode&& error) {
    return tl::expected<T, ErrorCode>(tl::unexpected(std::forward<ErrorCode>(error)));
}

template<typename T>
Result<T> Err(const ErrorCode& error) {
    return tl::expected<T, ErrorCode>(tl::unexpected(error));
}


} // namespace sam2