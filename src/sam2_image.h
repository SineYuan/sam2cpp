#pragma once

//#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <variant>

#include "sam2_common.h"

namespace sam2 {

class ImageEmbedding;

class SAM2Image {
public:
    SAM2Image();
    ~SAM2Image();

    Result<bool> initialize(const std::string& model_path, const Params& params);
    Result<bool> initialize(uint8_t* buffer, size_t size, const Params& params);

    Result<std::shared_ptr<ImageEmbedding>> get_embedding(const std::string& img_path);
    Result<std::shared_ptr<ImageEmbedding>> get_embedding(const std::vector<uint8_t>& img_buf);

    Result<Mask> predict(const std::shared_ptr<ImageEmbedding>& embedding, const std::vector<Point>& points, const std::vector<BBox>& bboxes);

private:
    class Impl;
    std::unique_ptr<Impl> m_pimpl;
};

} // namespace sam2
