#include "sam2_image.h"

#include <chrono>

#include "spdlog/spdlog.h"
#include "cv/types.hpp"

#include "tar.h"
#include "sam2_base.h"

namespace sam2 {

class SAM2Image::Impl : public SAM2Base {
public:
    Impl() = default;
    ~Impl() = default;
};

SAM2Image::SAM2Image() : m_pimpl(std::make_unique<Impl>()) {}
SAM2Image::~SAM2Image() = default;

Result<bool> SAM2Image::initialize(const std::string& model_path, const Params& params) {
    if (!m_pimpl) {
        return Err<bool>(ErrorCode{-1, "Failed to create implementation"});
    }
    return m_pimpl->init_img_encoder_decoder(model_path, params);
}

Result<bool> SAM2Image::initialize(uint8_t* buffer, size_t size, const Params& params) {
    if (!m_pimpl) {
        return Err<bool>(ErrorCode{-1, "Failed to create implementation"});
    }
    auto entries = parseTar(buffer, size);
    return m_pimpl->init_img_encoder_decoder(entries, params);
}

Result<std::shared_ptr<ImageEmbedding>> SAM2Image::get_embedding(const std::string& img_path) {
    if (!m_pimpl) {
        return Err<std::shared_ptr<ImageEmbedding>>(ErrorCode{-1, "Implementation not initialized"});
    }
    
    auto result = m_pimpl->get_embedding(img_path);
    if (!result.has_value()) {
        return Err<std::shared_ptr<ImageEmbedding>>(result.error());
    }
    return std::make_shared<ImageEmbedding>(std::move(result.value()));
}

Result<std::shared_ptr<ImageEmbedding>> SAM2Image::get_embedding(const std::vector<uint8_t>& img_buf) {
    if (!m_pimpl) {
        return Err<std::shared_ptr<ImageEmbedding>>(ErrorCode{-1, "Implementation not initialized"});
    }
    
    auto result = m_pimpl->get_embedding(img_buf);
    if (!result.has_value()) {
        return Err<std::shared_ptr<ImageEmbedding>>(result.error());
    }
    return std::make_shared<ImageEmbedding>(std::move(result.value()));
}

Result<Mask> SAM2Image::predict(const std::shared_ptr<ImageEmbedding>& embedding, const std::vector<Point>& points, const std::vector<BBox>& bboxes) {
    if (!m_pimpl) {
        return Err<Mask>(ErrorCode{-1, "Implementation not initialized"});
    }
    if (!embedding) {
        return Err<Mask>(ErrorCode{-1, "Embedding invalid"});
    }
    
    auto outs = m_pimpl->predict(*embedding, points, bboxes);
    if (!outs.has_value()) {
        return Err<Mask>(ErrorCode{-1, "Failed to predict"});
    }

    auto& pred_mask = outs.value()[2];
    int h, w, c;
    MNN::CV::getVARPSize(pred_mask, &h, &w, &c);


    Mask mask;
    mask.height = h;
    mask.width = w;

    mask.data.resize(h * w);
    auto pred_mask_data = pred_mask->readMap<float>();
    for (int i = 0; i < mask.data.size(); i++) {
        mask.data[i] = pred_mask_data[i] > 0 ? 255 : 0;
    }

    mask.score = outs.value()[3]->readMap<float>()[0];

    return mask;
}


} // namespace sam2
