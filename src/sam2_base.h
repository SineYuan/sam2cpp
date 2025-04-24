#pragma once

#include <vector>
#include <string>
#include <memory>
#include <variant>
#include <tuple>
#include <unordered_map>

#include "MNN/Interpreter.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>

#include "tar.h"

#include "sam2_common.h"

namespace sam2 {

using namespace MNN::Express;

enum class ModelType {
    SAM,    // Standard SAM model
    ETAM    // Efficient Track Anything Model
};

struct ImageEmbedding {
    int original_height;
    int original_width;
    ModelType model_type;

    VARP image_embed = nullptr;
    VARP high_res_feat0 = nullptr;
    VARP high_res_feat1 = nullptr;
};

class SAM2Base {
public:
    SAM2Base();
    ~SAM2Base();

    Result<bool> init_img_encoder_decoder(const std::string& model_path, const Params& params);
    Result<bool> init_img_encoder_decoder(const std::vector<TarEntry>& entries, const Params& params);

    Result<bool> init_mem_encoder_atten(const std::string& model_path, const Params& params);
    Result<bool> init_mem_encoder_atten(const std::vector<TarEntry>& entries, const Params& params);

    Result<std::tuple<VARP, int, int>> preprocess_image(const std::string& img_path);
    Result<std::tuple<VARP, int, int>> preprocess_image(const std::vector<uint8_t>& img_buf);

    Result<ImageEmbedding> get_embedding(const std::string& img_path);
    Result<ImageEmbedding> get_embedding(const std::vector<uint8_t>& img_buf);
    Result<ImageEmbedding> get_embedding(const VARP& image_tensor, int original_height, int original_width);

    Result<std::vector<VARP>> predict(const ImageEmbedding& embedding, const std::vector<Point>& points, const std::vector<BBox>& bboxes);
    std::tuple<int, std::vector<float>, std::vector<float>> process_prompts(const std::vector<Point>& points, const std::vector<BBox>& bboxes, int img_height, int img_width);

    // New interfaces for running models
    Result<std::vector<VARP>> run_image_encoder(const VARP& image_tensor);
    Result<std::vector<VARP>> run_image_decoder(const std::vector<VARP>& inputs);
    Result<std::vector<VARP>> run_memory_encoder(const std::vector<VARP>& inputs);
    Result<std::vector<VARP>> run_memory_attention(const std::vector<VARP>& inputs);


protected:
    // run 完请立即copy outputs
    Result<bool> _run_image_encoder(const VARP& image_tensor);

    ModelType model_type_;

    std::unique_ptr<MNN::Interpreter> img_encoder_interpreter_;
    MNN::Session* img_encoder_session_ = nullptr;

    std::unique_ptr<MNN::Interpreter> mem_encoder_interpreter_;
    MNN::Session* mem_encoder_session_ = nullptr;

    std::unique_ptr<MNN::Express::Module> mem_attn_module_;
    std::unique_ptr<MNN::Express::Module> img_decoder_module_;
}; 

Mask convert_to_mask(const VARP& pred_mask, float score, float threshold=0.0);

} // namespace sam2
