#include "sam2_base.h"

#include <chrono>
#include <fstream>
#include <iterator>
#include <filesystem>

#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"
#include "cv/cv.hpp"
#include "cv/types.hpp"

#include "tar.h"


using namespace MNN::Express;

namespace sam2 {

const int IMG_SIZE = 1024;

const std::vector<std::string> SAM_IMAGE_DECODER_INPUT_NAMES = {"point_coords", "point_labels", "image_height", "image_width", "image_embed", "high_res_feats_0", "high_res_feats_1"};
const std::vector<std::string> ETAM_IMAGE_DECODER_INPUT_NAMES = {"point_coords", "point_labels", "image_height", "image_width", "image_embed"};

MNNForwardType get_mnn_forward_type(const Params& params) {
    switch (params.inference_backend) {
        case InferenceBackend::AUTO:
            return MNN_FORWARD_AUTO;
        case InferenceBackend::CPU:
            return MNN_FORWARD_CPU;
        case InferenceBackend::CUDA:
            return MNN_FORWARD_CUDA;
        case InferenceBackend::OPENCL:
            return MNN_FORWARD_OPENCL;
        case InferenceBackend::METAL:
            return MNN_FORWARD_METAL;
        default:
            return MNN_FORWARD_AUTO;
    }
}

SAM2Base::SAM2Base() : img_encoder_session_(nullptr), mem_encoder_session_(nullptr) {}
SAM2Base::~SAM2Base() {
    if (img_encoder_session_) {
        img_encoder_interpreter_->releaseSession(img_encoder_session_);
    }
    if (mem_encoder_session_) {
        mem_encoder_interpreter_->releaseSession(mem_encoder_session_);
    }
}


Result<std::tuple<VARP, int, int>> SAM2Base::preprocess_image(const std::string& img_path) {
    /*
    std::ifstream file(img_path, std::ios::binary);
    if (!file.is_open()) {
        return Err<VARP>(ErrorCode{-1, "failed to open image file"});
    }
    std::vector<uint8_t> img_buf(std::istreambuf_iterator<char>(file), {}); 
    VARP img = MNN::CV::imdecode(img_buf, MNN::CV::IMREAD_COLOR);
    */

    VARP img = MNN::CV::imread(img_path);
    if (img == nullptr) {
        return Err<std::tuple<VARP, int, int>>(ErrorCode{-1, "failed to read image"});
    }
    int h, w, c;
    MNN::CV::getVARPSize(img, &h, &w, &c);

    VARP resized_img = MNN::CV::resize(img, {IMG_SIZE, IMG_SIZE});
    auto resized_rgb = MNN::CV::cvtColor(resized_img, MNN::CV::COLOR_BGR2RGB);

    return std::make_tuple(resized_rgb, h, w);
}

Result<std::tuple<VARP, int, int>> SAM2Base::preprocess_image(const std::vector<uint8_t>& img_buf) {

    VARP img = MNN::CV::imdecode(img_buf, MNN::CV::IMREAD_COLOR);
    if (img == nullptr) {
        return Err<std::tuple<VARP, int, int>>(ErrorCode{-1, "failed to read image"});
    }
    int h, w, c;
    MNN::CV::getVARPSize(img, &h, &w, &c);

    VARP resized_img = MNN::CV::resize(img, {IMG_SIZE, IMG_SIZE});
    auto resized_rgb = MNN::CV::cvtColor(resized_img, MNN::CV::COLOR_BGR2RGB);

    return std::make_tuple(resized_rgb, h, w);
}


std::tuple<int, std::vector<float>, std::vector<float>> SAM2Base::process_prompts(const std::vector<Point>& points, const std::vector<BBox>& bboxes, int img_height, int img_width) {
    float scaleX = float(IMG_SIZE) / float(img_width);
    float scaleY = float(IMG_SIZE) / float(img_height);

    std::vector<float> inputPointValues, inputLabelValues;

    // Process point prompts
    for (const auto& point : points) {
        inputPointValues.push_back((float)point.x * scaleX);
        inputPointValues.push_back((float)point.y * scaleY);
        inputLabelValues.push_back((float)point.label);
    }

    // Process bounding box prompts
    for (const auto& bbox : bboxes) {
        inputPointValues.push_back((float)bbox.x_min * scaleX);
        inputPointValues.push_back((float)bbox.y_min * scaleY);
        inputPointValues.push_back((float)bbox.x_max * scaleX);
        inputPointValues.push_back((float)bbox.y_max * scaleY);
        inputLabelValues.push_back(2);  // Label for top-left point
        inputLabelValues.push_back(3);  // Label for bottom-right point
    }

    int total_points = inputLabelValues.size();
    return std::make_tuple(total_points, inputPointValues, inputLabelValues);
}


Result<ImageEmbedding> SAM2Base::get_embedding(const std::string& img_path) {
    if (!img_encoder_session_) {
        return Err<ImageEmbedding>(ErrorCode{-1, "image_encoder_session is not initialized"});
    }

    auto prep_out = preprocess_image(img_path);
    if (!prep_out.has_value()) {
        return Err<ImageEmbedding>(ErrorCode{-1, "failed to preprocess image"});
    }
    auto [img, h, w] = prep_out.value();

    return get_embedding(img, h, w);
}

Result<ImageEmbedding> SAM2Base::get_embedding(const std::vector<uint8_t>& img_buf) {
    if (!img_encoder_session_) {
        return Err<ImageEmbedding>(ErrorCode{-1, "image_encoder_session is not initialized"});
    }

    auto prep_out = preprocess_image(img_buf);
    if (!prep_out.has_value()) {
        return Err<ImageEmbedding>(ErrorCode{-1, "failed to preprocess image"});
    }
    auto [img, h, w] = prep_out.value();

    return get_embedding(img, h, w);
}


Result<bool> SAM2Base::_run_image_encoder(const VARP& image_tensor) {
    if (!img_encoder_session_) {
        return Err<bool>(ErrorCode{-1, "image_encoder_session is not initialized"});
    }
    
    auto inputTensor = img_encoder_interpreter_->getSessionInput(img_encoder_session_, "image");
    void* host = inputTensor->map(MNN::Tensor::MAP_TENSOR_WRITE, inputTensor->getDimensionType());
    const uint8_t* img_data = image_tensor->readMap<uint8_t>();
    ::memcpy(host, img_data, inputTensor->size());
    inputTensor->unmap(MNN::Tensor::MAP_TENSOR_WRITE, inputTensor->getDimensionType(), host);

    auto start = std::chrono::high_resolution_clock::now();
    img_encoder_interpreter_->runSession(img_encoder_session_);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    spdlog::debug("image_encoder inference time: {}ms", duration.count());

    return true;
}

Result<std::vector<VARP>> SAM2Base::run_image_encoder(const VARP& image_tensor) {

    auto run_result = _run_image_encoder(image_tensor);
    if (!run_result.has_value()) {
        return Err<std::vector<VARP>>(ErrorCode{-1, "failed to run image_encoder"});
    }

    MNN::Tensor* pix_feat = img_encoder_interpreter_->getSessionOutput(img_encoder_session_, "pix_feat");
    MNN::Tensor* vision_feats = img_encoder_interpreter_->getSessionOutput(img_encoder_session_, "vision_feats");
    MNN::Tensor* vision_pos_embed = img_encoder_interpreter_->getSessionOutput(img_encoder_session_, "vision_pos_embed");

    std::vector<VARP> outputs;
    outputs.push_back(_Input(pix_feat->shape(), NCHW));
    ::memcpy(outputs[0]->writeMap<float>(), pix_feat->host<float>(), pix_feat->size());

    outputs.push_back(_Input(vision_feats->shape(), NCHW));
    ::memcpy(outputs[1]->writeMap<float>(), vision_feats->host<float>(), vision_feats->size());

    outputs.push_back(_Input(vision_pos_embed->shape(), NCHW));
    ::memcpy(outputs[2]->writeMap<float>(), vision_pos_embed->host<float>(), vision_pos_embed->size());

    if (model_type_ == ModelType::SAM) {
        MNN::Tensor* high_res_feat0 = img_encoder_interpreter_->getSessionOutput(img_encoder_session_, "high_res_feat0");
        MNN::Tensor* high_res_feat1 = img_encoder_interpreter_->getSessionOutput(img_encoder_session_, "high_res_feat1");

        outputs.push_back(_Input(high_res_feat0->shape(), NCHW));
        ::memcpy(outputs[3]->writeMap<float>(), high_res_feat0->host<float>(), high_res_feat0->size());

        outputs.push_back(_Input(high_res_feat1->shape(), NCHW));
        ::memcpy(outputs[4]->writeMap<float>(), high_res_feat1->host<float>(), high_res_feat1->size());
    }

    return std::move(outputs);
}

Result<ImageEmbedding> SAM2Base::get_embedding(const VARP& image_tensor, int original_height, int original_width) {

    auto run_result = _run_image_encoder(image_tensor);
    if (!run_result.has_value()) {
        return Err<ImageEmbedding>(ErrorCode{-1, "failed to run image_encoder"});
    }

    MNN::Tensor* pix_feat = img_encoder_interpreter_->getSessionOutput(img_encoder_session_, "pix_feat");


    ImageEmbedding embedding;
    embedding.original_height = original_height;
    embedding.original_width = original_width;
    embedding.model_type = model_type_;

    embedding.image_embed = _Input(pix_feat->shape(), NCHW);
    ::memcpy(embedding.image_embed->writeMap<float>(), pix_feat->host<float>(), pix_feat->size());


    if (embedding.model_type == ModelType::SAM) {
        MNN::Tensor* high_res_feat0 = img_encoder_interpreter_->getSessionOutput(img_encoder_session_, "high_res_feat0");
        MNN::Tensor* high_res_feat1 = img_encoder_interpreter_->getSessionOutput(img_encoder_session_, "high_res_feat1");
        if (!high_res_feat0 || !high_res_feat1) {
            return Err<ImageEmbedding>(ErrorCode{-1, "high_res_feat0 or high_res_feat1 is not found in output"});
        }
        embedding.high_res_feat0 = _Input(high_res_feat0->shape(), NCHW);
        ::memcpy(embedding.high_res_feat0->writeMap<float>(), high_res_feat0->host<float>(), high_res_feat0->size());

        embedding.high_res_feat1 = _Input(high_res_feat1->shape(), NCHW);
        ::memcpy(embedding.high_res_feat1->writeMap<float>(), high_res_feat1->host<float>(), high_res_feat1->size());

    } else {
        // spdlog::info("no high_res_feat0 and high_res_feat1");
    }

    return embedding;
}


Result<std::vector<VARP>> SAM2Base::predict(const ImageEmbedding& embedding, const std::vector<Point>& points, const std::vector<BBox>& bboxes) {
    if (!img_decoder_module_) {
        return Err<std::vector<VARP>>(ErrorCode{-1, "image_decoder_module is not initialized"});
    }
    if (embedding.model_type != model_type_) {
        return Err<std::vector<VARP>>(ErrorCode{-1, "model_type mismatch"});
    }
    if (embedding.image_embed == nullptr) {
        return Err<std::vector<VARP>>(ErrorCode{-1, "Invalid embedding data"});
    }
    if (embedding.model_type == ModelType::SAM && (embedding.high_res_feat0 == nullptr || embedding.high_res_feat1 == nullptr)) {
        return Err<std::vector<VARP>>(ErrorCode{-1, "Invalid embedding data"});
    }

    auto [point_num, point_values, label_values] = process_prompts(points, bboxes, embedding.original_height, embedding.original_width);

    std::vector<VARP> inputs;
    inputs.push_back(_Input({1, point_num, 2}, NCHW));
    inputs.push_back(_Input({1, point_num}, NCHW));
    inputs.push_back(_Scalar(int32_t(embedding.original_height)));
    inputs.push_back(_Scalar(int32_t(embedding.original_width)));
    inputs.push_back(embedding.image_embed);
    if (embedding.model_type == ModelType::SAM) {
        inputs.push_back(embedding.high_res_feat0);
        inputs.push_back(embedding.high_res_feat1);
    } else {
        // spdlog::info("no high_res_feat0 and high_res_feat1");
    }


    auto point_coords = inputs[0]->writeMap<float>();
    ::memcpy(point_coords, point_values.data(), point_values.size() * sizeof(float));
    
    auto point_labels = inputs[1]->writeMap<float>();
    ::memcpy(point_labels, label_values.data(), label_values.size() * sizeof(float));


    auto start = std::chrono::high_resolution_clock::now();
    auto outs = img_decoder_module_->onForward(inputs);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    spdlog::debug("image_decoder inference time: {}ms", duration.count());

    return std::move(outs);
}


Result<bool> SAM2Base::init_img_encoder_decoder(const std::string& model_path, const Params& params) {
    // Check if model_path is a tar file
    if (model_path.substr(model_path.length() - 4) == ".tar") {
        std::ifstream file(model_path, std::ios::binary);
        if (!file.is_open()) {
            spdlog::error("Failed to open tar file: {}", model_path);
            return Err<bool>(ErrorCode{-1, "failed to open tar file"});
        }
        
        std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        auto entries = parseTar(buffer.data(), buffer.size());
        
        if (entries.empty()) {
            spdlog::error("Failed to parse tar file: {}", model_path);
            return Err<bool>(ErrorCode{-1, "failed to parse tar file"});
        }
        
        return init_img_encoder_decoder(entries, params);
    }

    // Initialize image encoder
    std::string encoder_path = (std::filesystem::path(model_path) / "image_encoder.mnn").string();
    img_encoder_interpreter_ = std::unique_ptr<MNN::Interpreter>(
        MNN::Interpreter::createFromFile(encoder_path.c_str())
    );
    
    if (!img_encoder_interpreter_) {
        spdlog::error("Failed to create interpreter from model path: {}", encoder_path);
        return Err<bool>(ErrorCode{-1, "failed to create image_encoder_interpreter from model path"});
    }

    MNN::ScheduleConfig config;
    config.type = get_mnn_forward_type(params);
    //config.numThread = params.num_threads;

    img_encoder_session_ = img_encoder_interpreter_->createSession(config);
    
    if (!img_encoder_session_) {
        spdlog::error("Failed to create image encoder session from model path: {}", encoder_path);
        return Err<bool>(ErrorCode{-1, "failed to create image encoder session from model path"});
    }

    auto encoder_outputs = img_encoder_interpreter_->getSessionOutputAll(img_encoder_session_);
    if (encoder_outputs.size() == 5) {
        model_type_ = ModelType::SAM;
        spdlog::info("model type: SAM");
    } else if (encoder_outputs.size() == 3) {
        model_type_ = ModelType::ETAM;
        spdlog::info("model type: ETAM");
    } else {
        return Err<bool>(ErrorCode{-1, "failed to determine model type"});
    }

    // Initialize image decoder
    std::string decoder_path = (std::filesystem::path(model_path) / "image_decoder.mnn").string();
    if (model_type_ == ModelType::SAM) {
        img_decoder_module_ = std::unique_ptr<MNN::Express::Module>(
        MNN::Express::Module::load(SAM_IMAGE_DECODER_INPUT_NAMES,
                                 {"obj_ptr","mask_for_mem","pred_mask","ious", "object_score_logits"},
                                 decoder_path.c_str())
        );
    } else if (model_type_ == ModelType::ETAM) {
        img_decoder_module_ = std::unique_ptr<MNN::Express::Module>(
            MNN::Express::Module::load(ETAM_IMAGE_DECODER_INPUT_NAMES,
                                 {"obj_ptr","mask_for_mem","pred_mask","ious", "object_score_logits"},
                                 decoder_path.c_str())
        );
    }
    
    if (!img_decoder_module_) {
        spdlog::error("Failed to load image decoder model from path: {}", decoder_path);
        return Err<bool>(ErrorCode{-1, "failed to load image decoder model from path"});
    }
    
    return true;
}

Result<bool> SAM2Base::init_img_encoder_decoder(const std::vector<TarEntry>& entries, const Params& params) {
    const uint8_t* encoder_data = nullptr;
    size_t encoder_size = 0;
    const uint8_t* decoder_data = nullptr;
    size_t decoder_size = 0;
    
    for (const auto& entry : entries) {
        if (entry.name == "image_encoder.mnn") {
            encoder_data = entry.data;
            encoder_size = entry.size;
        } else if (entry.name == "image_decoder.mnn") {
            decoder_data = entry.data;
            decoder_size = entry.size;
        }
    }
    
    if (!encoder_data || !decoder_data) {
        spdlog::error("Failed to find required model files in tar entries");
        return Err<bool>(ErrorCode{-1, "failed to find required model files in tar entries"});
    }

    // Initialize image encoder
    img_encoder_interpreter_ = std::unique_ptr<MNN::Interpreter>(
        MNN::Interpreter::createFromBuffer(encoder_data, encoder_size)
    );
    
    if (!img_encoder_interpreter_) {
        spdlog::error("Failed to create interpreter from encoder data");
        return Err<bool>(ErrorCode{-1, "failed to create image_encoder_interpreter from data"});
    }

    MNN::ScheduleConfig config;
    config.type = get_mnn_forward_type(params);
    //config.numThread = params.num_threads;

    img_encoder_session_ = img_encoder_interpreter_->createSession(config);
    
    if (!img_encoder_session_) {
        spdlog::error("Failed to create image encoder session");
        return Err<bool>(ErrorCode{-1, "failed to create image encoder session"});
    }

    auto encoder_outputs = img_encoder_interpreter_->getSessionOutputAll(img_encoder_session_);
    if (encoder_outputs.size() == 5) {
        model_type_ = ModelType::SAM;
        spdlog::info("model type: SAM");
    } else if (encoder_outputs.size() == 3) {
        model_type_ = ModelType::ETAM;
        spdlog::info("model type: ETAM");
    } else {
        return Err<bool>(ErrorCode{-1, "failed to determine model type"});
    }
    

    // Initialize image decoder
    if (model_type_ == ModelType::SAM) {
        img_decoder_module_ = std::unique_ptr<MNN::Express::Module>(
            MNN::Express::Module::load(SAM_IMAGE_DECODER_INPUT_NAMES,
                                 {"obj_ptr","mask_for_mem","pred_mask","ious", "object_score_logits"},
                                 decoder_data, decoder_size)
        );
    } else if (model_type_ == ModelType::ETAM) {
        img_decoder_module_ = std::unique_ptr<MNN::Express::Module>(
            MNN::Express::Module::load(ETAM_IMAGE_DECODER_INPUT_NAMES,
                                 {"obj_ptr","mask_for_mem","pred_mask","ious", "object_score_logits"},
                                 decoder_data, decoder_size)
        );
    }
    
    if (!img_decoder_module_) {
        spdlog::error("Failed to load image decoder model");
        return Err<bool>(ErrorCode{-1, "failed to load image decoder model"});
    }
    
    return true;
}

Result<bool> SAM2Base::init_mem_encoder_atten(const std::string& model_path, const Params& params) {
    // Check if model_path is a tar file
    if (model_path.substr(model_path.length() - 4) == ".tar") {
        std::ifstream file(model_path, std::ios::binary);
        if (!file.is_open()) {
            spdlog::error("Failed to open tar file: {}", model_path);
            return Err<bool>(ErrorCode{-1, "failed to open tar file"});
        }
        
        std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        auto entries = parseTar(buffer.data(), buffer.size());
        
        if (entries.empty()) {
            spdlog::error("Failed to parse tar file: {}", model_path);
            return Err<bool>(ErrorCode{-1, "failed to parse tar file"});
        }
        
        return init_mem_encoder_atten(entries, params);
    }

    // Initialize memory encoder
    std::string mem_encoder_path = (std::filesystem::path(model_path) / "memory_encoder.mnn").string();
    mem_encoder_interpreter_ = std::unique_ptr<MNN::Interpreter>(
        MNN::Interpreter::createFromFile(mem_encoder_path.c_str())
    );
    
    if (!mem_encoder_interpreter_) {
        spdlog::error("Failed to create interpreter from model path: {}", mem_encoder_path);
        return Err<bool>(ErrorCode{-1, "failed to create mem_encoder_interpreter from model path"});
    }

    MNN::ScheduleConfig config;
    config.type = get_mnn_forward_type(params);
    //config.numThread = params.num_threads;
    
    mem_encoder_session_ = mem_encoder_interpreter_->createSession(config);
    
    if (!mem_encoder_session_) {
        spdlog::error("Failed to create memory encoder session from model path: {}", mem_encoder_path);
        return Err<bool>(ErrorCode{-1, "failed to create memory encoder session from model path"});
    }

    // Initialize memory attention module
    std::string mem_attention_path = (std::filesystem::path(model_path) / "memory_attention.mnn").string();
    mem_attn_module_ = std::unique_ptr<MNN::Express::Module>(
        MNN::Express::Module::load({"current_vision_feat","current_vision_pos_embed","maskmem_feats","maskmem_feat_pos_embeds","maskmem_feat_pos_idx","obj_ptrs","obj_pos_list"},
                                 {"image_embed"},
                                 mem_attention_path.c_str())
    );
    
    if (!mem_attn_module_) {
        spdlog::error("Failed to load memory attention model from path: {}", mem_attention_path);
        return Err<bool>(ErrorCode{-1, "failed to load memory attention model from path"});
    }
    
    return true;
}

Result<bool> SAM2Base::init_mem_encoder_atten(const std::vector<TarEntry>& entries, const Params& params) {
    const uint8_t* mem_encoder_data = nullptr;
    size_t mem_encoder_size = 0;
    const uint8_t* mem_attention_data = nullptr;
    size_t mem_attention_size = 0;
    
    for (const auto& entry : entries) {
        if (entry.name == "memory_encoder.mnn") {
            mem_encoder_data = entry.data;
            mem_encoder_size = entry.size;
        } else if (entry.name == "memory_attention.mnn") {
            mem_attention_data = entry.data;
            mem_attention_size = entry.size;
        }
    }
    
    if (!mem_encoder_data || !mem_attention_data) {
        spdlog::error("Failed to find required model files in tar entries");
        return Err<bool>(ErrorCode{-1, "failed to find required model files in tar entries"});
    }

    // Initialize memory encoder
    mem_encoder_interpreter_ = std::unique_ptr<MNN::Interpreter>(
        MNN::Interpreter::createFromBuffer(mem_encoder_data, mem_encoder_size)
    );
    
    if (!mem_encoder_interpreter_) {
        spdlog::error("Failed to create interpreter from memory encoder data");
        return Err<bool>(ErrorCode{-1, "failed to create mem_encoder_interpreter from data"});
    }

    MNN::ScheduleConfig config;
    config.type = get_mnn_forward_type(params);
    //config.numThread = params.num_threads;
    
    mem_encoder_session_ = mem_encoder_interpreter_->createSession(config);
    
    if (!mem_encoder_session_) {
        spdlog::error("Failed to create memory encoder session");
        return Err<bool>(ErrorCode{-1, "failed to create memory encoder session"});
    }

    // Initialize memory attention module
    mem_attn_module_ = std::unique_ptr<MNN::Express::Module>(
        MNN::Express::Module::load({"current_vision_feat","current_vision_pos_embed","maskmem_feats","maskmem_feat_pos_embeds","maskmem_feat_pos_idx","obj_ptrs","obj_pos_list"},
                                 {"image_embed"},
                                 mem_attention_data, mem_attention_size)
    );
    
    if (!mem_attn_module_) {
        spdlog::error("Failed to load memory attention model");
        return Err<bool>(ErrorCode{-1, "failed to load memory attention model"});
    }
    
    return true;
}

Result<std::vector<VARP>> SAM2Base::run_image_decoder(const std::vector<VARP>& inputs) {
    if (!img_decoder_module_) {
        spdlog::error("Image decoder module is not initialized");
        return Err<std::vector<VARP>>(ErrorCode{-1, "Image decoder module is not initialized"});
    }
    return img_decoder_module_->onForward(inputs);
}

Result<std::vector<VARP>> SAM2Base::run_memory_encoder(const std::vector<VARP>& inputs) {
    if (!mem_encoder_session_) {
        spdlog::error("Memory encoder session is not initialized");
        return Err<std::vector<VARP>>(ErrorCode{-1, "Memory encoder session is not initialized"});
    }

    // Get input tensors
    auto input_tensor = mem_encoder_interpreter_->getSessionInput(mem_encoder_session_, "mask_for_mem");
    auto pix_feat_tensor = mem_encoder_interpreter_->getSessionInput(mem_encoder_session_, "pix_feat");

    // Convert VARP to Tensor
    void* host = input_tensor->map(MNN::Tensor::MAP_TENSOR_WRITE, input_tensor->getDimensionType());
    const float* input_data = inputs[0]->readMap<float>();
    ::memcpy(host, input_data, input_tensor->size());
    input_tensor->unmap(MNN::Tensor::MAP_TENSOR_WRITE, input_tensor->getDimensionType(), host);

    void* pix_feat_host = pix_feat_tensor->map(MNN::Tensor::MAP_TENSOR_WRITE, pix_feat_tensor->getDimensionType());
    const float* pix_feat_data = inputs[1]->readMap<float>();
    ::memcpy(pix_feat_host, pix_feat_data, pix_feat_tensor->size());
    pix_feat_tensor->unmap(MNN::Tensor::MAP_TENSOR_WRITE, pix_feat_tensor->getDimensionType(), pix_feat_host);

    //auto obj_score_logits_tensor = mem_encoder_interpreter_->getSessionInput(mem_encoder_session_, "object_score_logits");
    //if (obj_score_logits_tensor) {
    if (model_type_ == ModelType::SAM) {
        auto obj_score_logits_tensor = mem_encoder_interpreter_->getSessionInput(mem_encoder_session_, "object_score_logits");
        void* obj_score_logits_host = obj_score_logits_tensor->map(MNN::Tensor::MAP_TENSOR_WRITE, obj_score_logits_tensor->getDimensionType());
        const float* obj_score_logits_data = inputs[2]->readMap<float>();
        ::memcpy(obj_score_logits_host, obj_score_logits_data, obj_score_logits_tensor->size());
        obj_score_logits_tensor->unmap(MNN::Tensor::MAP_TENSOR_WRITE, obj_score_logits_tensor->getDimensionType(), obj_score_logits_host);
    } else {
        //spdlog::info("no object_score_logits");
    }

    // Run session
    mem_encoder_interpreter_->runSession(mem_encoder_session_);

    // Get output tensors
    auto maskmem_features = mem_encoder_interpreter_->getSessionOutput(mem_encoder_session_, "maskmem_features");
    auto maskmem_pos_enc = mem_encoder_interpreter_->getSessionOutput(mem_encoder_session_, "maskmem_pos_enc");

    // Convert Tensor to VARP
    std::vector<VARP> outputs;
    outputs.push_back(_Input(maskmem_features->shape(), NCHW));
    outputs.push_back(_Input(maskmem_pos_enc->shape(), NCHW));

    void* maskmem_features_host = maskmem_features->map(MNN::Tensor::MAP_TENSOR_READ, maskmem_features->getDimensionType());
    ::memcpy(outputs[0]->writeMap<float>(), maskmem_features_host, maskmem_features->size());
    maskmem_features->unmap(MNN::Tensor::MAP_TENSOR_READ, maskmem_features->getDimensionType(), maskmem_features_host);

    void* maskmem_pos_enc_host = maskmem_pos_enc->map(MNN::Tensor::MAP_TENSOR_READ, maskmem_pos_enc->getDimensionType());
    ::memcpy(outputs[1]->writeMap<float>(), maskmem_pos_enc_host, maskmem_pos_enc->size());
    maskmem_pos_enc->unmap(MNN::Tensor::MAP_TENSOR_READ, maskmem_pos_enc->getDimensionType(), maskmem_pos_enc_host);

    return outputs;
}

Result<std::vector<VARP>> SAM2Base::run_memory_attention(const std::vector<VARP>& inputs) {
    if (!mem_attn_module_) {
        spdlog::error("Memory attention module is not initialized");
        return Err<std::vector<VARP>>(ErrorCode{-1, "Memory attention module is not initialized"});
    }
    return mem_attn_module_->onForward(inputs);
} 

Mask convert_to_mask(const VARP& pred_mask, float score, float threshold) {
    int h, w, c;
    MNN::CV::getVARPSize(pred_mask, &h, &w, &c);

    Mask mask;
    mask.height = h;
    mask.width = w;

    mask.data.resize(h * w);
    auto pred_mask_data = pred_mask->readMap<float>();
    for (int i = 0; i < mask.data.size(); i++) {
        mask.data[i] = pred_mask_data[i] > threshold ? 255 : 0;
    }

    mask.score = score;

    return mask;
}   

} // namespace sam2