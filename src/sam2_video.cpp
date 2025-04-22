#include "sam2_video.h"

#include <deque>
#include <unordered_map>
#include <sstream>
#include <chrono>

#include "spdlog/spdlog.h"

#include "sam2_base.h"

namespace sam2 {

const int IMG_SIZE = 1024;
const int NUM_MASKMEM = 7;

// Internal structures
struct FrameOutput {
    VARP maskmem_features;
    VARP maskmem_pos_enc;
    VARP obj_ptr;
    int frame_idx;
    VARP pred_mask;
    VARP iou;
};

struct NonCondFrameObj {
    VARP obj_ptr;
    int frame_idx;
};

struct NonCondFrameMaskmem {
    VARP maskmem_features;
    VARP maskmem_pos_enc;
    int frame_idx;
};

struct ObjectState {
    int id;
    std::vector<Point> points;
    std::vector<BBox> boxes;
    std::vector<FrameOutput> cond_frame_outputs;
    std::deque<NonCondFrameObj> non_cond_frame_objs;
    std::deque<NonCondFrameMaskmem> non_cond_frame_maskmems;

    ObjectState(int id) : id(id) {
        // Initialize empty queues
        non_cond_frame_objs.clear();
        non_cond_frame_maskmems.clear();
    }

    void add_non_cond_frame_obj(const NonCondFrameObj& obj) {
        if (non_cond_frame_objs.size() >= 16) {
            non_cond_frame_objs.pop_front();
        }
        non_cond_frame_objs.push_back(obj);
    }

    void add_non_cond_frame_maskmem(const NonCondFrameMaskmem& maskmem) {
        if (non_cond_frame_maskmems.size() >= NUM_MASKMEM-1) {
            non_cond_frame_maskmems.pop_front();
        }
        non_cond_frame_maskmems.push_back(maskmem);
    }
};

// InferenceState implementation
class InferenceState::Impl {
public:
    std::unordered_map<int, std::shared_ptr<ObjectState>> obj_states;
    std::pair<int, int> video_size;
    int next_frame_idx;
    std::unordered_map<int, std::vector<VARP>> cond_frame_embeddings;

    void reset() {
        obj_states.clear();
        video_size = {0, 0};
        next_frame_idx = 0;
        cond_frame_embeddings.clear();
    }

    void remove_object(int obj_id) {
        obj_states.erase(obj_id);
        cond_frame_embeddings.erase(obj_id);
    }

    std::shared_ptr<ObjectState> get_object_state(int obj_id) {
        auto it = obj_states.find(obj_id);
        if (it == obj_states.end()) {
            auto state = std::make_shared<ObjectState>(obj_id);
            obj_states[obj_id] = state;
            return state;
        }
        return it->second;
    }
};

InferenceState::InferenceState() : m_pimpl(std::make_unique<Impl>()) {}
InferenceState::~InferenceState() = default;

void InferenceState::reset() { m_pimpl->reset(); }
void InferenceState::remove_object(int obj_id) { m_pimpl->remove_object(obj_id); }

// SAM2Video implementation
class SAM2Video::Impl {
public:
    std::unique_ptr<SAM2Base> sam2_base;

    Impl() = default;
    ~Impl() = default;

    Result<bool> initialize(const std::string& model_path, const Params& params) {
        sam2_base = std::make_unique<SAM2Base>();
        auto result = sam2_base->init_img_encoder_decoder(model_path, params);
        if (!result.has_value()) {
            return Err<bool>(result.error());
        }
        result = sam2_base->init_mem_encoder_atten(model_path, params);
        if (!result.has_value()) {
            return Err<bool>(result.error());
        }
        return true;
    }

    Result<bool> initialize(uint8_t* buffer, size_t size, const Params& params) {
        sam2_base = std::make_unique<SAM2Base>();
        auto entries = parseTar(buffer, size);
        auto result = sam2_base->init_img_encoder_decoder(entries, params);
        if (!result.has_value()) {
            return Err<bool>(result.error());
        }
        result = sam2_base->init_mem_encoder_atten(entries, params);
        if (!result.has_value()) {
            return Err<bool>(result.error());
        }
        return true;
    }

    Result<std::tuple<VARP, int, int>> prepare_image(const std::string& img_path, InferenceState* inference_state = nullptr) {
        auto prep_out = sam2_base->preprocess_image(img_path);
        if (!prep_out.has_value()) {
            return Err<std::tuple<VARP, int, int>>(
                ErrorCode{-1, "Failed to preprocess image"}
            );
        }
        auto [img_tensor, h, w] = prep_out.value();
        
        // If inference_state is provided, check and update video size
        if (inference_state) {
            if (inference_state->m_pimpl->video_size.first == 0 && inference_state->m_pimpl->video_size.second == 0) {
                inference_state->m_pimpl->video_size = {h, w};
            } else if (inference_state->m_pimpl->video_size != std::make_pair(h, w)) {
                return Err<std::tuple<VARP, int, int>>(
                    ErrorCode{-1, "Image size does not match video size"}
                );
            }
        }

        return prep_out.value();
    }


    Result<VARP> prepare_memory_attention(
        int frame_idx,
        const std::vector<VARP>& img_encoder_out,
        const std::shared_ptr<ObjectState>& object_state
    ) {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto vision_feats = img_encoder_out[1];
        auto vision_feats_pos = img_encoder_out[2];

        auto [to_cat_maskmem_feats, to_cat_maskmem_feat_pos_embeds, maskmem_feat_frame_idx, maskmem_feat_pos_embed_idx] = 
            prepare_maskmem(frame_idx, object_state->cond_frame_outputs, object_state->non_cond_frame_maskmems);

        auto [to_cat_obj_ptrs, obj_frame_idx, obj_pos_list] = 
            prepare_obj_ptrs(frame_idx, object_state->cond_frame_outputs, object_state->non_cond_frame_objs);

        // Stack the tensors
        auto maskmem_feats = _Stack(to_cat_maskmem_feats);
        auto maskmem_feat_pos_embeds = _Stack(to_cat_maskmem_feat_pos_embeds);
        auto maskmem_feat_pos_idx = _Input({static_cast<int32_t>(maskmem_feat_pos_embed_idx.size())}, NCHW, halide_type_of<int32_t>());
        ::memcpy(maskmem_feat_pos_idx->writeMap<int32_t>(), maskmem_feat_pos_embed_idx.data(), 
                 maskmem_feat_pos_embed_idx.size() * sizeof(int32_t));

        auto obj_ptrs = _Stack(to_cat_obj_ptrs);
        auto obj_pos_list_tensor = _Input({static_cast<int32_t>(obj_pos_list.size())}, NCHW, halide_type_of<int32_t>());
        ::memcpy(obj_pos_list_tensor->writeMap<int32_t>(), obj_pos_list.data(), 
                 obj_pos_list.size() * sizeof(int32_t));

        // Prepare inputs for memory attention
        std::vector<VARP> inputs = {
            vision_feats,
            vision_feats_pos,
            maskmem_feats,
            maskmem_feat_pos_embeds,
            maskmem_feat_pos_idx,
            obj_ptrs,
            obj_pos_list_tensor
        };

        // Run memory attention
        auto memory_attention_start = std::chrono::high_resolution_clock::now();
        auto memory_attention_result = sam2_base->run_memory_attention(inputs);
        auto memory_attention_end = std::chrono::high_resolution_clock::now();
        auto memory_attention_time = std::chrono::duration_cast<std::chrono::milliseconds>(memory_attention_end - memory_attention_start).count();
        spdlog::debug("memory_attention inference time: {} ms", memory_attention_time);

        if (!memory_attention_result.has_value()) {
            return Err<VARP>(memory_attention_result.error());
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        spdlog::debug("prepare_memory_attention total time: {} ms", total_time);

        return memory_attention_result.value()[0];
    }

    std::tuple<std::vector<VARP>, std::vector<VARP>, std::vector<int>, std::vector<int>> 
    prepare_maskmem(
        int frame_idx,
        const std::vector<FrameOutput>& cond_frame_outputs,
        const std::deque<NonCondFrameMaskmem>& non_cond_frame_maskmems
    ) {
        std::vector<VARP> to_cat_maskmem_feats;
        std::vector<VARP> to_cat_maskmem_feat_pos_embeds;
        std::vector<int> maskmem_feat_frame_idx;
        std::vector<int> maskmem_feat_pos_embed_idx;

        for (const auto& cond_frame_output : cond_frame_outputs) {
            to_cat_maskmem_feats.push_back(cond_frame_output.maskmem_features);
            to_cat_maskmem_feat_pos_embeds.push_back(cond_frame_output.maskmem_pos_enc);
            maskmem_feat_frame_idx.push_back(cond_frame_output.frame_idx);
            maskmem_feat_pos_embed_idx.push_back(NUM_MASKMEM-1);
        }

        for (const auto& non_cond_frame_maskmem : non_cond_frame_maskmems) {
            to_cat_maskmem_feats.push_back(non_cond_frame_maskmem.maskmem_features);
            to_cat_maskmem_feat_pos_embeds.push_back(non_cond_frame_maskmem.maskmem_pos_enc);
            maskmem_feat_frame_idx.push_back(non_cond_frame_maskmem.frame_idx);
            maskmem_feat_pos_embed_idx.push_back(frame_idx-non_cond_frame_maskmem.frame_idx-1);
        }

        return std::make_tuple(to_cat_maskmem_feats, to_cat_maskmem_feat_pos_embeds, 
                              maskmem_feat_frame_idx, maskmem_feat_pos_embed_idx);
    }

    std::tuple<std::vector<VARP>, std::vector<int>, std::vector<int>> 
    prepare_obj_ptrs(
        int frame_idx,
        const std::vector<FrameOutput>& cond_frame_outputs,
        const std::deque<NonCondFrameObj>& non_cond_frame_objs
    ) {
        std::vector<VARP> to_cat_obj_ptrs;
        std::vector<int> obj_frame_idx;
        std::vector<int> obj_pos_list;

        for (const auto& cond_frame_output : cond_frame_outputs) {
            to_cat_obj_ptrs.push_back(cond_frame_output.obj_ptr);
            obj_frame_idx.push_back(cond_frame_output.frame_idx);
            obj_pos_list.push_back(frame_idx-cond_frame_output.frame_idx);
        }

        for (auto it = non_cond_frame_objs.rbegin(); it != non_cond_frame_objs.rend(); ++it) {
            to_cat_obj_ptrs.push_back(it->obj_ptr);
            obj_frame_idx.push_back(it->frame_idx);
            obj_pos_list.push_back(frame_idx-it->frame_idx);
        }

        return std::make_tuple(to_cat_obj_ptrs, obj_frame_idx, obj_pos_list);
    }

    Result<std::tuple<int, Mask>> add_new_points_or_box(
        InferenceState& inference_state,
        const std::string& image_path,
        int obj_id,
        const std::vector<Point>& points = {},
        const std::vector<BBox>& boxes = {},
        bool clear_old_points = true
    ) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto prep_result = prepare_image(image_path, &inference_state);
        if (!prep_result.has_value()) {
            return Err<std::tuple<int, Mask>>(prep_result.error());
        }
        auto [img_tensor, h, w] = prep_result.value();

        auto object_state = inference_state.m_pimpl->get_object_state(obj_id);
        int frame_idx = inference_state.m_pimpl->next_frame_idx;


        
        std::vector<Point> all_points;
        std::vector<BBox> all_boxes;

        if (clear_old_points) {
            all_points = points;
            all_boxes = boxes;
        } else {
            all_points.insert(all_points.end(), object_state->points.begin(), object_state->points.end());
            all_boxes.insert(all_boxes.end(), object_state->boxes.begin(), object_state->boxes.end());
            all_points.insert(all_points.end(), points.begin(), points.end());
            all_boxes.insert(all_boxes.end(), boxes.begin(), boxes.end());
        }

        auto [point_num, coord_points, coord_labels] = sam2_base->process_prompts(all_points, all_boxes, h, w);

        // Get or compute image encoder output
        std::vector<VARP> img_encoder_out;
        if (inference_state.m_pimpl->cond_frame_embeddings.find(frame_idx) != inference_state.m_pimpl->cond_frame_embeddings.end()) {
            img_encoder_out = inference_state.m_pimpl->cond_frame_embeddings[frame_idx];
        } else {
            auto image_encoder_start = std::chrono::high_resolution_clock::now();
            auto embedding_result = sam2_base->run_image_encoder(img_tensor);
            auto image_encoder_end = std::chrono::high_resolution_clock::now();
            auto image_encoder_time = std::chrono::duration_cast<std::chrono::milliseconds>(image_encoder_end - image_encoder_start).count();
            spdlog::debug("image_encoder inference time: {} ms", image_encoder_time);

            if (!embedding_result.has_value()) {
                return Err<std::tuple<int, Mask>>(
                    ErrorCode{-1, "Failed to get image embedding"}
                );
            }
            img_encoder_out = embedding_result.value();
            inference_state.m_pimpl->cond_frame_embeddings[frame_idx] = img_encoder_out;
        }

        // Prepare decoder inputs
        std::vector<VARP> dec_inputs = {
            _Input({1, point_num, 2}, NCHW),
            _Input({1, point_num}, NCHW),
            _Scalar(int32_t(h)),
            _Scalar(int32_t(w)),
            img_encoder_out[1]
        };

        auto point_coords = dec_inputs[0]->writeMap<float>();
        ::memcpy(point_coords, coord_points.data(), coord_points.size() * sizeof(float));
        auto point_labels = dec_inputs[1]->writeMap<float>();
        ::memcpy(point_labels, coord_labels.data(), coord_labels.size() * sizeof(float));

        if (img_encoder_out.size() == 5) { // not etam model, has high_res_feat0 and high_res_feat1
            dec_inputs.push_back(img_encoder_out[3]);
            dec_inputs.push_back(img_encoder_out[4]);
        }

        // Run decoder
        auto image_decoder_start = std::chrono::high_resolution_clock::now();
        auto img_decoder_result = sam2_base->run_image_decoder(dec_inputs);
        auto image_decoder_end = std::chrono::high_resolution_clock::now();
        auto image_decoder_time = std::chrono::duration_cast<std::chrono::milliseconds>(image_decoder_end - image_decoder_start).count();
        spdlog::debug("image_decoder inference time: {} ms", image_decoder_time);

        if (!img_decoder_result.has_value()) {
            return Err<std::tuple<int, Mask>>(img_decoder_result.error());
        }
        auto img_decoder_out = img_decoder_result.value();


        // Prepare memory encoder inputs
        std::vector<VARP> mec_inputs = {
            img_decoder_out[1],  // mask_for_mem
            img_encoder_out[0]   // pix_feat
        };

        if (img_decoder_out.size() > 4) {
            mec_inputs.push_back(img_decoder_out[4]);  // object_score_logits
        } else {
            //spdlog::info("no object_score_logits");
        }

        // Run memory encoder
        auto memory_encoder_start = std::chrono::high_resolution_clock::now();
        auto memory_encoder_result = sam2_base->run_memory_encoder(mec_inputs);
        auto memory_encoder_end = std::chrono::high_resolution_clock::now();
        auto memory_encoder_time = std::chrono::duration_cast<std::chrono::milliseconds>(memory_encoder_end - memory_encoder_start).count();
        spdlog::debug("memory_encoder inference time: {} ms", memory_encoder_time);

        if (!memory_encoder_result.has_value()) {
            return Err<std::tuple<int, Mask>>(memory_encoder_result.error());
        }
        auto memory_encoder_out = memory_encoder_result.value();

        // Update object state
        object_state->points = all_points;
        object_state->boxes = all_boxes;
        
        FrameOutput cond_frame_output;
        cond_frame_output.maskmem_features = memory_encoder_out[0];
        cond_frame_output.maskmem_pos_enc = memory_encoder_out[1];
        cond_frame_output.obj_ptr = img_decoder_out[0];
        cond_frame_output.frame_idx = frame_idx;
        cond_frame_output.pred_mask = img_decoder_out[2];
        cond_frame_output.iou = img_decoder_out[3];
        object_state->cond_frame_outputs.push_back(cond_frame_output);

        // Convert VARP to Mask
        float score = img_decoder_out[3]->readMap<float>()[0];
        Mask mask = convert_to_mask(img_decoder_out[2], score);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        spdlog::debug("add_new_points_or_box total time: {} ms", total_time);

        return std::make_tuple(frame_idx, mask);
    }

    Result<std::tuple<int, std::unordered_map<int, Mask>>> track_step(
        InferenceState& inference_state,
        const std::string& image_path
    ) {
        auto start_time = std::chrono::high_resolution_clock::now();

        int frame_idx = inference_state.m_pimpl->next_frame_idx;
        spdlog::debug("track_step frame_idx: {}", frame_idx);

        // Get or compute image encoder output
        std::vector<VARP> img_encoder_out;
        if (inference_state.m_pimpl->cond_frame_embeddings.find(frame_idx) != inference_state.m_pimpl->cond_frame_embeddings.end()) {
            img_encoder_out = inference_state.m_pimpl->cond_frame_embeddings[frame_idx];
        } else {
            auto prep_result = prepare_image(image_path, &inference_state);
            if (!prep_result.has_value()) {
                return Err<std::tuple<int, std::unordered_map<int, Mask>>>(prep_result.error());
            }
            auto [img_tensor, h, w] = prep_result.value();

            auto image_encoder_start = std::chrono::high_resolution_clock::now();
            auto embedding_result = sam2_base->run_image_encoder(img_tensor);
            auto image_encoder_end = std::chrono::high_resolution_clock::now();
            auto image_encoder_time = std::chrono::duration_cast<std::chrono::milliseconds>(image_encoder_end - image_encoder_start).count();
            spdlog::debug("image_encoder inference time: {} ms", image_encoder_time);

            if (!embedding_result.has_value()) {
                return Err<std::tuple<int, std::unordered_map<int, Mask>>>(
                    ErrorCode{-1, "Failed to get image embedding"}
                );
            }
            img_encoder_out = embedding_result.value();
            inference_state.m_pimpl->cond_frame_embeddings[frame_idx] = img_encoder_out;
        }

        std::unordered_map<int, Mask> outputs;

        for (const auto& [obj_id, object_state] : inference_state.m_pimpl->obj_states) {
            // Check if this is a conditional frame
            bool is_cond_frame = false;
            for (const auto& cond_out : object_state->cond_frame_outputs) {
                if (cond_out.frame_idx == frame_idx) {
                    // Convert VARP to Mask
                    float score = cond_out.iou->readMap<float>()[0];
                    auto mask = convert_to_mask(cond_out.pred_mask, score);

                    outputs[obj_id] = mask;
                    is_cond_frame = true;
                    break;
                }
            }

            if (!is_cond_frame) {
                // Use memory attention for non-conditional frames
                auto pix_feat_with_mem = prepare_memory_attention(frame_idx, img_encoder_out, object_state);
                if (!pix_feat_with_mem.has_value()) {
                    return Err<std::tuple<int, std::unordered_map<int, Mask>>>(pix_feat_with_mem.error());
                }

                auto [h, w] = inference_state.m_pimpl->video_size;
                // Prepare decoder inputs for tracking
                std::vector<VARP> dec_inputs = {
                    _Input({1, 1, 2}, NCHW),  // Empty point coords
                    _Input({1, 1}, NCHW),     // Empty point labels
                    _Scalar(h),
                    _Scalar(w),
                    pix_feat_with_mem.value()
                };

                // Initialize empty point coordinates and labels (similar to Python implementation)
                auto point_coords = dec_inputs[0]->writeMap<float>();
                std::fill(point_coords, point_coords + 2, 0.0f);  // Initialize to zeros
                
                auto point_labels = dec_inputs[1]->writeMap<float>();
                std::fill(point_labels, point_labels + 1, -1.0f);  // Initialize to -1

                if (img_encoder_out.size() == 5) { // not etam model, has high_res_feat0 and high_res_feat1
                    dec_inputs.push_back(img_encoder_out[3]);
                    dec_inputs.push_back(img_encoder_out[4]);
                }

                // Run decoder
                auto image_decoder_start = std::chrono::high_resolution_clock::now();
                auto img_decoder_result = sam2_base->run_image_decoder(dec_inputs);
                auto image_decoder_end = std::chrono::high_resolution_clock::now();
                auto image_decoder_time = std::chrono::duration_cast<std::chrono::milliseconds>(image_decoder_end - image_decoder_start).count();
                spdlog::debug("image_decoder inference time: {} ms", image_decoder_time);

                if (!img_decoder_result.has_value()) {
                    return Err<std::tuple<int, std::unordered_map<int, Mask>>>(img_decoder_result.error());
                }
                auto img_decoder_out = img_decoder_result.value();

                // Prepare memory encoder inputs
                std::vector<VARP> mec_inputs = {
                    img_decoder_out[1],  // mask_for_mem
                    img_encoder_out[0]   // pix_feat
                };

                if (img_decoder_out.size() > 4) {
                    mec_inputs.push_back(img_decoder_out[4]);  // object_score_logits
                }

                // Run memory encoder
                auto memory_encoder_start = std::chrono::high_resolution_clock::now();
                auto memory_encoder_result = sam2_base->run_memory_encoder(mec_inputs);
                auto memory_encoder_end = std::chrono::high_resolution_clock::now();
                auto memory_encoder_time = std::chrono::duration_cast<std::chrono::milliseconds>(memory_encoder_end - memory_encoder_start).count();
                spdlog::debug("memory_encoder inference time: {} ms", memory_encoder_time);

                if (!memory_encoder_result.has_value()) {
                    return Err<std::tuple<int, std::unordered_map<int, Mask>>>(memory_encoder_result.error());
                }
                auto memory_encoder_out = memory_encoder_result.value();

                // Update object state
                NonCondFrameMaskmem non_cond_frame_maskmem;
                non_cond_frame_maskmem.maskmem_features = memory_encoder_out[0];
                non_cond_frame_maskmem.maskmem_pos_enc = memory_encoder_out[1];
                non_cond_frame_maskmem.frame_idx = frame_idx;
                object_state->add_non_cond_frame_maskmem(non_cond_frame_maskmem);

                NonCondFrameObj non_cond_frame_obj;
                non_cond_frame_obj.obj_ptr = img_decoder_out[0];
                non_cond_frame_obj.frame_idx = frame_idx;
                object_state->add_non_cond_frame_obj(non_cond_frame_obj);

                // Convert VARP to Mask
                float score = img_decoder_out[3]->readMap<float>()[0];
                auto mask = convert_to_mask(img_decoder_out[2], score);
                outputs[obj_id] = mask;
            }
        }

        inference_state.m_pimpl->next_frame_idx++;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        spdlog::debug("track_step total time: {} ms", total_time);

        return std::make_tuple(frame_idx, outputs);
    }
};

SAM2Video::SAM2Video() : m_pimpl(std::make_unique<Impl>()) {}
SAM2Video::~SAM2Video() = default;

Result<bool> SAM2Video::initialize(const std::string& model_path, const Params& params) {
    return m_pimpl->initialize(model_path, params);
}

Result<bool> SAM2Video::initialize(uint8_t* buffer, size_t size, const Params& params) {
    return m_pimpl->initialize(buffer, size, params);
}

Result<std::tuple<int, Mask>> SAM2Video::add_new_points_or_box(
    InferenceState& inference_state,
    const std::string& image_path,
    int obj_id,
    const std::vector<Point>& points,
    const std::vector<BBox>& boxes,
    bool clear_old_points
) {
    return m_pimpl->add_new_points_or_box(inference_state, image_path, obj_id, points, boxes, clear_old_points);
}

Result<std::tuple<int, std::unordered_map<int, Mask>>> SAM2Video::track_step(
    InferenceState& inference_state,
    const std::string& image_path
) {
    return m_pimpl->track_step(inference_state, image_path);
}

} // namespace sam2