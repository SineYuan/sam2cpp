#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

#include "sam2_common.h"

namespace sam2 {

class SAM2Video;

class InferenceState {
    friend class SAM2Video;
public:
    InferenceState();
    ~InferenceState();

    void reset();
    void remove_object(int obj_id);

private:
    class Impl;
    std::unique_ptr<Impl> m_pimpl;
};

class SAM2Video {
public:
    SAM2Video();
    ~SAM2Video();

    Result<bool> initialize(const std::string& model_path, const Params& params);
    Result<bool> initialize(uint8_t* buffer, size_t size, const Params& params);

    // Main interface methods
    Result<std::tuple<int, Mask>> add_new_points_or_box(
        InferenceState& inference_state,
        const std::string& image_path,
        int obj_id,
        const std::vector<Point>& points = {},
        const std::vector<BBox>& boxes = {},
        bool clear_old_points = true
    );

    Result<std::tuple<int, std::unordered_map<int, Mask>>> track_step(
        InferenceState& inference_state,
        const std::string& image_path
    );

private:
    class Impl;
    std::unique_ptr<Impl> m_pimpl;
};

} // namespace sam2