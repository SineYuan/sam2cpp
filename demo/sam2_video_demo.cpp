#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <iomanip>
#include <sstream>

#include <opencv2/opencv.hpp>

#include "sam2_video.h"
#include "sam2_common.h"

using namespace sam2;

namespace fs = std::filesystem;

const int IMG_SIZE = 1024;

// Helper function to format frame number with leading zeros
std::string formatFrameNumber(int frame_num) {
    std::stringstream ss;
    ss << std::setw(5) << std::setfill('0') << frame_num;
    return ss.str();
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <model_path> <frames_directory> <output_directory> [max_frames]" << std::endl;
    std::cout << "  model_path: Path to the SAM2 model" << std::endl;
    std::cout << "  frames_directory: Directory containing frame images" << std::endl;
    std::cout << "  output_directory: Directory to save output images" << std::endl;
    std::cout << "  max_frames: Maximum number of frames to process (optional, default: 20)" << std::endl;
}

int main(int argc, char** argv) {
    // Check command line arguments
    if (argc < 4 || argc > 5) {
        printUsage(argv[0]);
        return -1;
    }

    // Parse command line arguments
    std::string model_path = argv[1];
    std::string frames_dir = argv[2];
    std::string output_dir = argv[3];
    int max_frame_num_to_track = 20; // Default value
    
    if (argc == 5) {
        try {
            max_frame_num_to_track = std::stoi(argv[4]);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing max_frames argument: " << e.what() << std::endl;
            return -1;
        }
    }

    // Ensure frames directory ends with a separator
    if (!frames_dir.empty() && frames_dir.back() != '/' && frames_dir.back() != '\\') {
        frames_dir += '/';
    }

    // Ensure output directory ends with a separator
    if (!output_dir.empty() && output_dir.back() != '/' && output_dir.back() != '\\') {
        output_dir += '/';
    }

    // Create output directory
    fs::create_directories(output_dir);

    // Initialize SAM2Video and InferenceState
    SAM2Video predictor;
    InferenceState inference_state;

    // Initialize predictor
    Params params;
    auto init_result = predictor.initialize(model_path, params);
    if (!init_result.has_value()) {
        std::cerr << "Failed to initialize predictor" << std::endl;
        return -1;
    }

    // Define test inputs
    struct TestInput {
        int frame_idx;
        int obj_id;
        std::vector<Point> points;
        std::vector<BBox> boxes;
    };

    std::vector<TestInput> test_inputs = {
        {0, 1, {{210, 350, 1}, {250, 220, 1}}, {}},
        {0, 2, {{460, 60, 1}}, {BBox{300, 0, 500, 400}}},
        {150, 1, {{82, 410, 0}}, {}}
    };

    // Process frames
    for (int i = 0; i < max_frame_num_to_track; ++i) {
        std::string frame_path = frames_dir + formatFrameNumber(i) + ".jpg";
        std::cout << "Processing frame: " << frame_path << std::endl;
        
        // Check if frame exists
        if (!fs::exists(frame_path)) {
            std::cerr << "Frame not found: " << frame_path << std::endl;
            break;
        }

        // Check for test inputs at this frame
        for (const auto& input : test_inputs) {
            if (input.frame_idx == i) {
                auto result = predictor.add_new_points_or_box(
                    inference_state,
                    frame_path,
                    input.obj_id,
                    input.points,
                    input.boxes
                );
                if (!result.has_value()) {
                    std::cerr << "Failed to add points/box for frame " << i << std::endl;
                    return -1;
                }
            }
        }

        // Track step
        auto track_result = predictor.track_step(inference_state, frame_path);
        if (!track_result.has_value()) {
            std::cerr << "Failed to track frame " << i << std::endl;
            return -1;
        }

        auto [frame_idx, outputs] = track_result.value();

        // Visualize and save results
        cv::Mat frame = cv::imread(frame_path);
        if (frame.empty()) {
            std::cerr << "Failed to read frame: " << frame_path << std::endl;
            continue;
        }
        
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        // Define colors for different objects
        const std::vector<cv::Scalar> COLORS = {
            cv::Scalar(255, 0, 0),    // Red
            cv::Scalar(0, 255, 0),    // Green
            cv::Scalar(0, 0, 255),    // Blue
            cv::Scalar(255, 255, 0),  // Cyan
            cv::Scalar(255, 0, 255),  // Magenta
            cv::Scalar(0, 255, 255)   // Yellow
        };

        for (auto& [obj_id, mask] : outputs) {
            
            // Create a 2D matrix from the mask data
            cv::Mat mask_mat(mask.height, mask.width, CV_8U, mask.data.data());
            
            // Resize mask to original frame size
            cv::Mat mask_resized;
            cv::resize(mask_mat, mask_resized, frame.size());
            
            // Create colored mask
            cv::Mat colored_mask(frame.size(), CV_8UC3, cv::Scalar(0, 0, 0));
            colored_mask.setTo(COLORS[obj_id % COLORS.size()], mask_resized);

            // Blend with original frame
            cv::addWeighted(frame, 0.7, colored_mask, 0.3, 0, frame);
        }

        // Save result
        cv::Mat result;
        cv::cvtColor(frame, result, cv::COLOR_RGB2BGR);
        std::string output_path = output_dir + formatFrameNumber(i) + ".jpg";
        cv::imwrite(output_path, result);
    }

    return 0;
} 