#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include "sam2_image.h"

using namespace cv;
using namespace std;

using namespace sam2;

// Store click coordinates and type
struct PointData {
    cv::Point pos;
    string type;  // "positive", "negative", "rectangle"
    cv::Point end_pos;  // For rectangle end point
};

vector<PointData> click_positions;
char current_mode = 'p';  // 'p' for positive point, 'n' for negative point, 'b' for box
cv::Point start_point;
bool is_dragging = false;
Mat frame;
Mat mask_binary;  // For storing binary mask
SAM2Image predictor;

// Mouse callback function
void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {  // Left button down
        if (current_mode == 'b') {
            start_point = cv::Point(x, y);
            is_dragging = true;
        } else {
            string type = (current_mode == 'p') ? "positive" : "negative";
            click_positions.push_back({cv::Point(x, y), type, cv::Point()});
            cout << "Added " << type << " point at: (" << x << ", " << y << ")" << endl;
        }
    } else if (event == EVENT_MOUSEMOVE) {  // Mouse movement
        if (is_dragging && current_mode == 'b') {
            // Update preview box in real-time while dragging
            Mat frame_copy = frame.clone();
            if (!mask_binary.empty()) {
                Mat overlay = frame_copy.clone();
                overlay.setTo(Scalar(0, 255, 0), mask_binary);
                addWeighted(frame_copy, 0.7, overlay, 0.3, 0, frame_copy);
            }
            rectangle(frame_copy, start_point, cv::Point(x, y), Scalar(0, 255, 0), 2);
            imshow("SAM2 Demo", frame_copy);
        }
    } else if (event == EVENT_LBUTTONUP) {  // Left button up
        if (is_dragging && current_mode == 'b') {
            cv::Point end_point(x, y);
            click_positions.push_back({start_point, "rectangle", end_point});
            cout << "Added rectangle: start(" << start_point.x << ", " << start_point.y 
                 << "), end(" << end_point.x << ", " << end_point.y << ")" << endl;
            is_dragging = false;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <model_path> <image_path>" << endl;
        return -1;
    }

    // Read image
    frame = imread(argv[2]);
    if (frame.empty()) {
        cout << "Cannot read image: " << argv[2] << endl;
        return -1;
    }

    // Initialize SAM2
    Params params;
    auto init_result = predictor.initialize(argv[1], params);
    if (!init_result) {
        cout << "SAM2 initialization failed: " << init_result.error().message << endl;
        return -1;
    }

    // Get image embedding
    auto embedding_result = predictor.get_embedding(argv[2]);
    if (!embedding_result) {
        cout << "Failed to get image embedding: " << embedding_result.error().message << endl;
        return -1;
    }
    
    auto embedding = embedding_result.value();
    
    cout << "Key controls:" << endl;
    cout << "'p' key: Switch to positive point mode" << endl;
    cout << "'n' key: Switch to negative point mode" << endl;
    cout << "'b' key: Switch to box drawing mode" << endl;
    cout << "'q' key: Quit program" << endl;
    cout << "ESC key: Clear all annotations" << endl;
    cout << "Enter key: Generate mask" << endl;
    
    // Set window name
    namedWindow("SAM2 Demo");
    // Set mouse callback function
    setMouseCallback("SAM2 Demo", mouseCallback);
    
    while (true) {
        // Display original image
        Mat display = frame.clone();
        
        // Draw all annotations on the image
        for (const auto& pos : click_positions) {
            if (pos.type == "rectangle") {
                rectangle(display, pos.pos, pos.end_pos, Scalar(0, 255, 0), 2);
            } else {
                Scalar color = (pos.type == "positive") ? Scalar(0, 0, 255) : Scalar(255, 0, 0);
                circle(display, pos.pos, 5, color, -1);
            }
        }

        // If there's a mask, overlay it
        if (!mask_binary.empty()) {
            Mat overlay = display.clone();
            overlay.setTo(Scalar(0, 255, 0), mask_binary);
            addWeighted(display, 0.7, overlay, 0.3, 0, display);
        }
        
        // Display image
        imshow("SAM2 Demo", display);
        
        // Get key press
        char key = waitKey(1);
        if (key == 'q') {  // 'q' key to quit
            break;
        } else if (key == 27) {  // ESC key to clear all annotations
            click_positions.clear();
            mask_binary = Mat();
            cout << "Cleared all annotations" << endl;
        } else if (key == 'p') {  // Switch to positive point mode
            current_mode = 'p';
            cout << "Switched to positive point mode" << endl;
        } else if (key == 'n') {  // Switch to negative point mode
            current_mode = 'n';
            cout << "Switched to negative point mode" << endl;
        } else if (key == 'b') {  // Switch to box drawing mode
            current_mode = 'b';
            cout << "Switched to box drawing mode" << endl;
        } else if (key == 13) {  // Enter key for prediction
            std::vector<sam2::Point> points;
            std::vector<BBox> bboxes;
            
            // Collect all points and boxes
            for (const auto& pos : click_positions) {
                if (pos.type == "rectangle") {
                    BBox bbox;
                    bbox.x_min = pos.pos.x;
                    bbox.y_min = pos.pos.y;
                    bbox.x_max = pos.end_pos.x;
                    bbox.y_max = pos.end_pos.y;
                    bboxes.push_back(bbox);
                } else {
                    sam2::Point point;
                    point.x = pos.pos.x;
                    point.y = pos.pos.y;
                    point.label = (pos.type == "positive") ? 1 : 0;
                    points.push_back(point);
                }
            }
            
            if (points.empty() && bboxes.empty()) {
                cout << "Please add points or boxes first" << endl;
                continue;
            }
            
            // Predict mask
            auto mask_result = predictor.predict(embedding, points, bboxes);
            if (!mask_result) {
                cout << "Mask prediction failed: " << mask_result.error().message << endl;
            } else {
                // Get mask
                const Mask& mask = mask_result.value();
                
                // Create mask image and resize to match original image
                Mat temp_mask(mask.height, mask.width, CV_8UC1, const_cast<uint8_t*>(mask.data.data()));
                resize(temp_mask, mask_binary, frame.size());
                
                cout << "Successfully generated mask, confidence: " << mask.score << endl;
            }
        }
    }
    
    destroyAllWindows();
    return 0;
}
