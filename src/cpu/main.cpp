#include "../../include/fast_tracker.cuh"
#include <iostream>
#include <chrono>
#include <tuple>
#include <iomanip>

// Global variables for mouse callback
cv::Point last_click(-1, -1);
bool tracking_active = false;
int template_size = 30;
int roi_size = 120;
bool use_cuda = true;

void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        last_click = cv::Point(x, y);
    }
}

int main() {
    std::cout << "=== FastTracker CUDA ===" << std::endl;
    std::cout << "Initializing camera..." << std::endl;

    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

    FastTracker tracker(false, "sad");
    cv::Size heatmap_window_size(640, 360);

    cv::namedWindow("Frame", cv::WINDOW_NORMAL);
    cv::namedWindow("Heat Map", cv::WINDOW_NORMAL);
    cv::resizeWindow("Heat Map", heatmap_window_size);
    cv::setMouseCallback("Frame", mouseCallback, nullptr);

    auto last_time = std::chrono::steady_clock::now();
    double fps = 0.0;

    std::cout << "Tracker initialized. Click on the image to select template." << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  Q: Stop tracking" << std::endl;
    std::cout << "  W/S: Increase/decrease template size" << std::endl;
    std::cout << "  D/A: Increase/decrease ROI size" << std::endl;
    std::cout << "  R/F: Increase/decrease confidence threshold" << std::endl;
    std::cout << "  T/G: Increase/decrease update threshold" << std::endl;
    std::cout << "  C: Toggle CUDA/CPU mode" << std::endl;
    std::cout << "  ESC: Exit" << std::endl;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Process click
        if (last_click.x >= 0 && last_click.y >= 0) {
            int half = template_size / 2;
            int x = std::max(0, last_click.x - half);
            int y = std::max(0, last_click.y - half);
            int w = std::min(template_size, frame.cols - x);
            int h = std::min(template_size, frame.rows - y);

            if (w > 0 && h > 0) {
                cv::Rect bbox(x, y, w, h);
                std::cout << "Initializing tracker with template size: " << template_size << std::endl;
                tracker.init(frame, bbox);
                tracking_active = true;
                std::cout << "Tracking started!" << std::endl;
            } else {
                std::cout << "Error: Template extends beyond frame boundaries" << std::endl;
            }
            last_click = cv::Point(-1, -1);
        }

        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time);
        fps = 1000.0 / elapsed.count();
        last_time = current_time;

        // Process keyboard input
        int key = cv::waitKey(1);
        if (key == 27) break; // ESC
        else if (key == 'q') {
            tracking_active = false;
            tracker.reset();
            std::cout << "Tracking stopped" << std::endl;
        }
        else if (key == 'w') {
            template_size = std::min(100, template_size + 10);
            std::cout << "Template size: " << template_size << std::endl;
        }
        else if (key == 's') {
            template_size = std::max(10, template_size - 10);
            std::cout << "Template size: " << template_size << std::endl;
        }
        else if (key == 'd') {
            roi_size = std::min(1000, roi_size + 20);
            std::cout << "ROI size: " << roi_size << std::endl;
        }
        else if (key == 'a') {
            roi_size = std::max(120, roi_size - 20);
            std::cout << "ROI size: " << roi_size << std::endl;
        }
        else if (key == 'r') {
            tracker.setConfThreshold(std::min(1.0, tracker.getConfThreshold() + 0.1));
            std::cout << "Confidence threshold: " << tracker.getConfThreshold() << std::endl;
        }
        else if (key == 'f') {
            tracker.setConfThreshold(std::max(0.1, tracker.getConfThreshold() - 0.1));
            std::cout << "Confidence threshold: " << tracker.getConfThreshold() << std::endl;
        }
        else if (key == 't') {
            tracker.setUpdateThreshold(std::min(1.0, tracker.getUpdateThreshold() + 0.05));
            std::cout << "Update threshold: " << tracker.getUpdateThreshold() << std::endl;
        }
        else if (key == 'g') {
            tracker.setUpdateThreshold(std::max(0.05, tracker.getUpdateThreshold() - 0.05));
            std::cout << "Update threshold: " << tracker.getUpdateThreshold() << std::endl;
        }
        else if (key == 'c') {
            use_cuda = !use_cuda;
            std::cout << (use_cuda ? "Using CUDA acceleration" : "Using CPU fallback") << std::endl;
        }

        // Tracking
        cv::Rect bbox;
        double score = 0.0;
        cv::Mat corr;

        if (tracking_active) {
            tracker.setRoiSize(roi_size);
            auto result = tracker.track(frame);
            bbox = std::get<0>(result);
            score = std::get<1>(result);
            corr = std::get<2>(result);
        }

        // Visualization
        // Visualization
        if (!bbox.empty() && score > tracker.getConfThreshold()) {
            cv::rectangle(frame, bbox, cv::Scalar(0, 0, 255), 2);
            std::stringstream score_text;
            score_text << "Score: " << std::fixed << std::setprecision(2) << score;
            cv::putText(frame, score_text.str(),
                       cv::Point(bbox.x, bbox.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

            cv::Point roi_tl(bbox.x + bbox.width/2 - roi_size/2,
                           bbox.y + bbox.height/2 - roi_size/2);
            cv::Point roi_br(bbox.x + bbox.width/2 + roi_size/2,
                           bbox.y + bbox.height/2 + roi_size/2);

            cv::rectangle(frame, roi_tl, roi_br, cv::Scalar(255, 255, 255), 1);

            cv::Mat display_tpl = tracker.getDisplayTemplate();
            if (!display_tpl.empty()) {
                int tpl_h = display_tpl.rows;
                int tpl_w = display_tpl.cols;

                // More robust boundary checking
                int tpl_x = roi_br.x;
                int tpl_y = roi_tl.y;

                // Ensure the template fits within frame boundaries
                if (tpl_x >= 0 && tpl_y >= 0 &&
                    tpl_x + tpl_w <= frame.cols &&
                    tpl_y + tpl_h <= frame.rows) {
                    cv::Rect tpl_roi(tpl_x, tpl_y, tpl_w, tpl_h);
                    display_tpl.copyTo(frame(tpl_roi));
                }
            }
        }

        // Heatmap
        if (!corr.empty()) {
            cv::Mat hm;
            cv::normalize(corr, hm, 0, 255, cv::NORM_MINMAX);
            hm.convertTo(hm, CV_8U);
            cv::applyColorMap(hm, hm, cv::COLORMAP_JET);
            cv::resize(hm, hm, heatmap_window_size);
            cv::imshow("Heat Map", hm);
        }

        // Draw UI info
        int y_pos = 30;
        cv::putText(frame, "FPS: " + std::to_string(static_cast<int>(fps)),
                   cv::Point(10, y_pos), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        y_pos += 30;
        cv::putText(frame, "Template: " + std::to_string(template_size),
                   cv::Point(10, y_pos), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        y_pos += 30;
        cv::putText(frame, "ROI: " + std::to_string(roi_size),
                   cv::Point(10, y_pos), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
        y_pos += 30;
        cv::putText(frame, use_cuda ? "CUDA: ON" : "CUDA: OFF",
                   cv::Point(10, y_pos), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                   use_cuda ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        y_pos += 30;
        cv::putText(frame, "Conf: " + std::to_string(tracker.getConfThreshold()).substr(0, 3),
                   cv::Point(10, y_pos), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 100, 100), 2);

        cv::imshow("Frame", frame);
    }

    cap.release();
    cv::destroyAllWindows();
    std::cout << "Application closed" << std::endl;

    return 0;
}
