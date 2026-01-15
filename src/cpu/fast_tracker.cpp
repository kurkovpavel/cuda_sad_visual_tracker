#include "../../include/fast_tracker.cuh"
#include <iostream>
#include <chrono>

// Utility function for CUDA error checking
void checkCudaError(cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << message << "): " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

bool isCudaAvailable() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess || deviceCount == 0) {
        return false;
    }
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    
    std::cout << "CUDA Device: " << deviceProp.name << std::endl;
    std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    
    return true;
}

void FastTracker::reset(int roi_size, int template_size) {
    template_ = cv::Mat();
    template_gray_ = cv::Mat();
    mask_ = cv::Mat();
    mask_binary_ = cv::Mat();
    mask_sum_ = 0;
    t_mean_ = 0.0;
    t_std_ = 1.0;
    tracking_pos_ = cv::Point(-1, -1);
    conf_threshold_ = 0.5;
    update_threshold_ = 0.8;
    roi_size_ = roi_size;
    template_size_ = template_size;
    display_template_ = cv::Mat();
}

bool FastTracker::init(const cv::Mat& frame, const cv::Rect& bbox) {
    template_ = frame(bbox).clone();
    
    if (template_.channels() == 3) {
        cv::cvtColor(template_, template_gray_, cv::COLOR_BGR2GRAY);
    } else if (template_.channels() == 1) {
        template_gray_ = template_.clone();
    } else {
        std::vector<cv::Mat> channels;
        cv::split(template_, channels);
        template_gray_ = channels[0].clone();
    }
    
    updateMask(cv::Point(template_.cols / 2, template_.rows / 2));
    
    if (!use_contours_) {
        tracking_pos_ = cv::Point(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
    } else {
        if (!mask_.empty()) {
            cv::Moments m = cv::moments(mask_);
            if (m.m00 > 0) {
                int cX = static_cast<int>(m.m10 / m.m00);
                int cY = static_cast<int>(m.m01 / m.m00);
                
                int offset_x = cX - template_.cols / 2;
                int offset_y = cY - template_.rows / 2;
                
                tracking_pos_ = cv::Point(
                    bbox.x + bbox.width / 2 + offset_x,
                    bbox.y + bbox.height / 2 + offset_y
                );
                
                int new_x = std::max(0, bbox.x + offset_x);
                int new_y = std::max(0, bbox.y + offset_y);
                int new_w = std::min(bbox.width, frame.cols - new_x);
                int new_h = std::min(bbox.height, frame.rows - new_y);
                
                cv::Rect new_bbox(new_x, new_y, new_w, new_h);
                if (new_bbox.x >= 0 && new_bbox.y >= 0 && 
                    new_bbox.x + new_bbox.width <= frame.cols &&
                    new_bbox.y + new_bbox.height <= frame.rows) {
                    template_ = frame(new_bbox).clone();
                    cv::cvtColor(template_, template_gray_, cv::COLOR_BGR2GRAY);
                    updateMask();
                }
            }
        }
    }
    
    updateDisplayTemplate();
    
    // Initialize CUDA memory after template is set
    if (cuda_initialized_ && !template_gray_.empty() && !mask_.empty()) {
        allocateCUDAMemory(template_gray_, mask_);
    }
    
    return true;
}

std::tuple<cv::Rect, double, cv::Mat> FastTracker::track(const cv::Mat& frame) {
    if (tracking_pos_.x < 0 || tracking_pos_.y < 0 || template_gray_.empty()) {
        return std::make_tuple(cv::Rect(), 0.0, cv::Mat());
    }
    
    int h = frame.rows;
    int w = frame.cols;
    int cx = tracking_pos_.x;
    int cy = tracking_pos_.y;
    int hw = roi_size_ / 2;
    
    int y1 = std::max(0, cy - hw);
    int y2 = std::min(h, cy + hw);
    int x1 = std::max(0, cx - hw);
    int x2 = std::min(w, cx + hw);
    
    cv::Rect roi_rect(x1, y1, x2 - x1, y2 - y1);
    cv::Mat roi = frame(roi_rect);
    
    cv::Mat gray;
    if (roi.channels() == 3) {
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = roi.clone();
    }
    
    cv::Mat corr;
    if (cuda_initialized_) {
        corr = fastSAD_CUDA(gray, template_gray_, mask_, mask_sum_);
    } else {
        corr = fastSAD_CPU(gray, template_gray_, mask_, mask_sum_);
    }
    
    if (corr.empty()) {
        return std::make_tuple(cv::Rect(), 0.0, cv::Mat());
    }
    
    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(corr, &min_val, &max_val, &min_loc, &max_loc);
    
    int dx = max_loc.x;
    int dy = max_loc.y;
    int new_cx = x1 + dx + template_.cols / 2;
    int new_cy = y1 + dy + template_.rows / 2;
    tracking_pos_ = cv::Point(new_cx, new_cy);
    
    cv::Rect bbox = getBbox();
    
    if (max_val > update_threshold_) {
        updateTemplate(frame);
    }
    
    return std::make_tuple(bbox, max_val, corr);
}

cv::Mat FastTracker::fastSAD_CPU(const cv::Mat& search_gray, const cv::Mat& tpl, 
                                const cv::Mat& mask, int mask_sum) {
    int th = tpl.rows;
    int tw = tpl.cols;
    int sh = search_gray.rows;
    int sw = search_gray.cols;
    
    int corr_h = sh - th + 1;
    int corr_w = sw - tw + 1;
    
    if (corr_h <= 0 || corr_w <= 0) {
        return cv::Mat();
    }
    
    double max_possible = 255.0 * mask_sum;
    cv::Mat sad = cv::Mat::zeros(corr_h, corr_w, CV_32F);
    
    for (int i = 0; i < corr_h; ++i) {
        for (int j = 0; j < corr_w; ++j) {
            cv::Rect roi(j, i, tw, th);
            cv::Mat block = search_gray(roi);
            double total = 0.0;
            
            for (int y = 0; y < th; ++y) {
                for (int x = 0; x < tw; ++x) {
                    if (mask.at<uchar>(y, x) > 0) {
                        total += std::abs(tpl.at<uchar>(y, x) - block.at<uchar>(y, x));
                    }
                }
            }
            
            double normalized = (max_possible > 1e-5) ? (total / max_possible) : 0.0;
            sad.at<float>(i, j) = 1.0f - static_cast<float>(normalized);
        }
    }
    
    return sad;
}

void FastTracker::updateMask(cv::Point click_center) {
    if (use_contours_ && !template_gray_.empty()) {
        cv::Mat blur;
        cv::GaussianBlur(template_gray_, blur, cv::Size(5, 5), 0);
        
        cv::Mat thresh;
        cv::threshold(blur, thresh, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        mask_ = cv::Mat::zeros(template_gray_.size(), CV_8UC1);
        
        if (!contours.empty()) {
            std::vector<cv::Point> selected_contour;
            
            if (click_center.x >= 0 && click_center.y >= 0) {
                for (const auto& contour : contours) {
                    if (cv::pointPolygonTest(contour, click_center, false) >= 0) {
                        selected_contour = contour;
                        break;
                    }
                }
            }
            
            if (selected_contour.empty()) {
                double max_area = 0;
                for (const auto& contour : contours) {
                    double area = cv::contourArea(contour);
                    if (area > max_area) {
                        max_area = area;
                        selected_contour = contour;
                    }
                }
            }
            
            if (!selected_contour.empty()) {
                cv::drawContours(mask_, std::vector<std::vector<cv::Point>>{selected_contour}, 
                                -1, cv::Scalar(255), cv::FILLED);
            }
        }
    } else {
        mask_ = cv::Mat(template_gray_.size(), CV_8UC1, cv::Scalar(255));
    }
    
    mask_binary_ = (mask_ > 0);
    mask_sum_ = cv::countNonZero(mask_binary_);
    
    if (correlation_method_ == "ncc" && mask_sum_ > 0) {
        cv::Mat masked_pixels;
        template_gray_.copyTo(masked_pixels, mask_binary_);
        
        cv::Scalar mean, stddev;
        cv::meanStdDev(masked_pixels, mean, stddev, mask_binary_);
        t_mean_ = mean[0];
        t_std_ = std::max(stddev[0], 1e-5);
    }
}

void FastTracker::updateDisplayTemplate() {
    if (!template_.empty() && !mask_.empty()) {
        cv::bitwise_and(template_, template_, display_template_, mask_);
    }
}

void FastTracker::updateTemplate(const cv::Mat& frame) {
    int th = template_gray_.rows;
    int tw = template_gray_.cols;
    int cx = tracking_pos_.x;
    int cy = tracking_pos_.y;
    
    if (use_contours_ && !mask_.empty()) {
        cv::Moments m = cv::moments(mask_);
        if (m.m00 > 0) {
            int obj_cX = static_cast<int>(m.m10 / m.m00);
            int obj_cY = static_cast<int>(m.m01 / m.m00);
            int offset_x = obj_cX - tw / 2;
            int offset_y = obj_cY - th / 2;
            cx += offset_x;
            cy += offset_y;
        }
    }
    
    int x = cx - tw / 2;
    int y = cy - th / 2;
    
    if (x >= 0 && y >= 0 && x + tw <= frame.cols && y + th <= frame.rows) {
        cv::Rect new_bbox(x, y, tw, th);
        template_ = frame(new_bbox).clone();
        cv::cvtColor(template_, template_gray_, cv::COLOR_BGR2GRAY);
        
        if (use_contours_) {
            updateMask(cv::Point(tw / 2, th / 2));
        } else {
            updateMask();
        }
        
        updateDisplayTemplate();
        tracking_pos_ = cv::Point(x + tw / 2, y + th / 2);
        
        // Update CUDA memory with new template
        if (cuda_initialized_) {
            freeCUDAMemory();
            allocateCUDAMemory(template_gray_, mask_);
        }
    }
}

cv::Rect FastTracker::getBbox() {
    if (tracking_pos_.x < 0 || tracking_pos_.y < 0 || template_.empty()) {
        return cv::Rect();
    }
    
    int cx = tracking_pos_.x;
    int cy = tracking_pos_.y;
    int tw = template_.cols;
    int th = template_.rows;
    
    return cv::Rect(cx - tw / 2, cy - th / 2, tw, th);
}

void FastTracker::allocateCUDAMemory(const cv::Mat& template_img, const cv::Mat& mask) {
    if (!cuda_initialized_) return;
    
    template_height_ = template_img.rows;
    template_width_ = template_img.cols;
    size_t template_size = template_width_ * template_height_ * sizeof(unsigned char);
    
    try {
        // Allocate device memory for template
        checkCudaError(cudaMalloc(&d_template_, template_size), "Allocate template memory");
        checkCudaError(cudaMemcpy(d_template_, template_img.data, template_size, 
                      cudaMemcpyHostToDevice), "Copy template to device");
        
        // Allocate device memory for mask
        checkCudaError(cudaMalloc(&d_mask_, template_size), "Allocate mask memory");
        checkCudaError(cudaMemcpy(d_mask_, mask.data, template_size, 
                      cudaMemcpyHostToDevice), "Copy mask to device");
        
    } catch (const std::exception& e) {
        std::cerr << "CUDA memory allocation failed: " << e.what() << std::endl;
        cuda_initialized_ = false;
        freeCUDAMemory();
    }
}

void FastTracker::freeCUDAMemory() {
    if (d_template_) {
        cudaFree(d_template_);
        d_template_ = nullptr;
    }
    if (d_mask_) {
        cudaFree(d_mask_);
        d_mask_ = nullptr;
    }
}

void FastTracker::initializeCUDA() {
    cuda_initialized_ = isCudaAvailable();
    if (cuda_initialized_) {
        std::cout << "CUDA initialized successfully" << std::endl;
    } else {
        std::cout << "CUDA not available, using CPU fallback" << std::endl;
    }
}

FastTracker::FastTracker(bool use_contours, const std::string& correlation_method,
                       int roi_size, int template_size) 
    : d_template_(nullptr), d_mask_(nullptr), cuda_initialized_(false) {
    reset(roi_size, template_size);
    use_contours_ = use_contours;
    correlation_method_ = correlation_method;
    initializeCUDA();
}

FastTracker::~FastTracker() {
    freeCUDAMemory();
}
