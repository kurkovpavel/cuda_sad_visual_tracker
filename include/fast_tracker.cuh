#ifndef FAST_TRACKER_CUH
#define FAST_TRACKER_CUH

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <string>

class FastTracker {
private:
    // CPU members
    cv::Mat template_;
    cv::Mat template_gray_;
    cv::Mat mask_;
    cv::Mat mask_binary_;
    cv::Point tracking_pos_;
    
    bool use_contours_;
    std::string correlation_method_;
    
    double t_mean_;
    double t_std_;
    int mask_sum_;
    
    double conf_threshold_;
    double update_threshold_;
    int roi_size_;
    int template_size_;
    
    cv::Mat display_template_;

    // CUDA device pointers
    unsigned char* d_template_;
    unsigned char* d_mask_;
    int template_width_, template_height_;
    bool cuda_initialized_;

public:
    // Constructor & Destructor
    FastTracker(bool use_contours = true, 
                const std::string& correlation_method = "sad",
                int roi_size = 120,
                int template_size = 30);
    ~FastTracker();
    
    // Core methods
    void reset(int roi_size = 120, int template_size = 30);
    bool init(const cv::Mat& frame, const cv::Rect& bbox);
    std::tuple<cv::Rect, double, cv::Mat> track(const cv::Mat& frame);
    
    // CUDA-accelerated methods
    cv::Mat fastSAD_CUDA(const cv::Mat& search_gray, const cv::Mat& tpl, 
                        const cv::Mat& mask, int mask_sum);
    cv::Mat fastSAD_CPU(const cv::Mat& search_gray, const cv::Mat& tpl, 
                       const cv::Mat& mask, int mask_sum);
    
    // Getters & Setters
    cv::Mat getDisplayTemplate() const { return display_template_; }
    double getConfThreshold() const { return conf_threshold_; }
    double getUpdateThreshold() const { return update_threshold_; }
    void setConfThreshold(double threshold) { conf_threshold_ = threshold; }
    void setUpdateThreshold(double threshold) { update_threshold_ = threshold; }
    void setRoiSize(int size) { roi_size_ = size; }
    void setTemplateSize(int size) { template_size_ = size; }
    bool isCudaInitialized() const { return cuda_initialized_; }

private:
    // Internal methods
    void updateMask(cv::Point click_center = cv::Point(-1, -1));
    void updateDisplayTemplate();
    void updateTemplate(const cv::Mat& frame);
    cv::Rect getBbox();
    
    // CUDA memory management
    void allocateCUDAMemory(const cv::Mat& template_img, const cv::Mat& mask);
    void freeCUDAMemory();
    void initializeCUDA();
};

// CUDA kernel declarations
extern __global__ void sadKernel(const unsigned char* search, int search_width, int search_height,
                                const unsigned char* template_img, int template_width, int template_height,
                                const unsigned char* mask, int mask_sum,
                                float* result, double max_possible);

extern __global__ void optimizedSADKernel(const unsigned char* search, int search_width, int search_height,
                                         const unsigned char* template_img, int template_width, int template_height,
                                         const unsigned char* mask, int mask_sum,
                                         float* result, double max_possible);

// Utility functions
void checkCudaError(cudaError_t err, const char* message);
bool isCudaAvailable();

#endif
