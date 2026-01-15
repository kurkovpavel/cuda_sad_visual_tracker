#include "../../include/fast_tracker.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Basic SAD kernel
__global__ void sadKernel(const unsigned char* search, int search_width, int search_height,
                         const unsigned char* template_img, int template_width, int template_height,
                         const unsigned char* mask, int mask_sum,
                         float* result, double max_possible) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int corr_h = search_height - template_height + 1;
    int corr_w = search_width - template_width + 1;

    if (i >= corr_h || j >= corr_w) {
        return;
    }

    double total = 0.0;

    for (int y = 0; y < template_height; ++y) {
        for (int x = 0; x < template_width; ++x) {
            if (mask[y * template_width + x] > 0) {
                unsigned char t_val = template_img[y * template_width + x];
                unsigned char s_val = search[(i + y) * search_width + (j + x)];
                total += abs(static_cast<int>(t_val) - static_cast<int>(s_val));
            }
        }
    }

    double normalized = (max_possible > 1e-5) ? (total / max_possible) : 0.0;
    result[i * corr_w + j] = 1.0f - static_cast<float>(normalized);
}

// Optimized SAD kernel with shared memory
__global__ void optimizedSADKernel(const unsigned char* search, int search_width, int search_height,
                                  const unsigned char* template_img, int template_width, int template_height,
                                  const unsigned char* mask, int mask_sum,
                                  float* result, double max_possible) {

    // Use shared memory for template and mask
    extern __shared__ unsigned char shared_mem[];
    unsigned char* shared_template = shared_mem;
    unsigned char* shared_mask = shared_mem + template_width * template_height;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int corr_h = search_height - template_height + 1;
    int corr_w = search_width - template_width + 1;

    // Load template and mask into shared memory (cooperative loading)
    if (ty < template_height && tx < template_width) {
        int idx = ty * template_width + tx;
        shared_template[idx] = template_img[idx];
        shared_mask[idx] = mask[idx];
    }
    __syncthreads();

    if (i >= corr_h || j >= corr_w) {
        return;
    }

    // Use integer arithmetic for better performance
    int total = 0;

    for (int y = 0; y < template_height; ++y) {
        for (int x = 0; x < template_width; ++x) {
            if (shared_mask[y * template_width + x] > 0) {
                int t_val = shared_template[y * template_width + x];
                int s_val = search[(i + y) * search_width + (j + x)];
                total += abs(t_val - s_val);
            }
        }
    }

    double normalized = (max_possible > 1e-5) ? (total / max_possible) : 0.0;
    result[i * corr_w + j] = 1.0f - static_cast<float>(normalized);
}

// CUDA-accelerated SAD implementation
cv::Mat FastTracker::fastSAD_CUDA(const cv::Mat& search_gray, const cv::Mat& tpl,
                                 const cv::Mat& mask, int mask_sum) {
    if (!cuda_initialized_) {
        std::cerr << "CUDA not initialized, falling back to CPU" << std::endl;
        return fastSAD_CPU(search_gray, tpl, mask, mask_sum);
    }

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
    cv::Mat result_host = cv::Mat::zeros(corr_h, corr_w, CV_32F);

    try {
        // Allocate device memory for search image
        unsigned char* d_search;
        checkCudaError(cudaMalloc(&d_search, sw * sh * sizeof(unsigned char)),
                      "Allocate search memory");
        checkCudaError(cudaMemcpy(d_search, search_gray.data, sw * sh * sizeof(unsigned char),
                      cudaMemcpyHostToDevice), "Copy search to device");

        // Allocate device memory for result
        float* d_result;
        checkCudaError(cudaMalloc(&d_result, corr_h * corr_w * sizeof(float)),
                      "Allocate result memory");

        // Choose kernel based on template size
        dim3 blockSize(16, 16);
        dim3 gridSize((corr_w + blockSize.x - 1) / blockSize.x,
                      (corr_h + blockSize.y - 1) / blockSize.y);

        if (tw * th <= 1024) {  // Use optimized kernel for smaller templates
            size_t shared_mem_size = (tw * th * 2) * sizeof(unsigned char);
            optimizedSADKernel<<<gridSize, blockSize, shared_mem_size>>>(
                d_search, sw, sh,
                d_template_, tw, th,
                d_mask_, mask_sum,
                d_result, max_possible
            );
        } else {  // Use basic kernel for larger templates
            sadKernel<<<gridSize, blockSize>>>(
                d_search, sw, sh,
                d_template_, tw, th,
                d_mask_, mask_sum,
                d_result, max_possible
            );
        }

        // Check for kernel errors
        checkCudaError(cudaGetLastError(), "Kernel execution");
        checkCudaError(cudaDeviceSynchronize(), "Device synchronization");

        // Copy result back to host
        checkCudaError(cudaMemcpy(result_host.data, d_result, corr_h * corr_w * sizeof(float),
                      cudaMemcpyDeviceToHost), "Copy result to host");

        // Free device memory
        cudaFree(d_search);
        cudaFree(d_result);

    } catch (const std::exception& e) {
        std::cerr << "CUDA SAD failed: " << e.what() << std::endl;
        cuda_initialized_ = false;
        return fastSAD_CPU(search_gray, tpl, mask, mask_sum);
    }

    return result_host;
}
