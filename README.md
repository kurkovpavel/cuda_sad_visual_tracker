# cuda_sad_visual_tracker
GPU Visual Tracker
<img width="963" height="541" alt="1" src="https://github.com/user-attachments/assets/1caa2ca1-625c-4906-b37e-5c63d2e78107" />

# Compiling:

```
mkdir build
cd build
# Choose CUDA arch for your GPU (6.1 here)
cmake -DCUDA_ARCHITECTURES=61 ..
make -j$(nproc)
cd bin
```


# Run:
```
./FastTrackerCUDA
=== FastTracker CUDA ===
Initializing camera...
CUDA Device: NVIDIA GeForce GTX 1060 6GB
Compute Capability: 6.1
Global Memory: 6069 MB
CUDA initialized successfully
Tracker initialized. Click on the image to select template.
Controls:
  Q: Stop tracking
  W/S: Increase/decrease template size
  D/A: Increase/decrease ROI size
  R/F: Increase/decrease confidence threshold
  T/G: Increase/decrease update threshold
  C: Toggle CUDA/CPU mode
  ESC: Exit
Initializing tracker with template size: 30
Tracking started!
```

# Press left mouse button to start tracking.


