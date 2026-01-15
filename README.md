# cuda_sad_visual_tracker
GPU Visual Tracker

Compiling:

mkdir build /n
cd build /n
# Choose CUDA arch for your GPU (6.1 here) /n
cmake -DCUDA_ARCHITECTURES=61 .. /n
make -j$(nproc) /n
cd bin /n


# Run:
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

# Press left mouse button the object and track it...

