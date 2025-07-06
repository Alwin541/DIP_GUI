# Deep Image Prior - Multi-Task GUI (MATLAB)

This project provides an interactive MATLAB GUI for performing **image restoration** tasks using the concept of **Deep Image Prior (DIP)**. DIP leverages the structure of a convolutional neural network (CNN), such as U-Net, to recover images from corrupted versions without needing a dataset for training.

---

## 🎯 Features

- 📁 Load and display images
- 🎛️ Select from 4 tasks:
  - Denoising
  - Deblurring
  - Super-Resolution
  - Inpainting
- 🎚️ Adjustable noise level and blur sigma (task-dependent sliders)
- 🧠 Uses U-Net architecture trained via Deep Image Prior concept
- 📈 Real-time loss curve plotting
- 🖼️ Visualization of corrupted and restored images
- 💾 Save output image and loss curve
- 📊 PSNR and SSIM metrics display

---

## 🛠️ Requirements

- MATLAB R2021a or later (recommended R2022b+)
- Deep Learning Toolbox
- Image Processing Toolbox

Tested on **MATLAB R2025a** with `dlnetwork`, `unet`, and U-Net support.

---

## 🚀 How to Use

1. Open `DIP_GUI_.m` in MATLAB.
2. Run the script.
3. Use the GUI to:
   - Browse and select an image
   - Choose a task
   - Adjust parameters if needed
   - Click **Run DIP** to begin optimization
4. View:
   - Corrupted input
   - DIP output image
   - Loss curve over iterations
   - PSNR & SSIM values
5. Save the output or loss curve as needed.

---

## 🧪 Supported Tasks

| Task              | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| Denoising         | Adds Gaussian noise and learns to restore the clean image                  |
| Deblurring        | Applies Gaussian blur and reconstructs the sharp image                     |
| Super-Resolution  | Downsamples the image and restores high-resolution details                 |
| Inpainting        | Randomly masks image regions and fills them in via DIP                     |




