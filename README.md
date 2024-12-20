# Comparative Analysis of Hardware-Accelerated Inference for Real-Time UAV Imagery

This project demonstrates a comparative analysis of hardware-accelerated inference frameworks to process real-time UAV imagery. The study includes the evaluation of inference performance across different devices (CPU, Coral Edge TPU) and operating systems (Windows, Raspberry Pi).

---

## How to Run the Code

1. **Connect the Coral Edge TPU**
   - Ensure the Coral Edge TPU is properly connected to the computer.
   - Verify that the LED on the Coral Edge TPU is ON, indicating it is powered and ready.

2. **Make Appropriate Changes in the Config File**
   - **Model Path**: Specify the path to the TensorFlow Lite model (`.tflite`) you want to use.
   - **Operating System**: Set the `os` field to your operating system (`windows` or `rpi`).
   - **Device**: Choose the device type by setting the `device` field to either `cpu` or `tpu`.
   - **Image Directory**: Update the `image_path` field with the correct directory or path of the image(s) you want to analyze.

3. **Run the Script**
   Execute the script using the following command:
   ```bash
   python run_script.py

---

## Example of configuration file
```
  system:
    device: "tpu"         # Options: "cpu", "tpu"
    os: "windows"         # Options: "windows", "rpi"
  model: "models/model.tflite"
  delegate: "libedgetpu.so.1.0"  # Required only for TPU on Raspberry Pi
  data:
    image_path: "data/images/"  # Path to the image(s)
  num_iterations: 1000          # Number of iterations for performance evaluation
```



