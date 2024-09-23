import warnings
import tensorflow as tf
from utils.utils import load_image, run_inference, load_configs
import yaml

warnings.filterwarnings('ignore', category=UserWarning, module='numpy')

if __name__ == "__main__":
    config = load_configs("config/config.yaml")
    device = config["system"]["device"]
    os = config["system"]["os"]

    if device == "cpu" and os == "windows":
        from pycoral.utils.edgetpu import make_interpreter
        interpreter = make_interpreter(config["model"])
    elif device == "tpu" and os == "windows":
        interpreter = tf.lite.Interpreter(config["model"])
    elif device == "tpu" and os == "rpi":
        import tflite_runtime.interpreter as tflite
        experimental_delegates=[tflite.load_delegate(config['delegate'])]
        interpreter = tflite.Interpreter(
        model_path=config["model"],  
        experimental_delegates=experimental_delegates)
    else:
        interpreter = tf.lite.Interpreter(config["model"])
    interpreter.allocate_tensors()

    # Load and preprocess the image
    image_path = config["data"]["image_path"]  # Update the path to your image file
    # image_array = load_image(image_path)

    # Run inference 1000 times on the same image
    run_inference(image_path, interpreter, num_iterations=config["num_iterations"])


# ***************** CPU ****************************
# Total time for 1000 inferences: 8.90 seconds
# Average time per inference: 0.0089 seconds

# ****************** TPU ***************************
# Running inferences: 100%|██████████| 1000/1000 [00:02<00:00, 346.58it/s]
# Predicted Class: 65
# Total time for 1000 inferences: 2.89 seconds
# Average time per inference: 0.0029 seconds