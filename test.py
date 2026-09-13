# This is a test file
import sys
import tensorflow as tf
import tensorboard
#import gensim

print("=" * 50)
print(f"Python Version: {sys.version.split()[0]}")
print(f"TensorFlow Version: {tf.__version__}")
print(f"TensorBoard Version: {tensorboard.__version__}")
#print(f"Gensim Version: {gensim.__version__}")
print("=" * 50)

# Check for GPU/Hardware Acceleration acceleration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ Success! TensorFlow detected {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  - {gpu}")
else:
    print("ℹ️ TensorFlow is running on your CPU.")
print("=" * 50)

import tensorflow as tf

print("=" * 50)
print(f"TensorFlow Version: {tf.__version__}")
print("=" * 50)

# Check if DirectML successfully routed TensorFlow to your AMD Graphics
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ Success! DirectML detected your AMD hardware:")
    for gpu in gpus:
        print(f"  - {gpu}")
else:
    print("❌ DirectML did not find the GPU. Running on CPU.")
print("=" * 50)

