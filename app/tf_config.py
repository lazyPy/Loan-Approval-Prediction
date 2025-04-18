import os
import tensorflow as tf

def configure_tensorflow():
    """Configure TensorFlow for optimal performance in CPU-only environment with limited memory"""
    
    # Disable GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Disable TensorFlow logging except for errors
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Configure memory growth and limit GPU visibility
    try:
        # Configure TensorFlow to use less memory
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass
    
    # Limit TensorFlow's memory usage
    try:
        tf.config.set_logical_device_configuration(
            tf.config.list_physical_devices('CPU')[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=512)])  # Reduced from 1024
    except:
        pass
    
    # Limit intra and inter-op parallelism
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    # Enable memory growth and soft placement
    tf.config.set_soft_device_placement(True)
    
    # Use dynamic memory growth
    physical_devices = tf.config.list_physical_devices('CPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        pass
    
    # Set lower precision to reduce memory usage
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Additional memory optimizations
    try:
        # Disable eager execution (reduces memory usage)
        tf.compat.v1.disable_eager_execution()
        # Optimize for inference
        tf.config.optimizer.set_jit(True)
    except:
        pass 