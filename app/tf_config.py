import os
import tensorflow as tf

def configure_tensorflow():
    """Configure TensorFlow for optimal performance in CPU-only environment with limited memory"""
    
    # Disable GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Configure TensorFlow to use less memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Limit TensorFlow's memory usage
    try:
        tf.config.set_logical_device_configuration(
            tf.config.list_physical_devices('CPU')[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    except:
        pass
    
    # Limit intra and inter-op parallelism
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    # Set memory growth
    tf.config.set_soft_device_placement(True)
    
    # Lower precision to reduce memory usage
    tf.keras.mixed_precision.set_global_policy('mixed_float16') 