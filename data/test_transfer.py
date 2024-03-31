import torch
import time

def measure_transfer_rate_images(batch_size, channels, height, width, iterations=100):
    """
    Measures the transfer rate for a batch of images from CPU to GPU.

    :param batch_size: Number of images in a batch
    :param channels: Number of channels per image (e.g., 3 for RGB)
    :param height: Height of each image
    :param width: Width of each image
    :param iterations: Number of iterations to average over
    :return: Transfer rate in GB/s
    """
    # Create a batch of images on the CPU
    images = torch.randn(batch_size, channels, height, width, dtype=torch.float32)
    
    # Calculate the total size of the batch in bytes
    total_size_bytes = images.nelement() * images.element_size()
    
    # Warm-up GPU and establish initial CUDA context
    _ = torch.cuda.FloatTensor(1)
    
    start_time = time.time()
    for _ in range(iterations):
        # Transfer the batch of images to GPU
        _ = images.cuda()
    torch.cuda.synchronize()  # Wait for all GPU operations to complete
    end_time = time.time()
    
    # Calculate average transfer time
    avg_time = (end_time - start_time) / iterations
    
    # Calculate transfer rate in GB/s
    transfer_rate = (total_size_bytes / 1e9) / avg_time
    
    return transfer_rate, avg_time

# Example image sizes and batch sizes to test
image_params = [
    (32, 3, 96, 96),  # 32 RGB images of 96x96 pixels (common for deep learning models)
    (256, 3, 96, 96),  # 256 RGB images of 96x96 pixels
    (1024, 3, 96, 96), # 1024 RGB images of 96x96 pixels
    (6400, 3, 96, 96), # 6400 RGB images of 96x96 pixels
    (12800, 3, 96, 96) # 12800 RGB images of 96x96 pixels
]

for batch_size, channels, height, width in image_params:
    rate, avg_time = measure_transfer_rate_images(batch_size, channels, height, width)
    print(f"Transfer Rate for batch of {batch_size} {channels}x{height}x{width} images: {rate:.3f} GB/s, Avg Time: {avg_time:.4f} s")
