"""
Pixlato - NumPy/CPU Backend Processor
=====================================
Pure NumPy and PIL based image processing for CPU-only environments.
This module provides fallback implementations when PyTorch is unavailable.
"""

from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from core.common import TaskManager

# ------------------------------------------------------------------------------
# Downsampling Functions
# ------------------------------------------------------------------------------

def downsample_numpy(img, out_w, out_h):
    """
    Standard downsampling using PIL's BOX resampling.
    Fast and efficient for most use cases.
    
    Args:
        img: PIL Image (RGBA)
        out_w: Output width
        out_h: Output height
    
    Returns:
        PIL Image (RGBA)
    """
    return img.resize((out_w, out_h), resample=Image.BOX)


def downsample_kmeans_numpy(img, pixel_size, out_w, out_h, task_id=None):
    """
    CPU-based adaptive K-Means downsampling using pure NumPy.
    This is a fallback when PyTorch is unavailable.
    
    Uses a simplified 2-cluster approach for high-variance blocks
    to preserve edge detail during pixelation.
    
    Args:
        img: PIL Image (RGBA)
        pixel_size: Size of each output pixel block
        out_w: Output width
        out_h: Output height
    
    Returns:
        PIL Image (RGBA)
    """
    # Convert to numpy array
    arr = np.array(img.convert("RGBA"), dtype=np.float32)
    h, w, _ = arr.shape
    
    # Calculate exact size needed
    target_h, target_w = out_h * pixel_size, out_w * pixel_size
    
    # Crop to exact fit
    arr = arr[:target_h, :target_w, :]
    
    # Reshape to blocks: (out_h, pixel_size, out_w, pixel_size, 4)
    # Then transpose to (out_h, out_w, pixel_size, pixel_size, 4)
    blocks = arr.reshape(out_h, pixel_size, out_w, pixel_size, 4)
    blocks = blocks.transpose(0, 2, 1, 3, 4)  # (out_h, out_w, ps, ps, 4)
    
    # Flatten blocks for easier processing
    num_blocks = out_h * out_w
    block_pixels = blocks.reshape(num_blocks, -1, 4)  # (B, N, 4)
    
    rgb = block_pixels[..., :3]  # (B, N, 3)
    alpha = block_pixels[..., 3]  # (B, N)
    
    # Initial result: mean of all pixels in each block
    final_rgb = np.mean(rgb, axis=1)  # (B, 3)
    final_alpha = np.mean(alpha, axis=1)  # (B,)
    
    # Compute variance per block to identify high-detail areas
    block_variances = np.var(rgb, axis=1).sum(axis=1)  # (B,)
    
    # Threshold for high variance (adjust for sensitivity)
    var_threshold = 40.0 * 3
    high_var_mask = block_variances > var_threshold
    
    if np.any(high_var_mask):
        # Process high-variance blocks with simple 2-cluster K-Means
        hv_indices = np.where(high_var_mask)[0]
        
        for idx in hv_indices:
            TaskManager.check(task_id)
            pixels = rgb[idx]  # (N, 3)
            
            # Calculate luminance for initial cluster selection
            lum = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
            
            # Initialize centers: darkest and brightest
            c0 = pixels[np.argmin(lum)]
            c1 = pixels[np.argmax(lum)]
            
            # Simple K-Means iteration (K=2, 4 iterations)
            for _ in range(4):
                # Assign pixels to nearest center
                dist0 = np.linalg.norm(pixels - c0, axis=1)
                dist1 = np.linalg.norm(pixels - c1, axis=1)
                labels = (dist1 < dist0).astype(int)
                
                # Update centers
                mask0 = labels == 0
                mask1 = labels == 1
                
                if np.any(mask0):
                    c0 = np.mean(pixels[mask0], axis=0)
                if np.any(mask1):
                    c1 = np.mean(pixels[mask1], axis=0)
            
            # Select cluster furthest from mean (contrast selection)
            block_mean = np.mean(pixels, axis=0)
            dist_c0 = np.linalg.norm(c0 - block_mean)
            dist_c1 = np.linalg.norm(c1 - block_mean)
            
            final_rgb[idx] = c1 if dist_c1 > dist_c0 else c0
    
    # Combine RGB and Alpha
    result = np.zeros((num_blocks, 4), dtype=np.float32)
    result[:, :3] = final_rgb
    result[:, 3] = final_alpha
    
    # Reshape back to image
    result_arr = result.reshape(out_h, out_w, 4).astype(np.uint8)
    
    return Image.fromarray(result_arr, "RGBA")


# ------------------------------------------------------------------------------
# Background Removal Functions
# ------------------------------------------------------------------------------

def remove_background_numpy(img, tolerance=50):
    """
    Removes background color by detecting corner colors and using floodfill.
    Pure PIL/NumPy implementation without external dependencies.
    
    Args:
        img: PIL Image
        tolerance: Color tolerance for floodfill (0-255)
    
    Returns:
        PIL Image (RGBA) with transparent background
    """
    try:
        img = img.convert("RGBA")
        width, height = img.size
        
        # Check if already transparent
        if img.getpixel((0, 0))[3] == 0:
            return img
        
        # Floodfill from each corner
        corners = [(0, 0), (width-1, 0), (0, height-1), (width-1, height-1)]
        
        for cx, cy in corners:
            if img.getpixel((cx, cy))[3] != 0:
                ImageDraw.floodfill(img, (cx, cy), (0, 0, 0, 0), thresh=tolerance)
        
        return img
    except Exception as e:
        print(f"[Pixlato] NumPy background removal error: {e}")
        return img


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def ensure_rgba(img):
    """
    Ensures the image is in RGBA mode for consistent processing.
    
    Args:
        img: PIL Image
    
    Returns:
        PIL Image (RGBA)
    """
    if img.mode != "RGBA":
        return img.convert("RGBA")
    return img
