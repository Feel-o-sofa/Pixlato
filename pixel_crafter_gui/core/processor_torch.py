"""
Pixlato - PyTorch/GPU Backend Processor
=======================================
Hardware-accelerated image processing using PyTorch.
Supports CUDA, DirectML, and CPU fallback.
"""

from PIL import Image, ImageFilter
import numpy as np
from core.common import debug_log

# ------------------------------------------------------------------------------
# Lazy Import and Availability Check
# ------------------------------------------------------------------------------

_TORCH_AVAILABLE = None
_TORCH_DEVICE = None
_TORCH_DEVICE_NAME = None # Cache for device name

def is_torch_available():
    """
    Lazily checks if PyTorch is available. Result is cached.
    
    Returns:
        bool: True if PyTorch can be imported, False otherwise
    """
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is None:
        try:
            import torch
            _TORCH_AVAILABLE = True
            debug_log("[Pixlato] PyTorch detected. GPU acceleration enabled.")
        except ImportError:
            _TORCH_AVAILABLE = False
            debug_log("[Pixlato] PyTorch not found. GPU acceleration disabled.")
    return _TORCH_AVAILABLE


def get_torch_device():
    """
    Gets the best available PyTorch device.
    
    Returns:
        torch.device or None if PyTorch unavailable
    """
    global _TORCH_DEVICE
    if not is_torch_available():
        return None
    
    if _TORCH_DEVICE is None:
        import torch
        if torch.cuda.is_available():
            _TORCH_DEVICE = torch.device("cuda")
            _TORCH_DEVICE_NAME = torch.cuda.get_device_name(0)
            debug_log(f"[Pixlato] Using CUDA device: {_TORCH_DEVICE_NAME}")
        else:
            _TORCH_DEVICE = torch.device("cpu")
            debug_log("[Pixlato] CUDA not available. Using CPU for PyTorch operations.")
    
    return _TORCH_DEVICE


# ------------------------------------------------------------------------------
# GPU Memory Management
# ------------------------------------------------------------------------------

# Maximum pixels before forcing CPU fallback (prevents OOM)
GPU_MAX_PIXELS = 4_000_000  # ~2000x2000


def check_gpu_memory_safe(img):
    """
    Checks if the image is small enough for safe GPU processing.
    
    Args:
        img: PIL Image
    
    Returns:
        bool: True if safe for GPU, False if too large
    """
    total_pixels = img.size[0] * img.size[1]
    return total_pixels <= GPU_MAX_PIXELS


# ------------------------------------------------------------------------------
# Downsampling Functions
# ------------------------------------------------------------------------------

def downsample_kmeans_torch(img, pixel_size, out_w, out_h):
    """
    Hardware-accelerated downsampling using PyTorch.
    Vectorizes K-Means logic across all blocks simultaneously.
    
    Automatically falls back to CPU NumPy implementation if:
    - Image is too large (risk of OOM)
    - GPU memory error occurs
    - PyTorch is not available
    
    Args:
        img: PIL Image (RGBA)
        pixel_size: Size of each output pixel block
        out_w: Output width
        out_h: Output height
    
    Returns:
        PIL Image (RGBA)
    """
    if not is_torch_available():
        from core.processor_numpy import downsample_kmeans_numpy
        return downsample_kmeans_numpy(img, pixel_size, out_w, out_h)
    
    # Check image size for GPU safety
    if not check_gpu_memory_safe(img):
        debug_log(f"[Pixlato] Image too large for GPU ({img.size[0]}x{img.size[1]}). Using CPU backend.")
        from core.processor_numpy import downsample_kmeans_numpy
        return downsample_kmeans_numpy(img, pixel_size, out_w, out_h)
    
    try:
        import torch
        device = get_torch_device()
        
        # Convert to float tensor
        arr = np.array(img.convert("RGBA"))
        h, w, _ = arr.shape
        img_tensor = torch.from_numpy(arr).to(device).float()
        
        # Crop to exact required size
        target_h, target_w = out_h * pixel_size, out_w * pixel_size
        img_tensor = img_tensor[:target_h, :target_w, :]
        
        # Reshape to blocks: (out_h, out_w, ps, ps, 4)
        blocks = img_tensor.reshape(out_h, pixel_size, out_w, pixel_size, 4).permute(0, 2, 1, 3, 4)
        num_blocks = out_h * out_w
        block_pixels = blocks.reshape(num_blocks, -1, 4)
        
        rgb = block_pixels[..., :3]
        alpha = block_pixels[..., 3]
        
        # Compute variance to identify high-detail blocks
        block_variances = torch.var(rgb, dim=1).sum(dim=1)
        
        # Initial result: mean for all blocks
        final_rgb = rgb.mean(dim=1)
        
        # Adaptive K-Means for high variance blocks
        var_threshold = 40.0 * 3
        high_var_mask = block_variances > var_threshold
        
        if high_var_mask.any():
            hv_pixels = rgb[high_var_mask]  # (B_hv, N, 3)
            b_hv = hv_pixels.shape[0]
            
            # Initialize 2 clusters: darkest and brightest pixels
            lum = 0.299 * hv_pixels[..., 0] + 0.587 * hv_pixels[..., 1] + 0.114 * hv_pixels[..., 2]
            c0_idx = lum.argmin(dim=1)
            c1_idx = lum.argmax(dim=1)
            
            batch_seq = torch.arange(b_hv, device=device)
            c0 = hv_pixels[batch_seq, c0_idx].unsqueeze(1)
            c1 = hv_pixels[batch_seq, c1_idx].unsqueeze(1)
            centers = torch.cat([c0, c1], dim=1)  # (B_hv, 2, 3)
            
            # K-Means iterations (K=2)
            for _ in range(4):
                dist = torch.cdist(hv_pixels, centers)
                labels = dist.argmin(dim=2)
                
                m0 = (labels == 0).float().unsqueeze(-1)
                m1 = (labels == 1).float().unsqueeze(-1)
                
                centers[:, 0] = (hv_pixels * m0).sum(dim=1) / m0.sum(dim=1).clamp(min=1)
                centers[:, 1] = (hv_pixels * m1).sum(dim=1) / m1.sum(dim=1).clamp(min=1)
            
            # Select cluster furthest from mean (contrast selection)
            hv_means = hv_pixels.mean(dim=1).unsqueeze(1)
            dist_to_mean = torch.norm(centers - hv_means, dim=2)
            best_cluster = dist_to_mean.argmax(dim=1)
            
            final_rgb[high_var_mask] = centers[batch_seq, best_cluster]
        
        # Combine RGB with average alpha
        final_alpha = alpha.mean(dim=1).unsqueeze(-1)
        result_tensor = torch.cat([final_rgb, final_alpha], dim=1)
        
        # Back to PIL
        result_arr = result_tensor.reshape(out_h, out_w, 4).byte().cpu().numpy()
        return Image.fromarray(result_arr, "RGBA")
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            debug_log(f"[Pixlato] GPU OOM detected. Falling back to CPU backend.")
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
            from core.processor_numpy import downsample_kmeans_numpy
            return downsample_kmeans_numpy(img, pixel_size, out_w, out_h)
        raise
    except Exception as e:
        print(f"[Pixlato] PyTorch error: {e}. Falling back to CPU.")
        from core.processor_numpy import downsample_kmeans_numpy
        return downsample_kmeans_numpy(img, pixel_size, out_w, out_h)


# ------------------------------------------------------------------------------
# AI Background Removal
# ------------------------------------------------------------------------------

# Global session cache for rembg
_REMBG_SESSION = None


def is_directml_supported():
    """
    Checks if DirectML (universal Windows GPU acceleration) is supported.
    
    Returns:
        bool: True if DmlExecutionProvider is available
    """
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return 'DmlExecutionProvider' in providers
    except:
        return False


def remove_background_ai_torch(img):
    """
    Uses rembg (AI model) to automatically extract the main subject.

    Provider priority delegated to EngineDispatcher._build_rembg_providers:
      CUDA (NVIDIA) > DirectML (AMD/Intel/NVIDIA Windows) > CPU

    Includes post-processing for clean pixel art edges:
    - Alpha binarization (threshold at 128)
    - Median filter for noise cleanup

    Args:
        img: PIL Image

    Returns:
        PIL Image (RGBA) with transparent background
    """
    global _REMBG_SESSION
    try:
        from rembg import remove, new_session
        from core.processor import EngineDispatcher

        # Initialize session once; provider priority centralized in EngineDispatcher
        if _REMBG_SESSION is None:
            target_providers = EngineDispatcher._build_rembg_providers()
            _REMBG_SESSION = new_session(model_name="silueta", providers=target_providers)
            print(f"[Pixlato] rembg session (torch) initialized with providers: {target_providers}")

        result = remove(img, session=_REMBG_SESSION)

        # Post-process: Alpha binarization for clean pixel art edges
        if result.mode == "RGBA":
            r, g, b, a = result.split()
            # Threshold: <128 -> 0, >=128 -> 255
            a = a.point(lambda p: 255 if p >= 128 else 0)

            # Matte cleanup (remove isolated noise)
            a = a.filter(ImageFilter.MedianFilter(size=3))

            result = Image.merge("RGBA", (r, g, b, a))

        return result

    except ImportError as e:
        print(f"[Pixlato] rembg not available: {e}")
        return img
    except Exception as e:
        print(f"[Pixlato] AI background removal error: {e}")
        return img


def remove_background_interactive_torch(img, bg_seeds, fg_seeds=None):
    """
    Uses OpenCV GrabCut for interactive background removal based on user points.
    
    Args:
        img: PIL Image
        bg_seeds: List of (x, y) coordinates for background
        fg_seeds: List of (x, y) coordinates for foreground (optional)
    
    Returns:
        PIL Image (RGBA) with transparent background
    """
    try:
        import cv2
        
        img_np = np.array(img.convert("RGB"))
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        mask = np.zeros(img_cv.shape[:2], np.uint8)
        mask.fill(cv2.GC_PR_FGD)  # Default: probably foreground
        
        # Apply background seeds
        for x, y in bg_seeds:
            if 0 <= x < img.width and 0 <= y < img.height:
                cv2.circle(mask, (int(x), int(y)), 5, cv2.GC_BGD, -1)
        
        # Apply foreground seeds
        if fg_seeds:
            for x, y in fg_seeds:
                if 0 <= x < img.width and 0 <= y < img.height:
                    cv2.circle(mask, (int(x), int(y)), 5, cv2.GC_FGD, -1)
        
        # GrabCut
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        if bg_seeds:
            cv2.grabCut(img_cv, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        else:
            return img
        
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        res_np = np.array(img.convert("RGBA"))
        res_np[..., 3] = mask2 * 255
        
        return Image.fromarray(res_np, "RGBA")
    
    except ImportError:
        print("[Pixlato] OpenCV not available for interactive background removal.")
        return img
    except Exception as e:
        print(f"[Pixlato] Interactive background removal error: {e}")
        return img
