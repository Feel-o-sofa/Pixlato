"""
Pixlato - Core Image Processor (Facade Layer)
==============================================
This module provides the public API for image processing.
Backend selection is handled by EngineDispatcher.
"""

from PIL import Image, ImageFilter
import numpy as np
from core.common import debug_log

# Security: Prevent decompression bomb attacks by limiting max pixels (e.g., 100MP)
Image.MAX_IMAGE_PIXELS = 100_000_000

# ------------------------------------------------------------------------------
# Engine Dispatcher - Backend Selection Logic
# ------------------------------------------------------------------------------

class EngineDispatcher:
    """
    Manages backend selection for image processing operations.
    Supports three modes: 'auto', 'cpu', 'gpu'
    """
    _mode = "auto"  # Default: auto-detect best backend
    _torch_available = None  # Cached availability check
    _has_cuda = None        # Cached CUDA check
    
    @classmethod
    def set_mode(cls, mode):
        """
        Sets the processing mode.
        
        Args:
            mode: 'auto' (default), 'cpu' (force NumPy), or 'gpu' (force PyTorch)
        """
        if mode in ["auto", "cpu", "gpu"]:
            cls._mode = mode
            debug_log(f"[Pixlato] Engine mode set to: {mode}")
    
    @classmethod
    def get_mode(cls):
        """Returns the current engine mode."""
        return cls._mode
    
    @classmethod
    def is_torch_available(cls):
        """
        Lazily checks if PyTorch is available. Result is cached.
        """
        if cls._torch_available is None:
            try:
                from core.processor_torch import is_torch_available
                cls._torch_available = is_torch_available()
            except ImportError:
                cls._torch_available = False
                debug_log("[Pixlato] processor_torch module not found.")
        return cls._torch_available
    
    @classmethod
    def get_backend(cls):
        """
        Determines which backend to use based on mode and availability.

        Returns:
            str: 'gpu' or 'cpu'
        """
        if cls._mode == "cpu":
            return "cpu"

        if cls._mode == "gpu":
            if cls.is_torch_available():
                return "gpu"
            else:
                print("[Pixlato] GPU mode requested but PyTorch unavailable. Using CPU.")
                return "cpu"

        # Auto mode: use GPU if available
        if cls._mode == "auto":
            return "gpu" if cls.is_torch_available() else "cpu"

        return "cpu"

    @classmethod
    def _build_rembg_providers(cls):
        """
        Single source of truth for rembg/ONNX provider priority.

        Respects EngineDispatcher._mode:
          - 'cpu': always returns CPU-only, regardless of available hardware.
          - 'auto' / 'gpu': probes hardware and returns ordered provider list.

        Priority order (non-CPU modes):
          1. CUDAExecutionProvider  — NVIDIA GPU, lowest latency
          2. DmlExecutionProvider   — DirectML (AMD/Intel/NVIDIA, Windows universal)
          3. CPUExecutionProvider   — CPU fallback, always present

        Also populates cls._has_cuda so other methods can query CUDA availability
        without re-importing onnxruntime.

        Returns:
            list[str]: Ordered provider list ready to pass to rembg.new_session().
        """
        # Respect explicit CPU mode: bypass all GPU providers
        if cls._mode == "cpu":
            cls._has_cuda = False
            debug_log("[Pixlato] rembg: CPU mode forced — skipping GPU provider probe.")
            return ["CPUExecutionProvider"]

        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
        except ImportError:
            debug_log("[Pixlato] onnxruntime not found; falling back to CPU provider list.")
            cls._has_cuda = False
            return ["CPUExecutionProvider"]

        ordered = []

        if "CUDAExecutionProvider" in available:
            ordered.append("CUDAExecutionProvider")
            cls._has_cuda = True
            debug_log("[Pixlato] rembg: CUDA provider available — using CUDAExecutionProvider.")
        else:
            cls._has_cuda = False

        if "DmlExecutionProvider" in available:
            ordered.append("DmlExecutionProvider")
            debug_log("[Pixlato] rembg: DirectML provider available — using DmlExecutionProvider.")

        if "CPUExecutionProvider" in available:
            ordered.append("CPUExecutionProvider")

        debug_log(f"[Pixlato] rembg provider order: {ordered}")
        return ordered

def enhance_internal_edges(img, sensitivity=1.0):
    """
    Applies Unsharp Mask to enhance internal edges/details before downsampling.
    This helps preserve features like eyes, nose, mouth during pixelation.
    
    Args:
        img (Image): PIL Image (RGBA).
        sensitivity (float): Edge enhancement strength (0.0 to 2.0).
                            0.0 = No enhancement, 2.0 = Maximum enhancement.
    Returns:
        Image: Enhanced image.
    """
    if sensitivity <= 0:
        return img
    
    # Unsharp Mask parameters scale with sensitivity
    # radius: controls blur size (higher = more global contrast)
    # percent: controls intensity of sharpening
    # threshold: controls which edges are affected (lower = more edges)
    radius = 0.5 + (sensitivity * 0.75)  # 0.5 to 2.0
    percent = int(100 + (sensitivity * 75))  # 100 to 250
    threshold = max(1, int(5 - sensitivity * 2))  # 5 to 1
    
    enhanced = img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
    return enhanced

def remove_background(img, tolerance=50):
    """
    Removes the background color (detected from corners) by making it transparent.
    Uses floodfill from corners.
    """
    try:
        from PIL import ImageDraw
        img = img.convert("RGBA")
        width, height = img.size
        
        # Sample corners
        corners = [(0, 0), (width-1, 0), (0, height-1), (width-1, height-1)]
        
        # We fill from each corner
        # Using a relatively high tolerance by default (50) to catch JPEG artifacts if any,
        # but for pixel art 0 or 10 is better. Let's stick to 20 usually.
        # User doesn't control tolerance yet, so hardcode 40.
        
        # Need to verify if the corner is transparent already
        if img.getpixel((0,0))[3] == 0: return img

        # Floodfill with (0,0,0,0) - Transparent
        # Note: thresh requires PIL > 8.something.
        ImageDraw.floodfill(img, (0, 0), (0, 0, 0, 0), thresh=tolerance)
        
        # Check other corners if they are still opaque and match the "removed" color logic 
        # (visual check handles it, blindly flooding corners is safer if they are consistent)
        
        # Top-Right
        if img.getpixel((width-1, 0))[3] != 0:
             ImageDraw.floodfill(img, (width-1, 0), (0, 0, 0, 0), thresh=tolerance)
             
        # Bottom-Left
        if img.getpixel((0, height-1))[3] != 0:
             ImageDraw.floodfill(img, (0, height-1), (0, 0, 0, 0), thresh=tolerance)

        # Bottom-Right
        if img.getpixel((width-1, height-1))[3] != 0:
             ImageDraw.floodfill(img, (width-1, height-1), (0, 0, 0, 0), thresh=tolerance)
             
        return img
    except Exception as e:
        print(f"Background Remove Error: {e}")
        return img

def pixelate_image(img, pixel_size, target_width=None, edge_enhance=False, edge_sensitivity=1.0, downsample_method="Standard", plugin_engine=None, plugin_params=None, task_id=None):
    """
    [PUBLIC API - Signature must remain stable]
    
    Processes a PIL image and reduces its resolution.
    Uses EngineDispatcher to select the optimal backend (GPU/CPU) via ProcessingContext.
    """
    if img is None:
        return None

    from core.context import ProcessingContext
    backend = EngineDispatcher.get_backend()
    
    # Init context with source image
    ctx = ProcessingContext(img, device="cuda" if backend == "gpu" else "cpu")

    # 1. PRE-PROCESS: Edge Enhancement
    if edge_enhance and edge_sensitivity > 0:
        enhanced = enhance_internal_edges(ctx.get_pil(), edge_sensitivity)
        ctx.set_pil(enhanced)

    # 2. [Hook] PRE_DOWNSAMPLE (Context-aware)
    if plugin_engine:
        # Update plugin_params with task_id for contextual awareness if needed
        params = (plugin_params or {}).copy()
        params["_task_id"] = task_id
        ctx = plugin_engine.execute_hook("PRE_DOWNSAMPLE", ctx, params)

    original_width, original_height = ctx.size
    
    # Output dimensions
    small_width = max(1, original_width // pixel_size)
    small_height = max(1, original_height // pixel_size)

    # 3. DOWNSAMPLE
    if downsample_method == "K-Means":
        # Implementation in processor_torch/processor_numpy already handles their own logic.
        # Here we use the legacy wrapper which now smartly dispatches.
        small_img_pil = downsample_kmeans_adaptive(ctx.get_pil(), pixel_size, small_width, small_height, task_id=task_id)
        ctx.set_pil(small_img_pil) # Downsampling changes resolution, we reset context to new small image
    else:
        # Standard BOX (PIL is fastest for simple resize)
        small_img_pil = ctx.get_pil().resize((small_width, small_height), resample=Image.BOX)
        ctx.set_pil(small_img_pil)

    # 4. [Hook] POST_DOWNSAMPLE (Context-aware)
    if plugin_engine:
        params = (plugin_params or {}).copy()
        params["_task_id"] = task_id
        ctx = plugin_engine.execute_hook("POST_DOWNSAMPLE", ctx, params)

    # Return final PIL image from context
    return ctx.get_pil()



def downsample_kmeans_adaptive(img, pixel_size, out_w, out_h, task_id=None):
    """
    [LEGACY WRAPPER - For backward compatibility]
    
    Hardware-accelerated downsampling using adaptive backend selection.
    Delegates to processor_torch or processor_numpy based on EngineDispatcher.
    
    Args:
        img: PIL Image (RGBA)
        pixel_size: Size of each output pixel block
        out_w: Output width
        out_h: Output height
    
    Returns:
        PIL Image (RGBA)
    """
    backend = EngineDispatcher.get_backend()
    if backend == "gpu":
        # Use PyTorch GPU backend
        try:
            from core.processor_torch import downsample_kmeans_torch
            return downsample_kmeans_torch(img, pixel_size, out_w, out_h)
        except ImportError:
            # Fallback to NumPy if import fails
            from core.processor_numpy import downsample_kmeans_numpy
            return downsample_kmeans_numpy(img, pixel_size, out_w, out_h, task_id=task_id)
    else:
        # Use NumPy CPU backend
        from core.processor_numpy import downsample_kmeans_numpy
        return downsample_kmeans_numpy(img, pixel_size, out_w, out_h, task_id=task_id)

def normalize_image_geometry(img, target_size, strategy="Fit & Pad", bg_color=(0, 0, 0, 0)):
    """
    Standardizes image dimensions based on a target size and geometric strategy.
    
    Strategies:
    - "Stretch" / "Compress": Simple Nearest Neighbor resize.
    - "Pad" / "Fit & Pad": Maintain ratio, fit inside target, center and pad.
    - "Center Crop": Center the image and crop/pad around it.
    """
    if img.size == target_size:
        return img
        
    tw, th = target_size
    iw, ih = img.size
    
    if strategy in ["Stretch", "Compress"]:
        return img.resize((tw, th), Image.NEAREST)
        
    elif strategy in ["Pad", "Fit & Pad"]:
        # Calculate scaling ratio to fit inside
        ratio = min(tw / iw, th / ih)
        nw, nh = max(1, int(iw * ratio)), max(1, int(ih * ratio))
        
        # Resize scaled image
        res_scaled = img.resize((nw, nh), Image.NEAREST)
        
        # Create canvas and paste
        canvas = Image.new("RGBA", (tw, th), bg_color)
        ox, oy = (tw - nw) // 2, (th - nh) // 2
        
        # Use split()[3] if RGBA for proper transparency mask
        mask = res_scaled.split()[3] if res_scaled.mode == "RGBA" else None
        canvas.paste(res_scaled, (ox, oy), mask=mask)
        return canvas
        
    elif strategy == "Center Crop":
        canvas = Image.new("RGBA", (tw, th), bg_color)
        # Calculate paste offset
        ox, oy = (tw - iw) // 2, (th - ih) // 2
        
        # If image is larger, we need to crop it
        # If image is smaller, we just paste it in center (padding)
        # Paste handles both (if ox/oy are negative, it crops)
        mask = img.split()[3] if img.mode == "RGBA" else None
        canvas.paste(img, (ox, oy), mask=mask)
        return canvas
        
    return img

def is_directml_supported():
    """
    Checks if DirectML (universal Windows GPU acceleration) is supported.
    """
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return 'DmlExecutionProvider' in providers
    except:
        return False

# Global session cache for rembg to avoid reloading the model
REMBG_SESSION = None

def remove_background_ai(img):
    """
    Uses rembg (AI model) to automatically extract the main subject.

    Provider priority (handled by EngineDispatcher._build_rembg_providers):
      CUDA (NVIDIA) > DirectML (AMD/Intel/NVIDIA Windows) > CPU
    """
    global REMBG_SESSION
    try:
        from rembg import remove, new_session

        # Initialize session once; provider priority centralized in EngineDispatcher
        if REMBG_SESSION is None:
            target_providers = EngineDispatcher._build_rembg_providers()
            REMBG_SESSION = new_session(model_name="silueta", providers=target_providers)

        result = remove(img, session=REMBG_SESSION)

        # Post-process: Alpha Thresholding/Binarization
        # Pixel Art requires clean edges. Semi-transparent residue causes dirty outlines.
        if result.mode == "RGBA":
            r, g, b, a = result.split()
            # Threshold: Values < 128 become 0, >= 128 become 255
            a = a.point(lambda p: 255 if p >= 128 else 0)

            # Matte Cleanup: remove small isolated noise pixels
            from PIL import ImageFilter
            a = a.filter(ImageFilter.MedianFilter(size=3))

            result = Image.merge("RGBA", (r, g, b, a))

        return result
    except Exception as e:
        print(f"AI Background Removal Error: {e}")
        return img

def remove_background_interactive(img, bg_seeds, fg_seeds=None):
    """
    Uses OpenCV GrabCut to interactively remove background based on user points.
    bg_seeds: list of (x, y) coordinates for background
    fg_seeds: list of (x, y) coordinates for foreground (optional)

    When GPU backend is active, routes through processor_torch for architectural
    consistency. Note: GrabCut itself is a CPU-bound algorithm regardless of backend;
    the routing ensures all processing paths flow through the same module under GPU mode.
    """
    if EngineDispatcher.get_backend() == "gpu":
        try:
            from core.processor_torch import remove_background_interactive_torch
            return remove_background_interactive_torch(img, bg_seeds, fg_seeds)
        except Exception as e:
            # Catch all exceptions (ImportError, cv2.error, OOM, etc.) so the
            # CPU path below is always reachable as a guaranteed fallback.
            debug_log(f"[Pixlato] GPU interactive BG removal failed ({type(e).__name__}: {e}); falling back to CPU GrabCut.")

    # CPU path: direct OpenCV GrabCut
    try:
        import cv2
        img_np = np.array(img.convert("RGB"))
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        mask = np.zeros(img_cv.shape[:2], np.uint8)
        mask.fill(cv2.GC_PR_FGD)  # Default to probably foreground

        for x, y in bg_seeds:
            if 0 <= x < img.width and 0 <= y < img.height:
                cv2.circle(mask, (int(x), int(y)), 5, cv2.GC_BGD, -1)

        if fg_seeds:
            for x, y in fg_seeds:
                if 0 <= x < img.width and 0 <= y < img.height:
                    cv2.circle(mask, (int(x), int(y)), 5, cv2.GC_FGD, -1)

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
    except Exception as e:
        print(f"Interactive Background Removal Error: {e}")
        return img

def apply_grain_effect(img, intensity=15):
    """
    Adds film grain noise to the image.
    Uses Numpy for high performance.
    """
    if intensity <= 0:
        return img
        
    # Convert to RGBA
    img = img.convert("RGBA")
    arr = np.array(img).astype(np.float32)
    
    # Generate monochromatic noise
    # We apply the same noise to R, G, B to avoid color shifting
    noise = np.random.randint(-intensity, intensity + 1, size=(arr.shape[0], arr.shape[1], 1))
    
    # Apply noise to RGB channels only
    arr[..., :3] += noise
    
    # Clamp values
    arr[..., :3] = np.clip(arr[..., :3], 0, 255)
    
    return Image.fromarray(arr.astype(np.uint8), "RGBA")

def upscale_for_preview(small_img, original_size):
    """Upscales a small image to a larger size using NEAREST neighbor."""
    return small_img.resize(original_size, resample=Image.NEAREST)

def save_image(img, path):
    img.save(path)

def add_outline(img, color=(0, 0, 0, 255)):
    """
    Adds a 1-pixel outline around non-transparent pixels.
    Assumes 'img' is RGBA.
    """
    # Create a mask of non-transparent pixels
    # Alpha > 0 considered 'content'
    # We can use ImageFilter.MaxFilter(3) on the alpha channel to dilate it.
    
    img = img.convert("RGBA")
    r, g, b, a = img.split()
    
    # Binarize alpha: 0 or 255
    mask = a.point(lambda p: 255 if p > 0 else 0)
    
    # Dilate the mask (expand by 1 pixel)
    # MaxFilter(3) looks at 3x3 window, taking max value.
    # Center pixel gets 255 if any neighbor is 255.
    dilated_mask = mask.filter(ImageFilter.MaxFilter(3))
    
    # The outline is (Dilated - Original)
    # However, we want the outline to be *behind* the original pixels regarding transparency?
    # Usually outline is drawn *around*, so it expands the sprite. 
    # Or strict outline replacing border pixels?
    # Standard outline: New pixels at border.
    
    # Let's subtract original mask from dilated mask to find the "new" edge pixels.
    # PIL doesn't have direct subtract for images easily without numpy or ImageChops.
    from PIL import ImageChops
    outline_area = ImageChops.difference(dilated_mask, mask)
    
    # Ideally we just composite.
    # Paste the outline color where dilated_mask is white, then paste original image over it.
    
    # Create solid color image
    outline_img = Image.new("RGBA", img.size, color)
    
    # Create a new blank image
    result = Image.new("RGBA", img.size, (0,0,0,0))
    
    # Paste outline using dilated mask
    result.paste(outline_img, (0,0), mask=dilated_mask)
    
    # Paste original image over it (restoring original colors)
    result.paste(img, (0,0), mask=img)
    
    return result
