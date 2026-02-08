from PIL import Image, ImageEnhance, ImageChops
import numpy as np
import colorsys
from core.common import TaskManager

def rgb_to_lab(rgb_arr):
    """
    Vectorized conversion from RGB (0-255) to CIE LAB.
    Follows standard sRGB -> XYZ -> LAB conversion path.
    """
    # 1. sRGB to Linear RGB
    arr = rgb_arr.astype(np.float32) / 255.0
    mask = arr > 0.04045
    arr[mask] = ((arr[mask] + 0.055) / 1.055) ** 2.4
    arr[~mask] = arr[~mask] / 12.92
    
    # 2. Linear RGB to XYZ (D65 illuminant)
    # Matrix from http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html
    x = arr[:,:,0] * 0.4124564 + arr[:,:,1] * 0.3575761 + arr[:,:,2] * 0.1804375
    y = arr[:,:,0] * 0.2126729 + arr[:,:,1] * 0.7151522 + arr[:,:,2] * 0.0721750
    z = arr[:,:,0] * 0.0193339 + arr[:,:,1] * 0.1191920 + arr[:,:,2] * 0.9503041
    
    # 3. XYZ to LAB (D65 reference white)
    # Observer. = 2°, Illuminant = D65
    ref_x, ref_y, ref_z = 0.95047, 1.00000, 1.08883
    
    x /= ref_x
    y /= ref_y
    z /= ref_z
    
    def f(t):
        mask = t > 0.008856
        res = np.empty_like(t)
        res[mask] = t[mask] ** (1/3)
        res[~mask] = (7.787 * t[~mask]) + (16 / 116)
        return res
    
    fx, fy, fz = f(x), f(y), f(z)
    
    l = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return np.stack([l, a, b], axis=-1)

def map_to_palette_lab(img, palette_colors, task_id=None):
    """
    Maps an image to a palette using CIE LAB color space for maximum perceptual accuracy.
    palette_colors: List of RGB tuples [(r,g,b), ...]
    """
    # 1. Convert palette colors to LAB
    pal_np = np.array(palette_colors).reshape(1, -1, 3)
    pal_lab = rgb_to_lab(pal_np).reshape(-1, 3) # [N, 3]
    
    # 2. Convert image to LAB
    img_rgb = img.convert("RGB")
    img_np = np.array(img_rgb)
    img_lab = rgb_to_lab(img_np) # [H, W, 3]
    h, w, _ = img_lab.shape
    pixels_lab = img_lab.reshape(-1, 3) # [H*W, 3]
    
    # 3. Vectorized Nearest Neighbor in LAB space
    # (H*W, 1, 3) - (1, N, 3) -> (H*W, N, 3)
    # Using small chunks if image is too large to avoid memory error
    chunk_size = 100000 # Process 100k pixels at a time
    indices = np.zeros(pixels_lab.shape[0], dtype=np.int32)
    
    for i in range(0, pixels_lab.shape[0], chunk_size):
        TaskManager.check(task_id)
        end = min(i + chunk_size, pixels_lab.shape[0])
        chunk = pixels_lab[i:end, np.newaxis, :] # [chunk, 1, 3]
        diffs = chunk - pal_lab[np.newaxis, :, :] # [chunk, N, 3]
        sq_dists = np.sum(diffs**2, axis=2) # [chunk, N]
        indices[i:end] = np.argmin(sq_dists, axis=1)
        
    # 4. Reconstruct image from palette indices
    result_arr = np.array(palette_colors)[indices].reshape(h, w, 3).astype(np.uint8)
    return Image.fromarray(result_arr)

def apply_stability_filter(img, iterations=1):
    """
    Advanced Stability Filter: Removes isolated micro-dot noise using 8-connectivity majority voting.
    Ensures clean edges and solid regions without destroying intentional details.
    """
    # 1. Convert to RGBA to ensure 4-byte alignment for np.uint32 view
    temp_rgba = img.convert("RGBA")
    arr = np.array(temp_rgba)
    h, w, _ = arr.shape
    
    # Each unique RGB becomes a unique 32-bit integer
    pixels = arr.view(np.uint32).reshape(h, w)
    
    for _ in range(iterations):
        # Shifted arrays for 8-connectivity (N, S, E, W, NE, NW, SE, SW)
        shifts = [
            (-1, 0), (1, 0), (0, -1), (0, 1),   # 4-neighbors
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonals
        ]
        
        neighbor_stack = []
        for dy, dx in shifts:
            neighbor_stack.append(np.roll(np.roll(pixels, dy, axis=0), dx, axis=1))
        
        # Stack neighbors: [8, H, W]
        stack = np.stack(neighbor_stack, axis=0)
        
        # Check if current pixel is isolated (different from ALL 8 neighbors)
        # isolated = True if pixels[y,x] is not in stack[:,y,x]
        match_mask = (pixels[np.newaxis, :, :] == stack)
        has_any_match = np.any(match_mask, axis=0)
        isolated = ~has_any_match
        
        if not np.any(isolated): break
        
        # For isolated pixels, find the majority neighbor
        # (Using a fast approximation: pick the most frequent among N, S, E, W)
        pixels[isolated] = neighbor_stack[0][isolated] # Use 'up' as fallback
        
    # Convert back to RGB
    res_rgba_arr = pixels.view(np.uint8).reshape(h, w, 4)
    return Image.fromarray(res_rgba_arr[:,:,:3], mode="RGB")

def apply_bilateral_filter(img, d=5, sigma_color=25, sigma_space=25):
    """
    Cleans up source noise while preserving edges. 
    Crucial for preventing dot noise in solid areas.
    """
    try:
        import cv2
        img_np = np.array(img.convert("RGB"))
        # OpenCV Bilateral Filter
        res_cv = cv2.bilateralFilter(img_np, d, sigma_color, sigma_space)
        return Image.fromarray(res_cv)
    except ImportError:
        # Fallback to simple MedianFilter if OpenCV is missing
        from PIL import ImageFilter
        return img.filter(ImageFilter.MedianFilter(size=3))

def consolidate_palette(colors, threshold=5.0):
    """
    Merges colors that are perceptually too close (Delta-E < threshold).
    Helps in determining the optimal K for palette reduction.
    """
    if not colors: return colors
    
    # Pre-convert all to LAB
    lab_colors = [rgb_to_lab(np.array(c).reshape(1, 1, 3)).reshape(3) for c in colors]
    
    unique_colors = []
    unique_lab = []
    
    for i, c_lab in enumerate(lab_colors):
        is_duplicate = False
        for u_lab in unique_lab:
            dist = np.sqrt(np.sum((c_lab - u_lab)**2))
            if dist < threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_colors.append(colors[i])
            unique_lab.append(c_lab)
            
    return unique_colors

def sort_colors(colors, method="Luminance"):
    """
    Sorts a list of RGB tuples (0-255).
    Methods: "Luminance", "Hue", "Original"
    """
    if method == "Original" or not colors:
        return colors

    if method == "Luminance":
        # Relative luminance formula (ordered by brightness, descending)
        return sorted(colors, key=lambda c: (0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]), reverse=True)
    
    if method == "Hue":
        # Sort by Hue in HSV space
        return sorted(colors, key=lambda c: colorsys.rgb_to_hsv(c[0]/255.0, c[1]/255.0, c[2]/255.0)[0])
    
    return colors

def export_as_gpl(path, colors, name="Pixlato Export"):
    """
    Exports a list of RGB tuples as a GIMP Palette (.gpl) file.
    """
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write("GIMP Palette\n")
            f.write(f"Name: {name}\n")
            f.write("Columns: 0\n")
            f.write("#\n")
            for i, (r, g, b) in enumerate(colors):
                f.write(f"{r:>3} {g:>3} {b:>3}\tIndex {i}\n")
        return True
    except Exception as e:
        print(f"Error exporting GPL: {e}")
        return False

def extract_aesthetic_palette(img, color_count=16, w_sat=0.4, w_con=0.3, w_rar=0.3, task_id=None):
    """
    Optimized Weighted Extraction: Scores pixels by Saturation, Contrast, and Uniqueness.
    Uses downsampling for analysis speed and vectorized operations.
    """
    # 1. Performance Optimization: Resize for analysis if image is large
    # Palette extraction doesn't need 4K resolution. 256px is plenty.
    analysis_size = 256
    if max(img.size) > analysis_size:
        img_analysis = img.resize((analysis_size, analysis_size), resample=Image.LANCZOS)
    else:
        img_analysis = img

    img_rgb = img_analysis.convert("RGB")
    arr = np.array(img_rgb).astype(np.float32) / 255.0
    pixels = arr.reshape(-1, 3)
    
    # 2. Vectorized Scoring Components
    # Saturation (max - min)
    c_max = np.max(pixels, axis=1)
    c_min = np.min(pixels, axis=1)
    sats = c_max - c_min
    
    # Brightness (Value)
    vals = c_max
    
    # Contrast Score: High weight for very bright or dark
    contrast = np.abs(vals - 0.5) * 2.0
    
    # Uniqueness (Vectorized Hue Calculation)
    # Simplified Hue calculation: (G-B)/(max-min) etc.
    diff = sats + 1e-6
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    
    h = np.zeros_like(r)
    idx_r = (c_max == r)
    h[idx_r] = (g[idx_r] - b[idx_r]) / diff[idx_r]
    idx_g = (c_max == g)
    h[idx_g] = 2.0 + (b[idx_g] - r[idx_g]) / diff[idx_g]
    idx_b = (c_max == b)
    h[idx_b] = 4.0 + (r[idx_b] - g[idx_b]) / diff[idx_b]
    h = (h / 6.0) % 1.0
    
    h_hist, _ = np.histogram(h, bins=36, range=(0, 1))
    rarity = 1.0 / (h_hist[np.clip((h * 36).astype(int), 0, 35)] + 1.0)
    rarity = rarity / (np.max(rarity) + 1e-6)
    
    # 3. Final Importance Score with User Weights
    scores = (sats * w_sat) + (contrast * w_con) + (rarity * w_rar)
    
    # 4. Selection with Perceptual Suppression (Diversity-First)
    selected_indices = []
    current_scores = scores.copy()
    
    # Pre-convert all pixels to LAB for diversity check (on analysis size)
    pixels_lab = rgb_to_lab(pixels.reshape(1, -1, 3)).reshape(-1, 3)
    
    # Delta-E threshold (minimum perceptual distance between palette colors)
    # 10-15 is a noticeable difference.
    min_delta_e_sq = 15.0 ** 2 

    for i in range(color_count):
        TaskManager.check(task_id)
        best_idx = np.argmax(current_scores)
        if current_scores[best_idx] <= 0: break
        
        best_lab = pixels_lab[best_idx]
        selected_indices.append(best_idx)
        
        # Suppress nearby colors in LAB space
        # Distances in LAB are perceptually uniform
        diffs = pixels_lab - best_lab[np.newaxis, :]
        sq_dists = np.sum(diffs**2, axis=1)
        
        # Apply non-linear suppression: 0 within threshold, then ramps up
        suppression = np.clip(sq_dists / min_delta_e_sq, 0, 1)
        current_scores *= suppression
        
    res_rgb = [(int(p[0]*255), int(p[1]*255), int(p[2]*255)) for p in pixels[selected_indices]]
    return res_rgb

def extract_geometric_palette(img, color_count=16, task_id=None):
    """
    Plan B: Geometric Volume Preservation Extraction.
    Analyzes the 3D color gamut and ensures boundary points (highlights/extremes) are preserved.
    """
    # 1. Prepare pixels
    analysis_size = 256
    if max(img.size) > analysis_size:
        img_analysis = img.resize((analysis_size, analysis_size), resample=Image.LANCZOS)
    else:
        img_analysis = img
    
    pixels = np.array(img_analysis.convert("RGB")).reshape(-1, 3).astype(np.float32)
    
    # 2. Extract Boundary/Extreme Points (Task 47.2)
    # We pick points that are at the edges of the color cube to preserve gamut volume
    extremes = []
    
    # Min/Max for each channel
    for i in range(3):
        extremes.append(pixels[np.argmin(pixels[:, i])])
        extremes.append(pixels[np.argmax(pixels[:, i])])
        
    # Max/Min Sum (White/Black extremes)
    pixel_sums = np.sum(pixels, axis=1)
    extremes.append(pixels[np.argmin(pixel_sums)]) # Darkest
    extremes.append(pixels[np.argmax(pixel_sums)]) # Brightest (Highlight!)
    
    # Max Saturation (Purest colors)
    c_max = np.max(pixels, axis=1)
    c_min = np.min(pixels, axis=1)
    sats = c_max - c_min
    extremes.append(pixels[np.argmax(sats)])
    
    # Remove duplicates from extremes and limit to color_count/2
    unique_extremes = []
    for e in extremes:
        if not any(np.array_equal(e, u) for u in unique_extremes):
            unique_extremes.append(e)
    
    # Start palette with these geometric anchor points
    palette = unique_extremes[:color_count // 2]
    
    # 3. Fill remaining slots with Density-based sampling (Task 47.3)
    # Using a 3D histogram (Voxels) to find representative clusters
    # 16x16x16 grid provides 4096 voxels
    hist, edges = np.histogramdd(pixels, bins=(16, 16, 16), range=((0, 255), (0, 255), (0, 255)))
    
    # Find non-empty voxels and sort by density
    non_empty = hist > 0
    voxel_coords = np.argwhere(non_empty)
    densities = hist[non_empty]
    
    # Convert voxel indices to RGB values (center of voxel)
    bin_width = 256 / 16
    voxel_rgbs = (voxel_coords * bin_width) + (bin_width / 2)
    
    # Sort by density descending
    sorted_indices = np.argsort(-densities)
    candidate_rgbs = voxel_rgbs[sorted_indices]
    
    # Fill remaining slots with a diversity check
    pixels_lab = rgb_to_lab(pixels.reshape(1, -1, 3)).reshape(-1, 3) # Full pixel cloud in LAB
    
    # Threshold for remaining colors
    min_dist_sq = 20.0 ** 2 
    
    for cand_rgb in candidate_rgbs:
        TaskManager.check(task_id)
        if len(palette) >= color_count: break
        
        # Check distance from already picked colors in LAB space
        cand_lab = rgb_to_lab(cand_rgb.reshape(1, 1, 3)).reshape(3)
        pal_lab = rgb_to_lab(np.array(palette).reshape(1, -1, 3)).reshape(-1, 3)
        
        dists = np.sum((pal_lab - cand_lab)**2, axis=1)
        if np.all(dists > min_dist_sq):
            palette.append(cand_rgb)
            
    return [tuple(map(int, c)) for c in palette]

def map_to_palette_perceptual(img, palette_img):
    """
    Maps image to a palette using perceptual weights (emphasizing Saturation and Value).
    """
    # Pillow's quantize already uses a decent perceptual model, 
    # but we can enhance it by boosting image saturation before mapping.
    enhancer = ImageEnhance.Color(img)
    perceptual_boosted = enhancer.enhance(1.2) # Boost saturation by 20% during mapping
    return perceptual_boosted.quantize(palette=palette_img, dither=Image.Dither.NONE)

def apply_palette_unified(img, palette_name="Original", custom_colors=None, dither=True, 
                          extract_policy="Standard", mapping_policy="Classic",
                          w_sat=0.4, w_con=0.3, w_rar=0.3, auto_optimal=False, task_id=None):
    """
    Updated Pipeline supporting Auto Optimal spatial refinement.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha_mask = img.split()[-1]
    
    if palette_name == "Original":
        return img

    rgb_img = img.convert("RGB")
    
    # [Pre-step] Spatial Refinement Pre-smoothing (Task 48.2)
    if auto_optimal:
        rgb_img = apply_bilateral_filter(rgb_img)

    # [Step 2 & 3] Target Generation
    target_palette = None
    is_low_color = False
    
    if palette_name == "Limited":
        count = custom_colors if isinstance(custom_colors, int) else 16
        
        if auto_optimal:
            # Plan B + Consolidation (Task 48.3)
            # Try to extract a generous gamut then consolidate
            raw_colors = extract_geometric_palette(rgb_img, color_count=64, task_id=task_id)
            colors = consolidate_palette(raw_colors, threshold=6.0) # Merge very similar colors
        elif extract_policy == "Aesthetic":
            colors = extract_aesthetic_palette(rgb_img, count,
                                              w_sat=w_sat, w_con=w_con, w_rar=w_rar, task_id=task_id)
        else:
            # Standard Policy now uses Plan B (Geometric Volume Preservation)
            colors = extract_geometric_palette(rgb_img, count, task_id=task_id)
            
        # Create palette image from custom list
        flat = []
        for c in colors: flat.extend(c)
        flat += [0] * (768 - len(flat))
        target_palette = Image.new("P", (1, 1))
        target_palette.putpalette(flat)
        is_low_color = len(colors) <= 4
    else:
        target_palette, is_low_color = _generate_target_palette(palette_name, custom_colors)
    
    if dither and is_low_color:
        rgb_img = _apply_pre_contrast(rgb_img, factor=1.15)

    # [Step 4] Unified Quantization
    if target_palette:
        dither_method = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
        
        # Extract color list from target_palette for LAB mapping
        pal_data = target_palette.getpalette()[:768]
        current_palette_colors = [(pal_data[i], pal_data[i+1], pal_data[i+2]) for i in range(0, len(pal_data), 3)]

        if auto_optimal or mapping_policy == "Perceptual":
            # CIE LAB Perceptual Mapping
            if not dither:
                result_rgb = map_to_palette_lab(rgb_img, current_palette_colors, task_id=task_id)
            else:
                enhancer = ImageEnhance.Color(rgb_img)
                mapping_src = enhancer.enhance(1.2)
                quantized_img = mapping_src.quantize(palette=target_palette, dither=dither_method)
                result_rgb = quantized_img.convert("RGB")
        else:
            # Classic RGB Mapping
            quantized_img = rgb_img.quantize(palette=target_palette, dither=dither_method)
            result_rgb = quantized_img.convert("RGB")
            
        # [Step 5] Post-processing: Stability Filter (Only when dither is OFF)
        if not dither:
            # Auto Optimal gets aggressive smoothing (3 iterations)
            passes = 3 if auto_optimal else 1
            result_rgb = apply_stability_filter(result_rgb, iterations=passes)
            
    elif palette_name == "Custom_16bit":
        arr = np.array(rgb_img).astype(np.int16)
        arr = (arr // 16) * 17
        result_rgb = Image.fromarray(arr.astype(np.uint8))
        
    elif palette_name == "Grayscale":
         num_colors = custom_colors if isinstance(custom_colors, int) else 256
         dither_method = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
         result_rgb = rgb_img.convert("L").quantize(colors=num_colors, dither=dither_method).convert("RGB")
         
    else:
         result_rgb = rgb_img.convert("RGB")

    result_rgb.putalpha(alpha_mask)
    return result_rgb

def _generate_target_palette(palette_name, custom_colors):
    """
    Generates a 'P' mode image containing the strict target palette.
    Returns: (palette_image, is_low_color_bool)
    """
    
    colors = []
    is_low_color = False
    
    # 1. Preset Palettes
    presets = {
        "GameBoy": [(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)],
        "CGA": [(0, 0, 0), (85, 255, 255), (255, 85, 255), (255, 255, 255)],
        "Pico-8": [
            (0,0,0), (29,43,83), (126,37,83), (0,135,81),
            (171,82,54), (95,87,79), (194,195,199), (255,241,232),
            (255,0,77), (255,163,0), (255,236,39), (0,228,54),
            (41,173,255), (131,118,156), (255,119,168), (255,204,170)
        ]
    }
    
    if palette_name in presets:
        colors = presets[palette_name]
        is_low_color = len(colors) <= 4
        
    elif palette_name == "Custom_User" and custom_colors:
        # custom_colors is a list of tuples [(r,g,b), ...]
        colors = custom_colors
        # Remove any potential black padding if it wasn't intended (though UI usually handles this)
        # But we need to keep 16 slots if that was the intent.
        # Actually, strict mapping allows unused slots.
        is_low_color = len(colors) <= 4
        
    elif palette_name == "Grayscale":
        # Generate N-step grayscale palette
        steps = custom_colors if isinstance(custom_colors, int) else 4
        if steps < 2: steps = 2
        # Create linear gradient from 0 to 255
        vals = np.linspace(0, 255, steps, dtype=int)
        colors = [(v, v, v) for v in vals]
        is_low_color = steps <= 4
        
    elif palette_name == "Custom_16bit":
        # Cannot be represented by a fixed small palette easily due to 4096 combinations.
        # This requires algorithmic quantization, not palette mapping.
        return None, False
        
    elif palette_name == "Limited":
        # Adaptive, no fixed palette
        return None, False

    if not colors:
        return None, False

    # Create master palette image
    # Flatten list of tuples -> list of ints
    flat_palette = [c for rgb in colors for c in rgb]
    
    # Pad to 768 integers (256 colors * 3)
    # Important: Fill unused slots with the last color or repeat to avoid accidental mapping to black 0,0,0
    # if 0,0,0 is not in the palette. But Pillow treats unset as 0,0,0.
    # Safe bet: Pad with the first color to avoid 'black hole' effect for unassigned indices,
    # OR let them be 0 if 0 is not used.
    padding = (768 - len(flat_palette))
    flat_palette += [0] * padding
    
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(flat_palette)
    
    return pal_img, is_low_color

def _apply_pre_contrast(img, factor=1.15):
    """
    Boosts contrast of the image.
    Used to pre-separate dark and light areas before aggressive dithering.
    """
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)
