"""
Image Manager for handling multiple images in inventory.
Supports GIF frame extraction and batch processing.
"""
from PIL import Image, ImageSequence
from PIL.Image import DecompressionBombError
import os
import json

class ImageManager:
    MAX_IMAGES = 256
    
    def __init__(self):
        # Each entry: {"id": int, "path": str, "name": str, "pil_image": Image, "thumbnail": Image}
        self.images = []
        self._next_id = 0
    
    def _extract_pixlato_metadata(self, img):
        """Attempts to extract Pixlato parameters from PNG metadata."""
        if hasattr(img, "info") and "Pixlato:Params" in img.info:
            try:
                return json.loads(img.info["Pixlato:Params"])
            except:
                pass
        return None

    def add_image(self, path):
        """
        Adds an image to the inventory.
        If it's a GIF, extracts all frames as separate images.
        Returns list of added IDs.
        """
        if len(self.images) >= self.MAX_IMAGES:
            print(f"Inventory full. Max {self.MAX_IMAGES} images.")
            return []
        
        added_ids = []
        
        try:
            img = Image.open(path)
            # Detect metadata
            embedded_params = self._extract_pixlato_metadata(img)
            
            filename = os.path.basename(path)
            name_base, ext = os.path.splitext(filename)
            
            # Check if it's an animated GIF
            if ext.lower() == '.gif' and hasattr(img, 'n_frames') and img.n_frames > 1:
                # Extract all frames
                for i, frame in enumerate(ImageSequence.Iterator(img)):
                    if len(self.images) >= self.MAX_IMAGES:
                        break
                    
                    frame_copy = frame.convert("RGBA")
                    entry = {
                        "id": self._next_id,
                        "path": path,
                        "name": f"{name_base}_{i}",
                        "pil_image": frame_copy,
                        "thumbnail": self._create_thumbnail(frame_copy),
                        "params": embedded_params, # Assign embedded parameters
                        "is_dirty": False,
                        "is_active_global": True, # For Phase 51
                        "bg_processed_image": None,
                        "last_bg_params": None,
                        "processed_image": None,
                        "preview_image": None
                    }
                    self.images.append(entry)
                    added_ids.append(self._next_id)
                    self._next_id += 1
            else:
                # Single image
                img_rgba = img.convert("RGBA")
                entry = {
                    "id": self._next_id,
                    "path": path,
                    "name": name_base,
                    "pil_image": img_rgba,
                    "thumbnail": self._create_thumbnail(img_rgba),
                    "params": embedded_params, # Assign embedded parameters
                    "is_dirty": False,
                    "is_active_global": True,
                    "bg_processed_image": None,
                    "last_bg_params": None,
                    "processed_image": None,
                    "preview_image": None
                }
                self.images.append(entry)
                added_ids.append(self._next_id)
                self._next_id += 1
                
        except DecompressionBombError:
            print(f"Security: DecompressionBomb detected at {path}. Image too large.")
        except Exception as e:
            print(f"Error adding image: {e}")
            
        return added_ids
    
    def remove_image(self, image_id):
        """Removes an image by its ID."""
        self.images = [img for img in self.images if img["id"] != image_id]
    
    def get_image(self, image_id):
        """Gets an image entry by ID."""
        for img in self.images:
            if img["id"] == image_id:
                return img
        return None
    
    def get_all(self):
        """Returns all image entries."""
        return self.images
    
    def update_image_params(self, image_id, params):
        """Updates parameters for a specific image ID."""
        for img in self.images:
            if img["id"] == image_id:
                img["params"] = params
                return True
        return False

    def clear(self):
        """Clears all images."""
        self.images = []
        self._next_id = 0
    
    def count(self):
        return len(self.images)
    
    def _create_thumbnail(self, img, size=(64, 64)):
        """Creates a thumbnail for display in inventory."""
        thumb = img.copy()
        thumb.thumbnail(size, Image.LANCZOS)
        return thumb
