"""
Pixlato - Intelligent Processing Context
=========================================
Manages data synchronization between PIL, NumPy, and PyTorch tensors.
Enables Zero-Copy pipelines and Lazy Synchronization.
"""

from PIL import Image
import numpy as np

class ProcessingContext:
    def __init__(self, image=None, device="cpu"):
        self.device = device
        
        # Data Containers
        self._pil = None
        self._tensor = None
        self._ndarray = None
        
        # Synchronization State
        # Indicates which data formats are up-to-date
        self._dirty = {
            "pil": True,
            "tensor": True,
            "ndarray": True
        }
        
        if image is not None:
            self.set_pil(image)

    def set_pil(self, img):
        """Sets the PIL image as the source of truth."""
        self._pil = img.convert("RGBA") if img.mode != "RGBA" else img
        self._dirty["pil"] = False
        self._dirty["tensor"] = True
        self._dirty["ndarray"] = True
        
    def get_pil(self):
        """Returns the synchronized PIL image."""
        if self._dirty["pil"]:
            self._sync_to_pil()
        return self._pil

    def get_tensor(self, device=None):
        """Returns the synchronized PyTorch tensor."""
        target_device = device or self.device
        if self._dirty["tensor"] or (self._tensor is not None and str(self._tensor.device) != str(target_device)):
            self._sync_to_tensor(target_device)
        return self._tensor

    def get_ndarray(self):
        """Returns the synchronized NumPy array."""
        if self._dirty["ndarray"]:
            self._sync_to_ndarray()
        return self._ndarray

    def update_tensor(self, tensor):
        """Updates the context with a modified tensor (GPU/CPU)."""
        self._tensor = tensor
        self._dirty["tensor"] = False
        self._dirty["pil"] = True
        self._dirty["ndarray"] = True

    def update_ndarray(self, array):
        """Updates the context with a modified NumPy array."""
        self._ndarray = array
        self._dirty["ndarray"] = False
        self._dirty["pil"] = True
        self._dirty["tensor"] = True

    # --- Internal Sync Logic ---

    def _sync_to_pil(self):
        if not self._dirty["tensor"] and self._tensor is not None:
            # Sync from Tensor
            arr = self._tensor.detach().cpu().byte().numpy()
            self._pil = Image.fromarray(arr, "RGBA")
        elif not self._dirty["ndarray"] and self._ndarray is not None:
            # Sync from NDArray
            self._pil = Image.fromarray(self._ndarray.astype(np.uint8), "RGBA")
        
        self._dirty["pil"] = False

    def _sync_to_tensor(self, device):
        import torch
        if not self._dirty["pil"] and self._pil is not None:
            # Sync from PIL
            arr = np.array(self._pil)
            self._tensor = torch.from_numpy(arr).to(device).float()
        elif not self._dirty["ndarray"] and self._ndarray is not None:
            # Sync from NDArray
            self._tensor = torch.from_numpy(self._ndarray).to(device).float()
            
        self._dirty["tensor"] = False

    def _sync_to_ndarray(self):
        if not self._dirty["pil"] and self._pil is not None:
            # Sync from PIL
            self._ndarray = np.array(self._pil).astype(np.float32)
        elif not self._dirty["tensor"] and self._tensor is not None:
            # Sync from Tensor
            self._ndarray = self._tensor.detach().cpu().numpy()
            
        self._dirty["ndarray"] = False

    def clear_gpu(self):
        """Explicitly release GPU memory if tensor is held."""
        if self._tensor is not None:
            import torch
            del self._tensor
            self._tensor = None
            self._dirty["tensor"] = True
            torch.cuda.empty_cache()

    @property
    def size(self):
        """Returns (width, height) of the current image."""
        if self._pil:
            return self._pil.size
        elif self._ndarray is not None:
            return (self._ndarray.shape[1], self._ndarray.shape[0])
        elif self._tensor is not None:
            return (self._tensor.shape[1], self._tensor.shape[0])
        return (0, 0)
