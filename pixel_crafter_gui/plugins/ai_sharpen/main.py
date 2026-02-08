class AISharpenPlugin(BasePlugin):
    """
    Enhanced Sharpening Plugin using PyTorch.
    Supports Zero-Copy Tensor manipulation (run_tensor) for maximum performance.
    """
    def run_tensor(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        """
        High-performance direct tensor processing.
        Args:
            tensor: (H, W, 4) FloatTensor on GPU/CPU in [0, 255]
        """
        h, w, c = tensor.shape
        # Extract RGB, normalize to [0, 1] and reshape to (1, 3, H, W)
        img_t = tensor[..., :3].permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # Get strength from params (registered in plugin.json)
        strength = params.get("ai_sharpen_strength", 1.0)
        
        # Center is 1 + 4*strength, neighbors are -strength. Sum is 1.
        k_val = 1.0 + 4.0 * strength
        kernel = torch.tensor([
            [ 0, -strength,  0],
            [-strength, k_val, -strength],
            [ 0, -strength,  0]
        ], dtype=torch.float32, device=tensor.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

        # Apply convolution
        sharpened = torch.nn.functional.conv2d(img_t, kernel, groups=3, padding=1)
        
        # Scale back to [0, 255] and clamp
        sharpened = torch.clamp(sharpened * 255.0, 0.0, 255.0)
        
        # Merge back with original alpha channel
        res = tensor.clone()
        res[..., :3] = sharpened.squeeze(0).permute(1, 2, 0)
        return res

    def run(self, image: Image.Image, params: dict) -> Image.Image:
        """Legacy PIL entry point."""
        # We manually use the tensor logic to avoid code duplication
        mode = image.mode
        arr = np.array(image.convert("RGBA")).astype(np.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.from_numpy(arr).to(device)
        
        processed_t = self.run_tensor(tensor, params)
        
        res_np = processed_t.detach().cpu().byte().numpy()
        res_img = Image.fromarray(res_np, "RGBA")
        
        return res_img if mode == "RGBA" else res_img.convert(mode)
