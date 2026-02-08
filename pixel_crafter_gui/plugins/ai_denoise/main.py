class AIDenoisePlugin(BasePlugin):
    """
    Implements a high-performance smoothing pipeline using PyTorch.
    Supports Zero-Copy Tensor manipulation (run_tensor).
    """
    def run_tensor(self, tensor: torch.Tensor, params: dict) -> torch.Tensor:
        """Direct tensor processing with adjustable smoothing strength."""
        h, w, c = tensor.shape
        img_t = tensor[..., :3].permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # Adjustable Strength (Sigma)
        sigma = params.get("ai_denoise_strength", 1.5)
        k_size = int(math.ceil(sigma * 3) * 2 + 1) # Rule of thumb for kernel size
        k_size = max(3, min(15, k_size)) # Limit to 3x3 ~ 15x15
        
        # 1D coords for Gaussian
        coords = torch.arange(k_size, device=tensor.device).float() - k_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        
        # 2D Kernel
        k2d = g.view(-1, 1) @ g.view(1, -1)
        kernel = k2d.view(1, 1, k_size, k_size).repeat(3, 1, 1, 1)

        # Smooth only RGB
        smoothed = torch.nn.functional.conv2d(img_t, kernel, groups=3, padding=k_size//2)
        
        # Merge back
        res = tensor.clone()
        res[..., :3] = (torch.clamp(smoothed, 0, 1).squeeze(0).permute(1, 2, 0) * 255.0)
        return res

    def run(self, image: Image.Image, params: dict) -> Image.Image:
        """Legacy PIL entry point."""
        arr = np.array(image.convert("RGBA")).astype(np.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.from_numpy(arr).to(device)
        
        processed_t = self.run_tensor(tensor, params)
        
        res_np = processed_t.detach().cpu().byte().numpy()
        return Image.fromarray(res_np, "RGBA")
