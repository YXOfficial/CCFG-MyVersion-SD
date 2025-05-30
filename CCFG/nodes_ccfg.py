# --- START OF FILE nodes_ccfg.py ---
import torch
import logging

class CCFGNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "tau": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model_patches"
    DESCRIPTION = "Applies CCFG-like adaptive guidance scaling. (Note: This is a simplified version due to sampler limitations.)"

    def patch(self, model, tau: float = 1.0):
        m = model.clone()

        # This is the function that will be called by the sampler
        def ccfg_guidance_function(args):
            cond_scale = args['cond_scale']        # Standard CFG scale
            cond_denoised = args['cond_denoised']   # Denoised output from positive prompt
            uncond_denoised = args['uncond_denoised'] # Denoised output from negative/unconditional prompt

            # Original shapes are typically (batch_size, channels, height, width)
            original_shape = cond_denoised.shape
            if len(original_shape) == 0:
                # Fallback to standard CFG if tensors are unexpectedly empty
                logging.warning("CCFG-Lite: Empty tensor shape, falling back to standard CFG.")
                return uncond_denoised + cond_scale * (cond_denoised - uncond_denoised)

            batch_size = original_shape[0]

            # Calculate the guidance vector: (positive_denoised - negative_denoised)
            # This is the 'diff' or 'guidance_vector' that we want to adaptively scale.
            guidance_vector = cond_denoised - uncond_denoised

            # Calculate L2 norm of the guidance vector
            # Flatten starting from the first dimension (channels) for sum
            # (B, C, H, W) -> (B, C*H*W)
            guidance_vector_flat = guidance_vector.reshape(batch_size, -1)
            
            # Squared L2 norm for each item in batch
            squared_l2_norm = torch.sum(guidance_vector_flat ** 2, dim=1, keepdim=True)

            # Apply tau parameter
            l2norm_scaled = tau * squared_l2_norm

            # Calculate the adaptive scaling factor using the negative scaling function from CCFG
            # scaling_neg = 2 * (exp(-l2norm) / (1 + exp(-l2norm)))
            # This function attenuates the guidance when l2norm_scaled is large.
            # It goes from ~1.0 (when l2norm_scaled is small) down to ~0.0 (when l2norm_scaled is large)
            adaptive_scale = 2 * (torch.exp(-l2norm_scaled) / (1 + torch.exp(-l2norm_scaled) + 1e-8)) # Add epsilon for stability

            # Reshape adaptive_scale for broadcasting with original tensor shapes
            # (batch_size, 1) -> (batch_size, 1, 1, 1) for 4D tensors like (B,C,H,W)
            adaptive_scale_reshaped = adaptive_scale.view(batch_size, *([1] * (len(original_shape) - 1)))

            # Apply the CCFG-like formula:
            # combined_pred = uncond_denoised + cond_scale * guidance_vector * adaptive_scale_reshaped
            combined_pred = uncond_denoised + cond_scale * guidance_vector * adaptive_scale_reshaped
            
            return combined_pred

        # "ccfg_guidance" is a unique name for this specific patch
        m.set_model_sampler_post_cfg_function(ccfg_guidance_function, "ccfg_guidance")
        logging.debug(f"CCFG-Lite: Model patched with tau = {tau}")
        return (m,)

# For ComfyUI compatibility (optional, but good practice)
NODE_CLASS_MAPPINGS = {
    "CCFGNode": CCFGNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CCFGNode": "YX-CCFG-Lite Guidance Patcher"
}

# --- END OF FILE nodes_ccfg.py ---
