"""
Target Model Interface: google/gemma-2-2b-it.
Provides controlled PyTorch forward hook primitives for activation capture, residual patching, and zero-ablation.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any


class GemmaTargetInterface:
    """
    Controlled target model interface for google/gemma-2-2b-it.
    Uses native PyTorch forward hooks on model.model.layers[layer_idx].
    """

    def __init__(self, model_id: str = "google/gemma-2-2b-it", mock: bool = False):
        self.model_id = model_id
        self.mock = mock
        self.device = "cuda" if torch.cuda.is_available() and not mock else "cpu"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() and not mock else torch.float32
        self.model = None
        self.tokenizer = None

        if not self.mock:
            self._load_model()

    def _load_model(self):
        """Load Hugging Face Gemma 2 2B IT model & tokenizer."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None
        )
        if not torch.cuda.is_available():
            self.model = self.model.to("cpu")
        self.model.eval()

    def run_inference(self, prompt: str, max_new_tokens: int = 10) -> str:
        """Run target model inference deterministically (temperature=0)."""
        if self.mock:
            return prompt + " [Mock Gemma Response]"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text

    def capture_activation(self, prompt: str, layer_idx: int, position_idx: int) -> Tuple[torch.Tensor, str]:
        """Capture residual-stream activation tensor output[0][:, pos:pos+1, :] at (layer_idx, position_idx)."""
        if self.mock:
            dummy_tensor = torch.randn(1, 1, 2304)
            return dummy_tensor, prompt + " [Mock Gemma Response]"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        seq_len = inputs["input_ids"].shape[1]
        pos = max(0, min(position_idx, seq_len - 1))
        
        target_layer = self.model.model.layers[layer_idx]
        captured = []

        def hook_fn(module, args, output):
            # Output is a tuple (hidden_states, attn_weights, ...)
            if isinstance(output, tuple):
                act = output[0][:, pos:pos+1, :].detach().clone()
            else:
                act = output[:, pos:pos+1, :].detach().clone()
            captured.append(act)

        handle = target_layer.register_forward_hook(hook_fn)
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
        handle.remove()

        text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return captured[0], text

    def patch_activation(
        self, 
        source_prompt: str, 
        target_prompt: str, 
        layer_idx: int, 
        source_pos: int, 
        target_pos: int
    ) -> Tuple[str, str, float, torch.Tensor]:
        """
        Capture residual activation from source_prompt at source_pos,
        and replace residual activation in target_prompt at target_pos during forward pass.
        """
        if self.mock:
            base_out = target_prompt + " Rome."
            patch_out = target_prompt + " Paris."
            return base_out, patch_out, 35.5, torch.randn(1, 1, 2304)

        # 1. Capture source activation
        source_tensor, _ = self.capture_activation(source_prompt, layer_idx, source_pos)

        # 2. Target baseline
        target_baseline = self.run_inference(target_prompt)

        # 3. Patch forward pass
        inputs = self.tokenizer(target_prompt, return_tensors="pt").to(self.device)
        seq_len = inputs["input_ids"].shape[1]
        t_pos = max(0, min(target_pos, seq_len - 1))
        target_layer = self.model.model.layers[layer_idx]
        delta_norms = []

        def patch_hook(module, args, output):
            if isinstance(output, tuple):
                hidden = output[0].clone()
                orig = hidden[:, t_pos:t_pos+1, :].clone()
                hidden[:, t_pos:t_pos+1, :] = source_tensor.to(hidden.device, dtype=hidden.dtype)
                delta_norms.append(torch.norm(hidden[:, t_pos:t_pos+1, :] - orig).item())
                return (hidden,) + output[1:]
            else:
                hidden = output.clone()
                orig = hidden[:, t_pos:t_pos+1, :].clone()
                hidden[:, t_pos:t_pos+1, :] = source_tensor.to(hidden.device, dtype=hidden.dtype)
                delta_norms.append(torch.norm(hidden[:, t_pos:t_pos+1, :] - orig).item())
                return hidden

        handle = target_layer.register_forward_hook(patch_hook)
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
        handle.remove()

        patched_out = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        delta_norm = delta_norms[0] if delta_norms else 0.0

        return target_baseline, patched_out, delta_norm, source_tensor

    def ablate_activation(
        self, 
        prompt: str, 
        layer_idx: int, 
        position_idx: int
    ) -> Tuple[str, str, float]:
        """
        Zero-ablate residual stream activation at (layer_idx, position_idx) during forward pass.
        Returns (baseline_output, ablated_output, ablate_delta_norm).
        """
        if self.mock:
            base_out = prompt + " Rome."
            ablated_out = prompt + " [Ablated Response]"
            return base_out, ablated_out, 48.2

        baseline_out = self.run_inference(prompt)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        seq_len = inputs["input_ids"].shape[1]
        pos = max(0, min(position_idx, seq_len - 1))
        target_layer = self.model.model.layers[layer_idx]
        delta_norms = []

        def ablate_hook(module, args, output):
            if isinstance(output, tuple):
                hidden = output[0].clone()
                orig = hidden[:, pos:pos+1, :].clone()
                hidden[:, pos:pos+1, :] = 0.0
                delta_norms.append(torch.norm(orig).item())
                return (hidden,) + output[1:]
            else:
                hidden = output.clone()
                orig = hidden[:, pos:pos+1, :].clone()
                hidden[:, pos:pos+1, :] = 0.0
                delta_norms.append(torch.norm(orig).item())
                return hidden

        handle = target_layer.register_forward_hook(ablate_hook)
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
        handle.remove()

        ablated_out = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        delta_norm = delta_norms[0] if delta_norms else 0.0

        return baseline_out, ablated_out, delta_norm
