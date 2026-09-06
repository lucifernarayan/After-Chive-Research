"""
Target Model Interface: google/gemma-2-2b-it.
Provides controlled PyTorch forward hook primitives for activation capture, residual patching, and zero-ablation.
Supports secure Hugging Face authentication via HF_TOKEN environment variable for gated model access.
"""

import os
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
        """Load Hugging Face Gemma 2 2B IT model & tokenizer with secure HF_TOKEN support."""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token or len(hf_token.strip()) == 0:
            print("  [NOTICE] HF_TOKEN environment variable is not set.")
            print("  [AUTHENTICATION REQUIRED] 'google/gemma-2-2b-it' is a gated Hugging Face model.")
            print("  [INSTRUCTION] Set HF_TOKEN in your environment or Colab secrets (os.environ['HF_TOKEN'] = 'hf_...') to download model weights.")

        token_kwarg = {"token": hf_token} if (hf_token and len(hf_token.strip()) > 0) else {}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **token_kwarg)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            **token_kwarg
        )
        if not torch.cuda.is_available():
            self.model = self.model.to("cpu")
        self.model.eval()

    def _compute_candidate_logprobs(self, prompt: str, candidate_tokens: List[str]) -> Dict[str, float]:
        """Helper method to compute next-token log-probabilities for candidate token strings."""
        if not candidate_tokens:
            return {}
        if self.mock:
            # Deterministic synthetic candidate logprobs for mock mode
            mock_probs = {}
            for idx, cand in enumerate(candidate_tokens):
                mock_probs[cand] = round(-0.15 - (idx * 3.5), 4)
            return mock_probs

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)

        result = {}
        for cand in candidate_tokens:
            cand_ids = self.tokenizer.encode(cand, add_special_tokens=False)
            if cand_ids:
                cand_id = cand_ids[0]
                result[cand] = float(log_probs[cand_id].item())
        return result

    def run_inference(self, prompt: str, max_new_tokens: int = 10, candidate_tokens: Optional[List[str]] = None) -> Tuple[str, Optional[Dict[str, float]]]:
        """Run target model inference deterministically (temperature=0) and compute candidate logprobs if requested."""
        logprobs = self._compute_candidate_logprobs(prompt, candidate_tokens) if candidate_tokens else None
        if self.mock:
            return prompt + " [Mock Gemma Response]", logprobs

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text, logprobs

    def capture_activation(self, prompt: str, layer_idx: int, position_idx: int, candidate_tokens: Optional[List[str]] = None) -> Tuple[torch.Tensor, str, Optional[Dict[str, float]]]:
        """Capture residual-stream activation tensor output[0][:, pos:pos+1, :] at (layer_idx, position_idx)."""
        logprobs = self._compute_candidate_logprobs(prompt, candidate_tokens) if candidate_tokens else None
        if self.mock:
            dummy_tensor = torch.randn(1, 1, 2304)
            return dummy_tensor, prompt + " [Mock Gemma Response]", logprobs

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        seq_len = inputs["input_ids"].shape[1]
        pos = max(0, min(position_idx, seq_len - 1))
        
        target_layer = self.model.model.layers[layer_idx]
        captured = []

        def hook_fn(module, args, output):
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
        return captured[0], text, logprobs

    def tokenize_prompt(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Tokenize prompt and return exact token alignment map:
        List of {"position": idx, "token_id": token_id, "token_text": token_str}
        """
        if self.mock:
            tokens = prompt.split()
            return [
                {"position": idx, "token_id": 100 + idx, "token_text": tok}
                for idx, tok in enumerate(tokens)
            ]

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"][0].tolist()
        result = []
        for idx, tid in enumerate(input_ids):
            text = self.tokenizer.decode([tid])
            result.append({"position": idx, "token_id": tid, "token_text": text})
        return result

    def patch_interpolation(
        self, 
        source_prompt: str, 
        target_prompt: str, 
        layer_idx: int, 
        source_pos: int, 
        target_pos: int,
        alpha: float = 1.0,
        candidate_tokens: Optional[List[str]] = None
    ) -> Tuple[str, str, float, torch.Tensor, Optional[Dict[str, float]]]:
        """
        Interpolated activation patching: h_intervened = (1 - alpha) * h_target + alpha * h_source.
        alpha=1.0 is full patching, alpha=0.0 is baseline target.
        """
        if self.mock:
            base_out = target_prompt + " Rome."
            patch_out = target_prompt + f" Paris (alpha={alpha:.2f})." if alpha > 0 else target_prompt + " Rome."
            logprobs = self._compute_candidate_logprobs(target_prompt, candidate_tokens) if candidate_tokens else None
            if logprobs and len(candidate_tokens or []) >= 2 and alpha < 1.0:
                c0 = candidate_tokens[0]
                c1 = candidate_tokens[1]
                logprobs[c0] = round(logprobs[c0] * (1.0 - alpha * 0.5), 4)
                logprobs[c1] = round(logprobs[c1] * alpha, 4)
            return base_out, patch_out, round(35.5 * alpha, 4), torch.randn(1, 1, 2304), logprobs

        source_tensor, _, _ = self.capture_activation(source_prompt, layer_idx, source_pos)
        target_baseline, _ = self.run_inference(target_prompt)

        inputs = self.tokenizer(target_prompt, return_tensors="pt").to(self.device)
        seq_len = inputs["input_ids"].shape[1]
        t_pos = max(0, min(target_pos, seq_len - 1))
        target_layer = self.model.model.layers[layer_idx]
        delta_norms = []

        def patch_hook(module, args, output):
            if isinstance(output, tuple):
                hidden = output[0].clone()
                orig = hidden[:, t_pos:t_pos+1, :].clone()
                src = source_tensor.to(hidden.device, dtype=hidden.dtype)
                blended = (1.0 - alpha) * orig + alpha * src
                hidden[:, t_pos:t_pos+1, :] = blended
                delta_norms.append(torch.norm(hidden[:, t_pos:t_pos+1, :] - orig).item())
                return (hidden,) + output[1:]
            else:
                hidden = output.clone()
                orig = hidden[:, t_pos:t_pos+1, :].clone()
                src = source_tensor.to(hidden.device, dtype=hidden.dtype)
                blended = (1.0 - alpha) * orig + alpha * src
                hidden[:, t_pos:t_pos+1, :] = blended
                delta_norms.append(torch.norm(hidden[:, t_pos:t_pos+1, :] - orig).item())
                return hidden

        handle = target_layer.register_forward_hook(patch_hook)
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
            
            logprobs = None
            if candidate_tokens:
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
                l_probs = F.log_softmax(logits, dim=-1)
                logprobs = {}
                for cand in candidate_tokens:
                    cand_ids = self.tokenizer.encode(cand, add_special_tokens=False)
                    if cand_ids:
                        logprobs[cand] = float(l_probs[cand_ids[0]].item())
        handle.remove()

        patched_out = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        delta_norm = delta_norms[0] if delta_norms else 0.0

        return target_baseline, patched_out, delta_norm, source_tensor, logprobs

    def patch_activation(
        self, 
        source_prompt: str, 
        target_prompt: str, 
        layer_idx: int, 
        source_pos: int, 
        target_pos: int,
        candidate_tokens: Optional[List[str]] = None
    ) -> Tuple[str, str, float, torch.Tensor, Optional[Dict[str, float]]]:
        """
        Capture residual activation from source_prompt at source_pos,
        and replace residual activation in target_prompt at target_pos during forward pass.
        """
        return self.patch_interpolation(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            layer_idx=layer_idx,
            source_pos=source_pos,
            target_pos=target_pos,
            alpha=1.0,
            candidate_tokens=candidate_tokens
        )

    def ablate_activation(
        self, 
        prompt: str, 
        layer_idx: int, 
        position_idx: int,
        candidate_tokens: Optional[List[str]] = None
    ) -> Tuple[str, str, float, Optional[Dict[str, float]]]:
        """
        Zero-ablate residual stream activation at (layer_idx, position_idx) during forward pass.
        Returns (baseline_output, ablated_output, ablate_delta_norm, candidate_logprobs).
        """
        if self.mock:
            base_out = prompt + " Rome."
            ablated_out = prompt + " [Ablated Response]"
            logprobs = self._compute_candidate_logprobs(prompt, candidate_tokens) if candidate_tokens else None
            return base_out, ablated_out, 48.2, logprobs

        baseline_out, _ = self.run_inference(prompt)

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
            
            # If candidate tokens provided, run single forward pass under ablation intervention to compute logprobs
            logprobs = None
            if candidate_tokens:
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
                l_probs = F.log_softmax(logits, dim=-1)
                logprobs = {}
                for cand in candidate_tokens:
                    cand_ids = self.tokenizer.encode(cand, add_special_tokens=False)
                    if cand_ids:
                        logprobs[cand] = float(l_probs[cand_ids[0]].item())
        handle.remove()

        ablated_out = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        delta_norm = delta_norms[0] if delta_norms else 0.0

        return baseline_out, ablated_out, delta_norm, logprobs

