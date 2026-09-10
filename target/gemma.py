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
            if logprobs and len(candidate_tokens or []) >= 2 and 0.0 < alpha < 1.0:
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

    def attribution_localization(self, prompt: str, candidate_tokens: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Compute gradient/activation attribution localization scores across layers and positions.
        Returns List of {"layer_idx": L, "position_idx": P, "token_text": T, "attribution_score": S}.
        """
        tokens = self.tokenize_prompt(prompt)
        scores = []
        if self.mock:
            for layer in [4, 8, 12, 16, 20]:
                for t in tokens:
                    pos = t["position"]
                    score = round(0.1 + (layer * 0.03) + (pos * 0.05), 4)
                    scores.append({
                        "layer_idx": layer,
                        "position_idx": pos,
                        "token_text": t["token_text"],
                        "attribution_score": score
                    })
            scores.sort(key=lambda x: x["attribution_score"], reverse=True)
            return scores

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        seq_len = inputs["input_ids"].shape[1]
        
        for layer_idx in range(0, min(26, len(self.model.model.layers)), 4):
            layer = self.model.model.layers[layer_idx]
            act_list = []
            def h_fn(m, a, o):
                val = o[0] if isinstance(o, tuple) else o
                act_list.append(val.detach())
            h = layer.register_forward_hook(h_fn)
            with torch.no_grad():
                self.model(**inputs)
            h.remove()
            if act_list:
                act = act_list[0][0]
                for p in range(seq_len):
                    tok_text = tokens[p]["token_text"] if p < len(tokens) else f"pos_{p}"
                    norm_val = float(torch.norm(act[p]).item())
                    scores.append({
                        "layer_idx": layer_idx,
                        "position_idx": p,
                        "token_text": tok_text,
                        "attribution_score": round(norm_val, 4)
                    })
        scores.sort(key=lambda x: x["attribution_score"], reverse=True)
        return scores

    def patch_head(
        self,
        source_prompt: str,
        target_prompt: str,
        layer_idx: int,
        head_idx: int,
        source_pos: int,
        target_pos: int,
        candidate_tokens: Optional[List[str]] = None
    ) -> Tuple[str, str, float, Optional[Dict[str, float]]]:
        """
        Patch specific attention head output slice at (layer_idx, head_idx, target_pos) from source_prompt.
        """
        if self.mock:
            base_out = target_prompt + " Rome."
            patch_out = target_prompt + f" [Head {head_idx} Patched Response]"
            logprobs = self._compute_candidate_logprobs(target_prompt, candidate_tokens) if candidate_tokens else None
            return base_out, patch_out, 12.4, logprobs

        target_baseline, _ = self.run_inference(target_prompt)
        inputs_src = self.tokenizer(source_prompt, return_tensors="pt").to(self.device)
        inputs_tgt = self.tokenizer(target_prompt, return_tensors="pt").to(self.device)

        target_layer = self.model.model.layers[layer_idx]
        attn_module = getattr(target_layer, "self_attn", target_layer)

        src_head_act = []
        def capture_head_hook(module, args, output):
            attn_out = output[0] if isinstance(output, tuple) else output
            hidden_dim = attn_out.shape[-1]
            num_heads = getattr(module, "num_heads", getattr(module, "num_attention_heads", 8))
            head_dim = hidden_dim // num_heads
            s_pos = max(0, min(source_pos, attn_out.shape[1] - 1))
            h_start = head_idx * head_dim
            h_end = (head_idx + 1) * head_dim
            src_head_act.append(attn_out[:, s_pos:s_pos+1, h_start:h_end].detach().clone())

        handle_src = attn_module.register_forward_hook(capture_head_hook)
        with torch.no_grad():
            self.model(**inputs_src)
        handle_src.remove()

        if not src_head_act:
            return target_baseline, target_baseline, 0.0, None

        src_tensor = src_head_act[0]
        delta_norms = []
        t_pos = max(0, min(target_pos, inputs_tgt["input_ids"].shape[1] - 1))

        def patch_head_hook(module, args, output):
            attn_out = output[0] if isinstance(output, tuple) else output
            attn_out_clone = attn_out.clone()
            hidden_dim = attn_out_clone.shape[-1]
            num_heads = getattr(module, "num_heads", getattr(module, "num_attention_heads", 8))
            head_dim = hidden_dim // num_heads
            h_start = head_idx * head_dim
            h_end = (head_idx + 1) * head_dim
            orig = attn_out_clone[:, t_pos:t_pos+1, h_start:h_end].clone()
            src = src_tensor.to(attn_out_clone.device, dtype=attn_out_clone.dtype)
            attn_out_clone[:, t_pos:t_pos+1, h_start:h_end] = src
            delta_norms.append(torch.norm(src - orig).item())
            if isinstance(output, tuple):
                return (attn_out_clone,) + output[1:]
            return attn_out_clone

        handle_tgt = attn_module.register_forward_hook(patch_head_hook)
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs_tgt, max_new_tokens=10, do_sample=False)
            logprobs = None
            if candidate_tokens:
                outputs = self.model(**inputs_tgt)
                logits = outputs.logits[0, -1, :]
                l_probs = F.log_softmax(logits, dim=-1)
                logprobs = {}
                for cand in candidate_tokens:
                    c_ids = self.tokenizer.encode(cand, add_special_tokens=False)
                    if c_ids:
                        logprobs[cand] = float(l_probs[c_ids[0]].item())
        handle_tgt.remove()

        patched_out = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        delta_norm = delta_norms[0] if delta_norms else 0.0
        return target_baseline, patched_out, delta_norm, logprobs

    def ablate_head(
        self,
        prompt: str,
        layer_idx: int,
        head_idx: int,
        position_idx: int,
        candidate_tokens: Optional[List[str]] = None
    ) -> Tuple[str, str, float, Optional[Dict[str, float]]]:
        """
        Zero-ablate specific attention head output slice at (layer_idx, head_idx, position_idx).
        """
        if self.mock:
            base_out = prompt + " Rome."
            ablated_out = prompt + f" [Head {head_idx} Ablated Response]"
            logprobs = self._compute_candidate_logprobs(prompt, candidate_tokens) if candidate_tokens else None
            return base_out, ablated_out, 15.6, logprobs

        target_baseline, _ = self.run_inference(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        target_layer = self.model.model.layers[layer_idx]
        attn_module = getattr(target_layer, "self_attn", target_layer)
        delta_norms = []
        pos = max(0, min(position_idx, inputs["input_ids"].shape[1] - 1))

        def ablate_head_hook(module, args, output):
            attn_out = output[0] if isinstance(output, tuple) else output
            attn_out_clone = attn_out.clone()
            hidden_dim = attn_out_clone.shape[-1]
            num_heads = getattr(module, "num_heads", getattr(module, "num_attention_heads", 8))
            head_dim = hidden_dim // num_heads
            h_start = head_idx * head_dim
            h_end = (head_idx + 1) * head_dim
            orig = attn_out_clone[:, pos:pos+1, h_start:h_end].clone()
            attn_out_clone[:, pos:pos+1, h_start:h_end] = 0.0
            delta_norms.append(torch.norm(orig).item())
            if isinstance(output, tuple):
                return (attn_out_clone,) + output[1:]
            return attn_out_clone

        handle = attn_module.register_forward_hook(ablate_head_hook)
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
            logprobs = None
            if candidate_tokens:
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
                l_probs = F.log_softmax(logits, dim=-1)
                logprobs = {}
                for cand in candidate_tokens:
                    c_ids = self.tokenizer.encode(cand, add_special_tokens=False)
                    if c_ids:
                        logprobs[cand] = float(l_probs[c_ids[0]].item())
        handle.remove()

        ablated_out = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        delta_norm = delta_norms[0] if delta_norms else 0.0
        return target_baseline, ablated_out, delta_norm, logprobs

    def patch_mlp(
        self,
        source_prompt: str,
        target_prompt: str,
        layer_idx: int,
        source_pos: int,
        target_pos: int,
        candidate_tokens: Optional[List[str]] = None
    ) -> Tuple[str, str, float, Optional[Dict[str, float]]]:
        """
        Patch MLP output tensor at (layer_idx, target_pos) from source_prompt.
        """
        if self.mock:
            base_out = target_prompt + " Rome."
            patch_out = target_prompt + " [MLP Patched Response]"
            logprobs = self._compute_candidate_logprobs(target_prompt, candidate_tokens) if candidate_tokens else None
            return base_out, patch_out, 18.2, logprobs

        target_baseline, _ = self.run_inference(target_prompt)
        inputs_src = self.tokenizer(source_prompt, return_tensors="pt").to(self.device)
        inputs_tgt = self.tokenizer(target_prompt, return_tensors="pt").to(self.device)

        target_layer = self.model.model.layers[layer_idx]
        mlp_module = getattr(target_layer, "mlp", target_layer)

        src_mlp_act = []
        def capture_mlp_hook(module, args, output):
            mlp_out = output[0] if isinstance(output, tuple) else output
            s_pos = max(0, min(source_pos, mlp_out.shape[1] - 1))
            src_mlp_act.append(mlp_out[:, s_pos:s_pos+1, :].detach().clone())

        handle_src = mlp_module.register_forward_hook(capture_mlp_hook)
        with torch.no_grad():
            self.model(**inputs_src)
        handle_src.remove()

        if not src_mlp_act:
            return target_baseline, target_baseline, 0.0, None

        src_tensor = src_mlp_act[0]
        delta_norms = []
        t_pos = max(0, min(target_pos, inputs_tgt["input_ids"].shape[1] - 1))

        def patch_mlp_hook(module, args, output):
            mlp_out = output[0] if isinstance(output, tuple) else output
            mlp_out_clone = mlp_out.clone()
            orig = mlp_out_clone[:, t_pos:t_pos+1, :].clone()
            src = src_tensor.to(mlp_out_clone.device, dtype=mlp_out_clone.dtype)
            mlp_out_clone[:, t_pos:t_pos+1, :] = src
            delta_norms.append(torch.norm(src - orig).item())
            if isinstance(output, tuple):
                return (mlp_out_clone,) + output[1:]
            return mlp_out_clone

        handle_tgt = mlp_module.register_forward_hook(patch_mlp_hook)
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs_tgt, max_new_tokens=10, do_sample=False)
            logprobs = None
            if candidate_tokens:
                outputs = self.model(**inputs_tgt)
                logits = outputs.logits[0, -1, :]
                l_probs = F.log_softmax(logits, dim=-1)
                logprobs = {}
                for cand in candidate_tokens:
                    c_ids = self.tokenizer.encode(cand, add_special_tokens=False)
                    if c_ids:
                        logprobs[cand] = float(l_probs[c_ids[0]].item())
        handle_tgt.remove()

        patched_out = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        delta_norm = delta_norms[0] if delta_norms else 0.0
        return target_baseline, patched_out, delta_norm, logprobs

    def ablate_mlp(
        self,
        prompt: str,
        layer_idx: int,
        position_idx: int,
        candidate_tokens: Optional[List[str]] = None
    ) -> Tuple[str, str, float, Optional[Dict[str, float]]]:
        """
        Zero-ablate MLP output tensor at (layer_idx, position_idx).
        """
        if self.mock:
            base_out = prompt + " Rome."
            ablated_out = prompt + " [MLP Ablated Response]"
            logprobs = self._compute_candidate_logprobs(prompt, candidate_tokens) if candidate_tokens else None
            return base_out, ablated_out, 22.4, logprobs

        target_baseline, _ = self.run_inference(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        target_layer = self.model.model.layers[layer_idx]
        mlp_module = getattr(target_layer, "mlp", target_layer)
        delta_norms = []
        pos = max(0, min(position_idx, inputs["input_ids"].shape[1] - 1))

        def ablate_mlp_hook(module, args, output):
            mlp_out = output[0] if isinstance(output, tuple) else output
            mlp_out_clone = mlp_out.clone()
            orig = mlp_out_clone[:, pos:pos+1, :].clone()
            mlp_out_clone[:, pos:pos+1, :] = 0.0
            delta_norms.append(torch.norm(orig).item())
            if isinstance(output, tuple):
                return (mlp_out_clone,) + output[1:]
            return mlp_out_clone

        handle = mlp_module.register_forward_hook(ablate_mlp_hook)
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
            logprobs = None
            if candidate_tokens:
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
                l_probs = F.log_softmax(logits, dim=-1)
                logprobs = {}
                for cand in candidate_tokens:
                    c_ids = self.tokenizer.encode(cand, add_special_tokens=False)
                    if c_ids:
                        logprobs[cand] = float(l_probs[c_ids[0]].item())
        handle.remove()

        ablated_out = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        delta_norm = delta_norms[0] if delta_norms else 0.0
        return target_baseline, ablated_out, delta_norm, logprobs


