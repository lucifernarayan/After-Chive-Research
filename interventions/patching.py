"""
Core Causal Residual-Stream Activation Patching Engine.
Uses native PyTorch forward hooks to capture and replace hidden state tensors in Gemma 2 2B IT.
"""

import uuid
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from schemas.patching import (
    TokenAlignment,
    BehavioralMeasurement,
    PatchingExperimentResult
)


class ActivationPatchingEngine:
    """
    Executes causal residual-stream activation patching on Gemma 2 2B IT.
    
    Hook Mechanism:
    - Target layer: model.model.layers[layer_idx]
    - Layer output tuple: (hidden_states, attn_weights, ...)
    - Residual stream tensor shape: [batch_size, seq_len, hidden_dim]
    """

    def __init__(self, model: Any, tokenizer: Any, device: str = "cpu", mock: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.mock = mock

    def get_token_alignment(self, prompt: str, position_idx: int) -> TokenAlignment:
        """Extract explicit token position indexing for a prompt."""
        if self.mock:
            tokens = prompt.split()
            pos = min(position_idx, len(tokens) - 1)
            return TokenAlignment(
                position_index=pos,
                token_id=100 + pos,
                token_str=tokens[pos] if pos < len(tokens) else "<pad>",
                sequence_length=len(tokens)
            )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        seq_len = len(input_ids)
        pos = position_idx if position_idx >= 0 else seq_len + position_idx
        pos = max(0, min(pos, seq_len - 1))
        
        token_id = input_ids[pos].item()
        token_str = self.tokenizer.decode([token_id])

        return TokenAlignment(
            position_index=pos,
            token_id=token_id,
            token_str=token_str,
            sequence_length=seq_len
        )

    def capture_activation(
        self, 
        prompt: str, 
        layer_idx: int, 
        position_idx: int
    ) -> Tuple[torch.Tensor, TokenAlignment, BehavioralMeasurement]:
        """
        Run forward pass on prompt and capture residual stream tensor at (layer_idx, position_idx).
        """
        token_alignment = self.get_token_alignment(prompt, position_idx)
        
        if self.mock:
            # Mock 2304-dim Gemma 2 activation vector
            hidden_dim = 2304
            captured_tensor = torch.randn(1, 1, hidden_dim)
            behavior = BehavioralMeasurement(
                generated_text=prompt + " Paris.",
                new_tokens_text=" Paris.",
                candidate_logprobs={"Paris": -0.15, "Rome": -7.20},
                top_predicted_token=" Paris"
            )
            return captured_tensor, token_alignment, behavior

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        pos = token_alignment.position_index
        captured_activation = []

        target_layer = self.model.model.layers[layer_idx]

        def capture_hook(module, args, output):
            if isinstance(output, tuple):
                # Shape: [batch, seq_len, hidden_dim] -> slice target position
                act = output[0][:, pos:pos+1, :].detach().clone()
            else:
                act = output[:, pos:pos+1, :].detach().clone()
            captured_activation.append(act)

        hook_handle = target_layer.register_forward_hook(capture_hook)

        with torch.no_grad():
            outputs = self.model(**inputs)
            gen_output = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )

        hook_handle.remove()

        captured_tensor = captured_activation[0]
        gen_text = self.tokenizer.decode(gen_output[0], skip_special_tokens=True)
        new_tokens = gen_text[len(prompt):]

        # Calculate next token candidate logprobs
        logits = outputs.logits[0, -1, :] # Last position logits
        logprobs = F.log_softmax(logits, dim=-1)

        behavior = BehavioralMeasurement(
            generated_text=gen_text,
            new_tokens_text=new_tokens,
            candidate_logprobs={},
            top_predicted_token=self.tokenizer.decode([torch.argmax(logits).item()])
        )

        return captured_tensor, token_alignment, behavior

    def run_patched_inference(
        self,
        prompt: str,
        layer_idx: int,
        position_idx: int,
        patch_tensor: torch.Tensor,
        candidate_tokens: Optional[List[str]] = None
    ) -> Tuple[BehavioralMeasurement, float]:
        """
        Run forward pass on prompt replacing residual activation at (layer_idx, position_idx) with patch_tensor.
        Returns behavioral measurement and delta norm (L2 norm of target activation - patch activation).
        """
        token_alignment = self.get_token_alignment(prompt, position_idx)
        pos = token_alignment.position_index

        if self.mock:
            behavior = BehavioralMeasurement(
                generated_text=prompt + (" Paris." if patch_tensor is not None else " Rome."),
                new_tokens_text=" Paris." if patch_tensor is not None else " Rome.",
                candidate_logprobs={"Paris": -0.45, "Rome": -2.10},
                top_predicted_token=" Paris" if patch_tensor is not None else " Rome"
            )
            return behavior, 35.2

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        target_layer = self.model.model.layers[layer_idx]
        delta_norms = []

        def patch_hook(module, args, output):
            if isinstance(output, tuple):
                hidden = output[0].clone()
                original_act = hidden[:, pos:pos+1, :].clone()
                hidden[:, pos:pos+1, :] = patch_tensor.to(hidden.device, dtype=hidden.dtype)
                delta = torch.norm(hidden[:, pos:pos+1, :] - original_act).item()
                delta_norms.append(delta)
                return (hidden,) + output[1:]
            else:
                hidden = output.clone()
                original_act = hidden[:, pos:pos+1, :].clone()
                hidden[:, pos:pos+1, :] = patch_tensor.to(hidden.device, dtype=hidden.dtype)
                delta = torch.norm(hidden[:, pos:pos+1, :] - original_act).item()
                delta_norms.append(delta)
                return hidden

        hook_handle = target_layer.register_forward_hook(patch_hook)

        with torch.no_grad():
            outputs = self.model(**inputs)
            gen_output = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )

        hook_handle.remove()

        gen_text = self.tokenizer.decode(gen_output[0], skip_special_tokens=True)
        new_tokens = gen_text[len(prompt):]
        logits = outputs.logits[0, -1, :]
        logprobs = F.log_softmax(logits, dim=-1)

        candidate_probs = {}
        if candidate_tokens:
            for cand in candidate_tokens:
                cand_ids = self.tokenizer.encode(cand, add_special_tokens=False)
                if len(cand_ids) > 0:
                    cand_id = cand_ids[0]
                    candidate_probs[cand] = logprobs[cand_id].item()

        behavior = BehavioralMeasurement(
            generated_text=gen_text,
            new_tokens_text=new_tokens,
            candidate_logprobs=candidate_probs,
            top_predicted_token=self.tokenizer.decode([torch.argmax(logits).item()])
        )

        patch_delta = delta_norms[0] if len(delta_norms) > 0 else 0.0
        return behavior, patch_delta

    def patch(
        self,
        source_prompt: str,
        target_prompt: str,
        layer_idx: int,
        source_position: int,
        target_position: int,
        candidate_labels: Optional[Tuple[str, str]] = None # (source_label, target_label)
    ) -> PatchingExperimentResult:
        """
        Execute complete 3-condition patching experiment:
        - Baseline Source A
        - Control 1: Baseline Target B
        - Control 2: Patch A -> B
        - Control 3: Random Norm-Matched Control -> B
        """
        exp_id = f"exp_{uuid.uuid4().hex[:8]}"

        # 1. Capture Source Activation from Prompt A
        source_tensor, source_token, source_behavior = self.capture_activation(
            source_prompt, layer_idx, source_position
        )

        # 2. Run Baseline Target B (Control 1 - No Intervention)
        target_token = self.get_token_alignment(target_prompt, target_position)
        
        cands = list(candidate_labels) if candidate_labels else None

        if self.mock:
            target_baseline_behavior = BehavioralMeasurement(
                generated_text=target_prompt + " Rome.",
                new_tokens_text=" Rome.",
                candidate_logprobs={"Paris": -6.50, "Rome": -0.20},
                top_predicted_token=" Rome"
            )
            patched_behavior = BehavioralMeasurement(
                generated_text=target_prompt + " Paris.",
                new_tokens_text=" Paris.",
                candidate_logprobs={"Paris": -0.50, "Rome": -3.10},
                top_predicted_token=" Paris"
            )
            random_control_behavior = BehavioralMeasurement(
                generated_text=target_prompt + " Rome.",
                new_tokens_text=" Rome.",
                candidate_logprobs={"Paris": -6.40, "Rome": -0.25},
                top_predicted_token=" Rome"
            )
            patch_delta_norm = 42.5
            random_delta_norm = 42.1
            act_l2_norm = torch.norm(source_tensor).item() if isinstance(source_tensor, torch.Tensor) else 30.0
        else:
            # Baseline Target B (No intervention)
            dummy_zero = torch.zeros_like(source_tensor)
            _, _ = self.run_patched_inference(
                target_prompt, layer_idx, target_token.position_index, dummy_zero, candidate_tokens=cands
            )
            target_baseline_behavior, _ = self.capture_activation(target_prompt, layer_idx, target_position)[2], 0.0

            # Control 2: Patch A -> B
            patched_behavior, patch_delta_norm = self.run_patched_inference(
                target_prompt, layer_idx, target_token.position_index, source_tensor, candidate_tokens=cands
            )

            # Control 3: Random Norm-Matched Control
            act_l2_norm = torch.norm(source_tensor).item()
            random_tensor = torch.randn_like(source_tensor)
            random_tensor = (random_tensor / (torch.norm(random_tensor) + 1e-8)) * act_l2_norm
            random_control_behavior, random_delta_norm = self.run_patched_inference(
                target_prompt, layer_idx, target_token.position_index, random_tensor, candidate_tokens=cands
            )

        # 3. Calculate Behavioral Movement Toward Source
        source_label = candidate_labels[0] if candidate_labels else None
        
        if source_label and patched_behavior.candidate_logprobs and target_baseline_behavior.candidate_logprobs:
            p_patched = patched_behavior.candidate_logprobs.get(source_label, -10.0)
            p_base = target_baseline_behavior.candidate_logprobs.get(source_label, -10.0)
            p_rand = random_control_behavior.candidate_logprobs.get(source_label, -10.0)
            
            movement = p_patched - p_base
            rand_movement = p_rand - p_base
        else:
            # Fallback text string movement heuristic
            movement = 1.0 if (source_label and source_label.lower() in patched_behavior.new_tokens_text.lower()) else 0.0
            rand_movement = 1.0 if (source_label and source_label.lower() in random_control_behavior.new_tokens_text.lower()) else 0.0

        # Specificity check: patch shifts behavior toward source AND random control does NOT
        is_specific = (movement > 0.3) and (rand_movement < 0.3 * movement)

        return PatchingExperimentResult(
            experiment_id=exp_id,
            model_name="google/gemma-2-2b-it",
            layer=layer_idx,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            source_token=source_token,
            target_token=target_token,
            source_behavior=source_behavior,
            target_baseline_behavior=target_baseline_behavior,
            patched_behavior=patched_behavior,
            random_control_behavior=random_control_behavior,
            activation_shape=list(source_tensor.shape) if isinstance(source_tensor, torch.Tensor) else [1, 1, 2304],
            activation_l2_norm=act_l2_norm,
            patch_delta_norm=patch_delta_norm,
            random_delta_norm=random_delta_norm,
            behavioral_movement_toward_source=round(float(movement), 4),
            random_movement_toward_source=round(float(rand_movement), 4),
            is_specific_causal_effect=is_specific,
            execution_success=True,
            metadata={
                "torch_version": torch.__version__,
                "device": self.device,
                "mock": self.mock
            }
        )
