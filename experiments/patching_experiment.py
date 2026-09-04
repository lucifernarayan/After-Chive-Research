"""
Validation experiment orchestration for Phase 1 residual-stream activation patching.
Runs synthetic validation cases, layer sweeps, position sweeps, and generates visualization reports.
"""

import json
import os
from typing import List, Dict, Tuple, Any, Optional
from interventions.patching import ActivationPatchingEngine
from schemas.patching import PatchingExperimentResult, LayerSweepResult


class PatchingExperimentRunner:
    """
    Orchestrates synthetic validation experiments for Phase 1 causal activation patching.
    """

    def __init__(self, engine: ActivationPatchingEngine):
        self.engine = engine

    def run_synthetic_case(
        self,
        source_prompt: str,
        target_prompt: str,
        source_pos: int,
        target_pos: int,
        layer_idx: int,
        candidate_labels: Tuple[str, str]
    ) -> PatchingExperimentResult:
        """Run single 3-control synthetic patching experiment."""
        return self.engine.patch(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            layer_idx=layer_idx,
            source_position=source_pos,
            target_position=target_pos,
            candidate_labels=candidate_labels
        )

    def run_layer_sweep(
        self,
        source_prompt: str,
        target_prompt: str,
        source_pos: int,
        target_pos: int,
        candidate_labels: Tuple[str, str],
        layer_indices: Optional[List[int]] = None
    ) -> LayerSweepResult:
        """
        Sweep activation patching across early, middle, and late transformer layers.
        """
        if layer_indices is None:
            # Gemma 2 2B IT has 26 layers (0 to 25)
            # Sample early (2, 6), middle (10, 14, 18), late (22, 25)
            layer_indices = [2, 6, 10, 14, 18, 22, 25]

        results_by_layer: Dict[int, PatchingExperimentResult] = {}
        best_layer = None
        max_movement = -999.0

        print(f"\n--- LAYER SWEEP: Testing Layers {layer_indices} ---")
        for layer in layer_indices:
            res = self.run_synthetic_case(
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                source_pos=source_pos,
                target_pos=target_pos,
                layer_idx=layer,
                candidate_labels=candidate_labels
            )
            results_by_layer[layer] = res
            
            movement = res.behavioral_movement_toward_source
            rand_mov = res.random_movement_toward_source
            spec = "SPECIFIC" if res.is_specific_causal_effect else "NON-SPECIFIC"

            print(f"  Layer {layer:2d} | Patch Delta Norm: {res.patch_delta_norm:6.2f} | "
                  f"Movement: {movement:+6.2f} | Rand Mov: {rand_mov:+6.2f} | Effect: {spec}")

            if res.is_specific_causal_effect and movement > max_movement:
                max_movement = movement
                best_layer = layer

        ascii_plot = self.generate_ascii_plot(results_by_layer)

        return LayerSweepResult(
            model_name="google/gemma-2-2b-it",
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            target_layer_indices=layer_indices,
            results_by_layer=results_by_layer,
            best_causal_layer=best_layer,
            summary_plot_ascii=ascii_plot
        )

    def generate_ascii_plot(self, results_by_layer: Dict[int, PatchingExperimentResult]) -> str:
        """Generate ASCII bar chart showing Layer -> Patch Effect Size."""
        lines = [
            "\n=========================================================",
            "  LAYER -> CAUSAL INTERVENTION EFFECT SIZE (ASCII PLOT)",
            "========================================================="
        ]
        
        for layer, res in results_by_layer.items():
            mov = max(0.0, res.behavioral_movement_toward_source)
            bar_len = int(mov * 5) # Scale bar
            bar = "#" * bar_len
            rand_mov = max(0.0, res.random_movement_toward_source)
            rand_bar = "." * int(rand_mov * 5)
            flag = "[*SPECIFIC*]" if res.is_specific_causal_effect else "[DISRUPTIVE]"
            lines.append(f"  Layer {layer:2d} | Patch: {bar:<25} ({mov:+.2f}) {flag}")
            lines.append(f"           | Rand : {rand_bar:<25} ({rand_mov:+.2f})")
            lines.append("  " + "-" * 55)

        return "\n".join(lines)
