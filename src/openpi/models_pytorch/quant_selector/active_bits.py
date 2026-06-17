from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_A_BITS = {2, 4, 8, 16}

ACTION_EXTRA_LAYER_NAMES = {
    "action_in_proj",
    "time_mlp_in",
    "time_mlp_out",
    "state_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
}


@dataclass
class ModulewiseABitsController:
    model: Any
    warn_unclassified: bool = True

    vlm_layers: list[tuple[str, Any]] = field(default_factory=list)
    action_layers: list[tuple[str, Any]] = field(default_factory=list)
    other_layers: list[tuple[str, Any]] = field(default_factory=list)

    current_vlm_a_bits: int | None = None
    current_action_a_bits: int | None = None
    built: bool = False

    def build(self) -> None:
        if self.built:
            return

        vlm_layers: list[tuple[str, Any]] = []
        action_layers: list[tuple[str, Any]] = []
        other_layers: list[tuple[str, Any]] = []

        for name, module in self.model.named_modules():
            if not hasattr(module, "set_a_bits"):
                continue

            if name.startswith("paligemma_with_expert.paligemma."):
                vlm_layers.append((name, module))

            elif (
                name.startswith("paligemma_with_expert.gemma_expert.")
                or name in ACTION_EXTRA_LAYER_NAMES
            ):
                action_layers.append((name, module))

            else:
                other_layers.append((name, module))

        self.vlm_layers = vlm_layers
        self.action_layers = action_layers
        self.other_layers = other_layers
        self.built = True

        if self.warn_unclassified and other_layers:
            print("[MODULEWISE-ABITS][WARN] unclassified quant layers:", flush=True)
            for name, module in other_layers[:20]:
                print(
                    f"  name={name} class={module.__class__.__name__} "
                    f"A={getattr(module, 'act_bits', None)}",
                    flush=True,
                )

    @staticmethod
    def _check_bits(name: str, value: int) -> int:
        value = int(value)
        if value not in VALID_A_BITS:
            raise ValueError(
                f"Invalid {name}={value}, expected {sorted(VALID_A_BITS)}"
            )
        return value

    def set(
        self,
        *,
        vlm_a_bits: int | None = None,
        action_a_bits: int | None = None,
    ) -> tuple[int | None, int | None]:
        if vlm_a_bits is None and action_a_bits is None:
            return self.current_vlm_a_bits, self.current_action_a_bits

        self.build()

        if vlm_a_bits is not None:
            vlm_a_bits = self._check_bits("vlm_a_bits", vlm_a_bits)

            if self.current_vlm_a_bits != vlm_a_bits:
                for _, module in self.vlm_layers:
                    module.set_a_bits(vlm_a_bits)
                self.current_vlm_a_bits = vlm_a_bits

        if action_a_bits is not None:
            action_a_bits = self._check_bits("action_a_bits", action_a_bits)

            if self.current_action_a_bits != action_a_bits:
                for _, module in self.action_layers:
                    module.set_a_bits(action_a_bits)
                self.current_action_a_bits = action_a_bits

        return self.current_vlm_a_bits, self.current_action_a_bits