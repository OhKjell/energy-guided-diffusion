from typing import Any, Optional, Tuple

import numpy as np
import torch
from mei.domain import State
from mei.stoppers import OptimizationStopper


class ActivationIncrease(OptimizationStopper):
    """Callable optimization stopper as used by Golan et al. 2020 monitoring the maximum change of pixel intensities."""

    def __init__(
        self,
        initial_activation: float,
        patience: int = 300,
        verbose: bool = True,
        minimal_change: float = 0.0001,
        max_steps: int = 50000,
        negative_steps_tolerance=100,
    ) -> None:
        """[summary]
        Args:
            initial_img (torch.Tensor): [description]
            patience (int, optional): [description]. Defaults to 10.
            verbose (bool, optional): [description]. Defaults to False.
        """
        self.steps_wo_change = 0
        self.negative_change_steps = 0
        self.patience = patience
        self.negative_patience = negative_steps_tolerance

        self.activation = initial_activation
        self.minimal_change = minimal_change
        self.verbose = verbose
        self.max_steps = max_steps

    def __call__(self, current_state: State) -> Tuple[bool, Optional[Any]]:
        """Returns true if stopping criterion is met
        Returns:
            bool: True if stopping criterion met, else: False
        """
        current_activation = current_state.evaluation
        change = current_activation - self.activation
        abs_change = abs(change)
        self.abs_change = abs_change
        if self.verbose:
            print(
                "abs. change",
                np.round(abs_change, 8),
                "minimal change required:",
                self.minimal_change * np.abs(self.activation),
            )
            print(f"steps w/o change: {self.steps_wo_change}")
            print(f"negative steps: {self.negative_change_steps}")

        if abs_change < (self.minimal_change * np.abs(self.activation)):
            self.steps_wo_change += 1
            if self.verbose:
                print("no change")
            if self.steps_wo_change > self.patience:
                print("STOP: Activation not changing")
                return True, "No significant activation change"
        else:
            self.steps_wo_change = 0

        # if change < 0:
        #     self.negative_change_steps += 1
        #     if self.negative_change_steps >= self.negative_patience:
        #         print('STOP: Max negative steps exceeded')
        #         return True, None

        self.activation = current_activation
        if current_state.i_iter > self.max_steps:
            print("STOP: Max steps exceeded")
            return True, "Exceeded max steps"
        return False, None
