import os
import pickle
from pathlib import Path
from typing import Any

import matplotlib.colors
import numpy as np
from matplotlib import animation, cm
from matplotlib import pyplot as plt
from mei.domain import State
from mei.objectives import RegularIntervalObjective
from mei.tracking import Tracker

import wandb
from meis.visualizer import save_mei_video


class LoggingTracker(Tracker):
    def __init__(
        self,
        log_dir,
        seed,
        wandb_log=False,
        log_frequency=2,
        video_log_frequency=20,
        **objectives,
    ):
        self.log_dir = os.path.join(log_dir)
        self.seed = seed
        self.wandb_log = wandb_log
        self.log_frequency = log_frequency
        self.video_log_frequency = video_log_frequency
        super().__init__(**objectives)
        if self.wandb_log:
            Path(
                os.path.join(
                    os.path.join(self.log_dir, "videos"),
                )
            ).mkdir(parents=True, exist_ok=True)
        # for name, objective in objectives.items():

    def save_tracks(self, current_state, end=False):
        print(f"saving tracked logs to {self.log_dir}")
        self.track(current_state, end=end)
        for name, objective in self.objectives.items():
            Path(os.path.join(self.log_dir, name)).mkdir(exist_ok=True, parents=True)
            with open(os.path.join(self.log_dir, name, "logs.pkl"), "wb") as f:
                pickle.dump(self.log[name], f)

    def show_mei(self, current_state):
        for time in range(current_state.shape[2]):
            plt.imshow(np.array(current_state[0, 0, time]))
            plt.title(f"MEI t {time}")
            plt.show()

    def track(self, current_state, end=False):
        super().track(current_state=current_state)
        if self.wandb_log and ((current_state.i_iter % self.log_frequency == 0) or end):
            wandb.log(
                {"activation": self.log["activation"]["values"][-1]},
                step=current_state.i_iter,
            )
            wandb.log({'max_grad_value': np.max(np.abs(np.array(current_state.grad.detach().cpu().numpy())))},
                      step=current_state.i_iter)
        if self.wandb_log and ((current_state.i_iter % self.video_log_frequency == 0) or end):
            wandb.log(
                {
                    "max_state_value": np.max(
                        np.abs(np.array(self.log["state"]["values"][-1]))
                    )
                },
                step=current_state.i_iter,
            )

            # wandb.log({'learning_rate': current_state.optimizer.param_groups[0]['lr']}, step=current_state.i_iteration)
            # wandb.log(
            #     {"video": wandb.Video(np.array(self.log['state']['values'][-1][0]).transpose((1,0,2,3)), fps=4, format="mp4")})

        # if self.wandb_log and ((current_state.i_iter % self.video_log_frequency == 0) or end):
            save_mei_video(
                current_state.i_iter,
                current_state=np.array(self.log["state"]["values"][-1]),
                log_dir=self.log_dir,
                colormap="gray",
            )
            #
            save_mei_video(
                current_state.i_iter,
                current_state=np.array(self.log["p_grad"]["values"][-1]),
                log_dir=self.log_dir,
                colormap="gray",
                vmin=None,
                vmax=None,
                prefix="p_grad",
            )
            wandb.log(
                {
                    "video": wandb.Video(
                        os.path.join(
                            self.log_dir, "videos", f"mei_e{current_state.i_iter}.mp4"
                        ),
                        format="mp4",
                    )
                },
                step=current_state.i_iter,
            )
            wandb.log(
                {
                    "p_grad": wandb.Video(
                        os.path.join(
                            self.log_dir,
                            "videos",
                            f"p_grad_e{current_state.i_iter}.mp4",
                        ),
                        format="mp4",
                    )
                },
                step=current_state.i_iter,
            )

        print(
            f"activation at {self.log['activation']['times'][-1]}: {self.log['activation']['values'][-1]}"
        )


class SuppressiveLoggingTracker(LoggingTracker):
    def __init__(
        self, log_dir, seed, wandb_log=False, log_frequency=2, video_log_frequency=10, mei=None, **objectives
    ):
        super(SuppressiveLoggingTracker, self).__init__(
            log_dir=log_dir,
            seed=seed,
            wandb_log=wandb_log,
            log_frequency=log_frequency,
            video_log_frequency=video_log_frequency,
            **objectives,
        )
        self.mei = mei
        self.video_log_frequency = video_log_frequency
        Path(
            os.path.join(
                os.path.join(self.log_dir, "videos"),
            )
        ).mkdir(parents=True, exist_ok=True)

    def track(self, current_state, end=False):
        super().track(current_state=current_state, end=end)
        surround = (1 - self.mei.envelope) * np.array(self.log["state"]["values"][-1])
        whole_input = self.mei.exciting_mei + surround
        if ((current_state.i_iter % self.video_log_frequency) == 0) or end:
            wandb.log({
                        "max_complete_mei_value": np.max(
                            np.abs(np.array(whole_input.clone().detach().cpu().numpy()))
                        )
                    },
                    step=current_state.i_iter,)
            wandb.log({
                "max_masked_surround_value": np.max(
                    np.abs(surround)
                )
            },
            step=current_state.i_iter, )
        if ((current_state.i_iter % self.video_log_frequency) == 0) or end:
            save_mei_video(
                current_state.i_iter,
                current_state=whole_input,
                log_dir=self.log_dir,
                colormap="gray",
                vmin=None,
                vmax=None,
                prefix="complete_mei",
            )
            save_mei_video(
                current_state.i_iter,
                current_state=surround,
                log_dir=self.log_dir,
                colormap="gray",
                vmin=-1,
                vmax=1,
                prefix="masked_surround",
            )

            wandb.log(
                {
                    "complete_mei": wandb.Video(
                        os.path.join(
                            self.log_dir,
                            "videos",
                            f"complete_mei_e{current_state.i_iter}.mp4",
                        ),
                        format="mp4",
                    )
                },
                step=current_state.i_iter,
            )
            wandb.log(
                {
                    "masked_surround": wandb.Video(
                        os.path.join(
                            self.log_dir, "videos", f"masked_surround_e{current_state.i_iter}.mp4"
                        ),
                        format="mp4",
                    )
                },
                step=current_state.i_iter,
            )


class GradientObjective(RegularIntervalObjective):
    """Objective used to track the gradients during the optimization process."""

    def compute(self, current_state: State) -> Any:
        return current_state.preconditioned_grad


class MaskedSurroundObjective(RegularIntervalObjective):
    """Objective used to track the mei + suppressive surround during the optimization process."""
    def __init__(self, interval: int, mei):
        super().__init__(interval)
        self.envelope = mei.envelope

    def compute(self, current_state: State) -> Any:
        return current_state.post_processed_input*(1-self.envelope)


class WholeMeiObjective(RegularIntervalObjective):
    """Objective used to track the mei + suppressive surround during the optimization process."""
    def __init__(self, interval: int, mei):
        super().__init__(interval)
        self.envelope = mei.envelope
        self.exciting_mei = mei.exciting_mei

    def compute(self, current_state: State) -> Any:
        return current_state.post_processed_input*(1-self.envelope) + self.exciting_mei


class MaxMeiValueObjective(RegularIntervalObjective):
    def compute(self, current_state: State) -> Any:
        return current_state.preconditioned_grad
