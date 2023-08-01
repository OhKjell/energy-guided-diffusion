from pathlib import Path
import os

from dynamic.evaluations.single_cell_performance import get_performance_for_single_cell
from dynamic.models.cnn import Encoder
from dynamic.models.helper_functions import plot_responses_vs_predictions
from dynamic.utils.global_functions import cuda, get_cell_numbers_after_crop, get_cell_names


class Visualizer:
    def __init__(self, model: Encoder):
        self.model = model

    def visualize_model(self):
        self.model.visualize()

    def visualize_responses(self, dataloaders, performance="validation", plot=False):
        visualization_dir = os.path.join(
            self.model.config_dict["model_dir"],
            self.model.config_dict["model_name"],
            "visualizations",
        )
        response_plot_dir = os.path.join(visualization_dir, "cell_responses")
        Path(response_plot_dir).mkdir(exist_ok=True, parents=True)

        retina_index = self.model.config_dict["retina_index"]
        if "oracle_correlation_threshold" in self.model.config_dict.keys():
            correlation_threshold = self.model.config_dict[
                "oracle_correlation_threshold"
            ]
            explainable_variance_threshold = self.model.config_dict[
                "explainable_variance_threshold"
            ]
        else:
            correlation_threshold, explainable_variance_threshold = None, None
        correlations, all_predictions, all_responses = get_performance_for_single_cell(
            dataloaders,
            self.model.double(),
            retina_index=self.model.config_dict["retina_index"],
            device="cuda" if cuda else "cpu",
            rf_size=(self.model.config_dict["img_h"], self.model.config_dict["img_w"]),
            # rf_size=(0,10),
            img_h=self.model.config_dict["img_h"],
            img_w=self.model.config_dict["img_w"],
            performance=performance,
        )
        print("prediction shape", len(all_predictions), all_predictions[0].shape)

        cell_correlations = correlations
        cell_names_list = get_cell_names(
            retina_index,
            correlation_threshold=correlation_threshold
            if correlation_threshold is not None
            else 0,
            explained_variance_threshold=explainable_variance_threshold
            if explainable_variance_threshold is not None
            else 0,
        )
        cell_dict = {}
        for cell in range(
            get_cell_numbers_after_crop(
                retina_index,
                correlation_threshold=correlation_threshold
                if correlation_threshold is not None
                else 0,
                explained_variance_threshold=explainable_variance_threshold
                if explainable_variance_threshold is not None
                else 0,
            )
        ):
            cell_dict[cell_names_list[cell]] = cell_correlations[cell]
            if plot:
                plot_responses_vs_predictions(
                    all_responses=all_responses,
                    all_predictions=all_predictions,
                    cell=cell,
                    cell_name=cell_names_list[cell],
                    save_file=f"{response_plot_dir}/cell_{cell_names_list[cell]}_{performance}.png",
                    max_cc=cell_correlations[cell],
                    start_index=self.model.config_dict["layers"]
                    * (self.model.config_dict["num_of_frames"] - 1),
                )
            cell_dict[cell_names_list[cell]] = cell_correlations[cell]

            print(
                f"Cell {cell_names_list[cell]}, Avg correlation: {cell_correlations[cell]}"
            )
        return cell_dict
