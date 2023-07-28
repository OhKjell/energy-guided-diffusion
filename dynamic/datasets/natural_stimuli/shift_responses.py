import numpy as np
import pickle
from utils.global_functions import home


def shift_responses(response_dict, test_frames=5067):
    responses_train = response_dict["train_responses"]
    responses_test = response_dict["test_responses"]
    # new_train_responses = np.array((responses_train.shape[:2]) + (responses_train.shape[2] + responses_test.shape[2]-test_frames,))
    # new_test_responses = np.array((responses_test.shape[:2]) + (test_frames,))

    new_test_responses = responses_test[:, :, :test_frames]
    new_train_responses = np.concatenate(
        (responses_test[:, :, test_frames:], responses_train), axis=2
    )

    new_response_dict = response_dict
    new_response_dict["seed"] = 2022
    new_response_dict["trial_separation"] = {
        2022: [x for x in range(new_train_responses.shape[1])]
    }
    new_response_dict["train_responses"] = new_train_responses
    new_response_dict["test_responses"] = new_test_responses

    return new_response_dict


if __name__ == "__main__":
    d = 9
    with open(
        f"{home}/data/marmoset_data/responses/06_fixationmovieflip_marmo_85Hz_seed2022.pkl",
        "rb",
    ) as f:
        responses = pickle.load(f)
    shifted_responses = shift_responses(responses, 5067)
    with open(
        f"{home}/data/marmoset_data/responses/cell_responses_02_fixation_movie_shifted.pkl",
        "wb",
    ) as f:
        pickle.dump(responses, f)
