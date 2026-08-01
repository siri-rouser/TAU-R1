import os
import re

# Directory that contains the Roundabout-TAU annotation JSON files.
# Download from: https://huggingface.co/datasets/yl4300/Roundabout-TAU
# Override with: export TAU_ANNOTATION_DIR=/path/to/annotations

_ANNOTATION_DIR = os.environ.get(
    "TAU_ANNOTATION_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../datasets/Roundabout_TAU")),
)


def _ann(name: str) -> str:
    return os.path.join(_ANNOTATION_DIR, name)


# ---------------------------------------------------------------------------
# Roundabout-TAU dataset splits
# ---------------------------------------------------------------------------

ROUNDABOUT_TAU_ALL = {
    "annotation_path": _ann("roundabout_tau_all_qa_pairs.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_ALL_TRAIN = {
    "annotation_path": _ann("roundabout_tau_all_qa_pairs_train.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_ALL_TEST = {
    "annotation_path": _ann("roundabout_tau_all_qa_pairs_test.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_SIMPLE_DESCRIPTION = {
    "annotation_path": _ann("roundabout_tau_simple_description_qa_pairs.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_SIMPLE_DESCRIPTION_TRAIN = {
    "annotation_path": _ann("roundabout_tau_simple_description_qa_pairs_train.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_SIMPLE_DESCRIPTION_TEST = {
    "annotation_path": _ann("roundabout_tau_simple_description_qa_pairs_test.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_SIMPLE_REASONING = {
    "annotation_path": _ann("roundabout_tau_simple_reasoning_qa_pairs.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_SIMPLE_REASONING_TRAIN = {
    "annotation_path": _ann("roundabout_tau_simple_reasoning_qa_pairs_train.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_SIMPLE_REASONING_TEST = {
    "annotation_path": _ann("roundabout_tau_simple_reasoning_qa_pairs_test.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_CATEGORY = {
    "annotation_path": _ann("roundabout_tau_category_qa_pairs.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_CATEGORY_TRAIN = {
    "annotation_path": _ann("roundabout_tau_category_qa_pairs_train.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_CATEGORY_TEST = {
    "annotation_path": _ann("roundabout_tau_category_qa_pairs_test.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_ENVIRONMENT = {
    "annotation_path": _ann("roundabout_tau_environment_qa_pairs.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_ENVIRONMENT_TRAIN = {
    "annotation_path": _ann("roundabout_tau_environment_qa_pairs_train.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_ENVIRONMENT_TEST = {
    "annotation_path": _ann("roundabout_tau_environment_qa_pairs_test.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_OBJECT_GROUNDING = {
    "annotation_path": _ann("roundabout_tau_object_grounding_qa_pairs.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_OBJECT_GROUNDING_TRAIN = {
    "annotation_path": _ann("roundabout_tau_object_grounding_qa_pairs_train.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_OBJECT_GROUNDING_TEST = {
    "annotation_path": _ann("roundabout_tau_object_grounding_qa_pairs_test.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_SUMMARY = {
    "annotation_path": _ann("roundabout_tau_summary_qa_pairs.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_SUMMARY_TRAIN = {
    "annotation_path": _ann("roundabout_tau_summary_qa_pairs_train.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_SUMMARY_TEST = {
    "annotation_path": _ann("roundabout_tau_summary_qa_pairs_test.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_TIME_WINDOW = {
    "annotation_path": _ann("roundabout_tau_time_window_qa_pairs.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_TIME_WINDOW_TRAIN = {
    "annotation_path": _ann("roundabout_tau_time_window_qa_pairs_train.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_TIME_WINDOW_TEST = {
    "annotation_path": _ann("roundabout_tau_time_window_qa_pairs_test.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_SIMPLE_REASONING_WITH_CAT_TRAIN = {
    "annotation_path": _ann("roundabout_tau_simple_reasoning_qa_pairs_train_with_category.json"),
    "data_path": "",
}

ROUNDABOUT_TAU_SIMPLE_DESCRIPTION_WITH_CAT_TRAIN = {
    "annotation_path": _ann("roundabout_tau_simple_description_qa_pairs_train_with_category.json"),
    "data_path": "",
}

data_dict = {
    "roundabout_tau_all": ROUNDABOUT_TAU_ALL,
    "roundabout_tau_all_train": ROUNDABOUT_TAU_ALL_TRAIN,
    "roundabout_tau_all_test": ROUNDABOUT_TAU_ALL_TEST,
    "roundabout_tau_category": ROUNDABOUT_TAU_CATEGORY,
    "roundabout_tau_category_train": ROUNDABOUT_TAU_CATEGORY_TRAIN,
    "roundabout_tau_category_test": ROUNDABOUT_TAU_CATEGORY_TEST,
    "roundabout_tau_environment": ROUNDABOUT_TAU_ENVIRONMENT,
    "roundabout_tau_environment_train": ROUNDABOUT_TAU_ENVIRONMENT_TRAIN,
    "roundabout_tau_environment_test": ROUNDABOUT_TAU_ENVIRONMENT_TEST,
    "roundabout_tau_object_grounding": ROUNDABOUT_TAU_OBJECT_GROUNDING,
    "roundabout_tau_object_grounding_train": ROUNDABOUT_TAU_OBJECT_GROUNDING_TRAIN,
    "roundabout_tau_object_grounding_test": ROUNDABOUT_TAU_OBJECT_GROUNDING_TEST,
    "roundabout_tau_summary": ROUNDABOUT_TAU_SUMMARY,
    "roundabout_tau_summary_train": ROUNDABOUT_TAU_SUMMARY_TRAIN,
    "roundabout_tau_summary_test": ROUNDABOUT_TAU_SUMMARY_TEST,
    "roundabout_tau_simple_description": ROUNDABOUT_TAU_SIMPLE_DESCRIPTION,
    "roundabout_tau_simple_description_train": ROUNDABOUT_TAU_SIMPLE_DESCRIPTION_TRAIN,
    "roundabout_tau_simple_description_with_cat_train": ROUNDABOUT_TAU_SIMPLE_DESCRIPTION_WITH_CAT_TRAIN,
    "roundabout_tau_simple_description_test": ROUNDABOUT_TAU_SIMPLE_DESCRIPTION_TEST,
    "roundabout_tau_simple_reasoning": ROUNDABOUT_TAU_SIMPLE_REASONING,
    "roundabout_tau_simple_reasoning_train": ROUNDABOUT_TAU_SIMPLE_REASONING_TRAIN,
    "roundabout_tau_simple_reasoning_with_cat_train": ROUNDABOUT_TAU_SIMPLE_REASONING_WITH_CAT_TRAIN,
    "roundabout_tau_simple_reasoning_test": ROUNDABOUT_TAU_SIMPLE_REASONING_TEST,
    "roundabout_tau_time_window": ROUNDABOUT_TAU_TIME_WINDOW,
    "roundabout_tau_time_window_train": ROUNDABOUT_TAU_TIME_WINDOW_TRAIN,
    "roundabout_tau_time_window_test": ROUNDABOUT_TAU_TIME_WINDOW_TEST,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["roundabout_tau_summary_train"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
