import os
import re

# Directory that contains the Roundabout-TAU annotation JSON files.
# Download from: https://huggingface.co/datasets/yl4300/Roundabout-TAU
# Override with: export TAU_ANNOTATION_DIR=/path/to/annotations
_ANNOTATION_DIR = os.environ.get("TAU_ANNOTATION_DIR", "/data/Roundabout-TAU")


def _ann(name: str) -> str:
    return os.path.join(_ANNOTATION_DIR, name)


# ---------------------------------------------------------------------------
# Roundabout-TAU dataset splits
# ---------------------------------------------------------------------------

ROUNDABOUT_VAU_ALL = {
    "annotation_path": _ann("roundabout_vau_all_qa_pairs.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_ALL_TRAIN = {
    "annotation_path": _ann("roundabout_vau_all_qa_pairs_train.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_ALL_TEST = {
    "annotation_path": _ann("roundabout_vau_all_qa_pairs_test.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_SIMPLE_DESCRIPTION = {
    "annotation_path": _ann("roundabout_vau_simple_description_qa_pairs.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_SIMPLE_DESCRIPTION_TRAIN = {
    "annotation_path": _ann("roundabout_vau_simple_description_qa_pairs_train.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_SIMPLE_DESCRIPTION_TEST = {
    "annotation_path": _ann("roundabout_vau_simple_description_qa_pairs_test.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_SIMPLE_REASONING = {
    "annotation_path": _ann("roundabout_vau_simple_reasoning_qa_pairs.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_SIMPLE_REASONING_TRAIN = {
    "annotation_path": _ann("roundabout_vau_simple_reasoning_qa_pairs_train.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_SIMPLE_REASONING_TEST = {
    "annotation_path": _ann("roundabout_vau_simple_reasoning_qa_pairs_test.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_CATEGORY = {
    "annotation_path": _ann("roundabout_vau_category_qa_pairs.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_CATEGORY_TRAIN = {
    "annotation_path": _ann("roundabout_vau_category_qa_pairs_train.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_CATEGORY_TEST = {
    "annotation_path": _ann("roundabout_vau_category_qa_pairs_test.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_ENVIRONMENT = {
    "annotation_path": _ann("roundabout_vau_environment_qa_pairs.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_ENVIRONMENT_TRAIN = {
    "annotation_path": _ann("roundabout_vau_environment_qa_pairs_train.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_ENVIRONMENT_TEST = {
    "annotation_path": _ann("roundabout_vau_environment_qa_pairs_test.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_OBJECT_GROUNDING = {
    "annotation_path": _ann("roundabout_vau_object_grounding_qa_pairs.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_OBJECT_GROUNDING_TRAIN = {
    "annotation_path": _ann("roundabout_vau_object_grounding_qa_pairs_train.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_OBJECT_GROUNDING_TEST = {
    "annotation_path": _ann("roundabout_vau_object_grounding_qa_pairs_test.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_SUMMARY = {
    "annotation_path": _ann("roundabout_vau_summary_qa_pairs.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_SUMMARY_TRAIN = {
    "annotation_path": _ann("roundabout_vau_summary_qa_pairs_train.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_SUMMARY_TEST = {
    "annotation_path": _ann("roundabout_vau_summary_qa_pairs_test.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_TIME_WINDOW = {
    "annotation_path": _ann("roundabout_vau_time_window_qa_pairs.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_TIME_WINDOW_TRAIN = {
    "annotation_path": _ann("roundabout_vau_time_window_qa_pairs_train.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_TIME_WINDOW_TEST = {
    "annotation_path": _ann("roundabout_vau_time_window_qa_pairs_test.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_SIMPLE_REASONING_WITH_CAT_TRAIN = {
    "annotation_path": _ann("roundabout_vau_simple_reasoning_qa_pairs_train_with_category.json"),
    "data_path": "",
}

ROUNDABOUT_VAU_SIMPLE_DESCRIPTION_WITH_CAT_TRAIN = {
    "annotation_path": _ann("roundabout_vau_simple_description_qa_pairs_train_with_category.json"),
    "data_path": "",
}

data_dict = {
    "roundabout_vau_all": ROUNDABOUT_VAU_ALL,
    "roundabout_vau_all_train": ROUNDABOUT_VAU_ALL_TRAIN,
    "roundabout_vau_all_test": ROUNDABOUT_VAU_ALL_TEST,
    "roundabout_vau_category": ROUNDABOUT_VAU_CATEGORY,
    "roundabout_vau_category_train": ROUNDABOUT_VAU_CATEGORY_TRAIN,
    "roundabout_vau_category_test": ROUNDABOUT_VAU_CATEGORY_TEST,
    "roundabout_vau_environment": ROUNDABOUT_VAU_ENVIRONMENT,
    "roundabout_vau_environment_train": ROUNDABOUT_VAU_ENVIRONMENT_TRAIN,
    "roundabout_vau_environment_test": ROUNDABOUT_VAU_ENVIRONMENT_TEST,
    "roundabout_vau_object_grounding": ROUNDABOUT_VAU_OBJECT_GROUNDING,
    "roundabout_vau_object_grounding_train": ROUNDABOUT_VAU_OBJECT_GROUNDING_TRAIN,
    "roundabout_vau_object_grounding_test": ROUNDABOUT_VAU_OBJECT_GROUNDING_TEST,
    "roundabout_vau_summary": ROUNDABOUT_VAU_SUMMARY,
    "roundabout_vau_summary_train": ROUNDABOUT_VAU_SUMMARY_TRAIN,
    "roundabout_vau_summary_test": ROUNDABOUT_VAU_SUMMARY_TEST,
    "roundabout_vau_simple_description": ROUNDABOUT_VAU_SIMPLE_DESCRIPTION,
    "roundabout_vau_simple_description_train": ROUNDABOUT_VAU_SIMPLE_DESCRIPTION_TRAIN,
    "roundabout_vau_simple_description_with_cat_train": ROUNDABOUT_VAU_SIMPLE_DESCRIPTION_WITH_CAT_TRAIN,
    "roundabout_vau_simple_description_test": ROUNDABOUT_VAU_SIMPLE_DESCRIPTION_TEST,
    "roundabout_vau_simple_reasoning": ROUNDABOUT_VAU_SIMPLE_REASONING,
    "roundabout_vau_simple_reasoning_train": ROUNDABOUT_VAU_SIMPLE_REASONING_TRAIN,
    "roundabout_vau_simple_reasoning_with_cat_train": ROUNDABOUT_VAU_SIMPLE_REASONING_WITH_CAT_TRAIN,
    "roundabout_vau_simple_reasoning_test": ROUNDABOUT_VAU_SIMPLE_REASONING_TEST,
    "roundabout_vau_time_window": ROUNDABOUT_VAU_TIME_WINDOW,
    "roundabout_vau_time_window_train": ROUNDABOUT_VAU_TIME_WINDOW_TRAIN,
    "roundabout_vau_time_window_test": ROUNDABOUT_VAU_TIME_WINDOW_TEST,
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
    dataset_names = ["roundabout_vau_summary_train"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
