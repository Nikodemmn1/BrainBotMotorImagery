import pickle
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from Models.OneDNet import OneDNet
from Models.OneDNetEnsembleInception import OneDNetEnsemble
from Dataset.dataset import EEGDataset
from Dataset.dataset_ensemble import EEGDatasetEnsemble
import torch
import os

included_classes = [0, 1, 2]
included_channels = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
model_names = ["left", "right", "forward", "noise"]
DATA_PATH = "/home/administrator3/Brainbot/WielkieTesty/Dane"
MODELS_PATH = "/home/administrator3/Brainbot/WielkieTesty/Modele"
RESULTS_PATH = "/home/administrator3/Brainbot/WielkieTesty/Wyniki"


tb_logger = TensorBoardLogger(save_dir=os.path.join(RESULTS_PATH, "lightning_logs"))


def renormalize_data(dataset, mean_std_old, mean_std_new):
    mean_old = mean_std_old["mean"][included_channels]
    std_old = mean_std_old["std"][included_channels]
    mean_new = mean_std_new["mean"][included_channels]
    std_new = mean_std_new["std"][included_channels]
    dataset.data = ((dataset.data * std_old[None, None, :, None] + mean_old[None, None, :, None]) - \
                   mean_new[None, None, :, None]) / std_new[None, None, :, None]


def load_mean_std(path):
    with open(path, "rb") as mean_std_file:
        mean_std = pickle.load(mean_std_file)
    return mean_std


def load_data_info():
    two_path = os.path.join(DATA_PATH, "Two")
    multi_path = os.path.join(DATA_PATH, "Multi")
    two_file_bases = [p[:-4] for p in os.listdir(two_path) if p.endswith(".npy")]
    multi_file_bases = [p[:-4] for p in os.listdir(multi_path) if p.endswith(".npy")]
    two = dict()
    multi = dict()

    for f in sorted(two_file_bases):
        f_path = os.path.join(two_path, f + ".npy")
        mean_std_path = os.path.join(two_path, f + ".pkl")
        two[f] = {"p": f_path, "ms": mean_std_path, "b": f}

    for f in sorted(multi_file_bases):
        f_path = os.path.join(multi_path, f + ".npy")
        mean_std_path = os.path.join(multi_path, f + ".pkl")
        multi[f] = {"p": f_path, "ms": mean_std_path, "b": f}

    return two, multi


def perform_test(model_path, data_path, old_mean_std, new_mean_std, results_path, test_type, class_id=None):
    if test_type == "multi":
        full_dataset = EEGDataset(data_path, data_path, data_path, included_classes, included_channels)
    else:
        full_dataset = EEGDatasetEnsemble(data_path, data_path, data_path, class_id, included_channels)

    renormalize_data(full_dataset, old_mean_std, new_mean_std)

    _, _, test_dataset = full_dataset.get_subsets()
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=12)

    if test_type == "multi":
        model = OneDNet.load_from_checkpoint(included_classes=included_classes, checkpoint_path=model_path)
    else:
        model = OneDNetEnsemble.load_from_checkpoint(checkpoint_path=model_path)

    trainer = Trainer(gpus=-1)
    test_results = trainer.test(model, test_data, verbose=True)
    test_lines = [f"{key}: {val}\n" for key, val in test_results[0].items()]
    with open(results_path, 'a') as f:
        if test_type == "two":
            f.write(f"----------- {model_names[class_id].capitalize()} -----------\n")
        f.writelines(test_lines)


def perform_two_tests(main_path, data, base_results_name):
    old_mean_std = load_mean_std(os.path.join(main_path, "mean_std.pkl"))
    for data_name in data:
        for class_id, model_name in enumerate(model_names):
            results_name = f"{base_results_name}{data_name}.txt"
            results_path = os.path.join(RESULTS_PATH, results_name)
            new_mean_std = load_mean_std(data[data_name]['ms'])
            model_path = os.path.join(main_path, f"{model_name}.ckpt")
            perform_test(model_path, data[data_name]['p'], old_mean_std, new_mean_std, results_path, "two", class_id)


def perform_multi_tests(main_path, data, base_results_name):
    old_mean_std = load_mean_std(os.path.join(main_path, "mean_std.pkl"))
    for data_name in data:
        results_name = f"{base_results_name}{data_name}.txt"
        results_path = os.path.join(RESULTS_PATH, results_name)
        new_mean_std = load_mean_std(data[data_name]['ms'])
        model_path = os.path.join(main_path, "model.ckpt")
        perform_test(model_path, data[data_name]['p'], old_mean_std, new_mean_std, results_path, "multi")


def main():
    model_base_names = os.listdir(MODELS_PATH)
    two_data, multi_data = load_data_info()
    for mbn in sorted(model_base_names):
        base_path = os.path.join(MODELS_PATH, mbn)
        multi_path = os.path.join(base_path, "Multi")
        two_path = os.path.join(base_path, "Two")
        perform_multi_tests(multi_path, multi_data, f"Multi_{mbn}_")
        perform_two_tests(two_path, two_data, f"Two_{mbn}_")



if __name__ == "__main__":
    main()
