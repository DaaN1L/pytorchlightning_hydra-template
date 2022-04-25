import albumentations as albu
import hydra
from omegaconf import DictConfig

from src.utils.utils import load_obj


def get_dataset(hparams: DictConfig):
    data = hparams.data
    image_folder = hydra.utils.to_absolute_path(data.image_folder)
    dataset_class = load_obj(hparams.dataset.class_name)

    train_augs_list = [
        load_obj(i["class_name"])(**i["params"])
        for i in hparams["augmentation"]["train"]["augs"]
    ]
    train_augs = albu.Compose(train_augs_list)

    valid_augs_list = [
        load_obj(i["class_name"])(**i["params"])
        for i in hparams["augmentation"]["valid"]["augs"]
    ]
    val_augs = albu.Compose(valid_augs_list)

    datasets = []
    for cur_path, augs in zip(
        [data.train_path, data.valid_path, data.test_path],
        [train_augs, val_augs, val_augs],
    ):

        cur_path = hydra.utils.to_absolute_path(cur_path)
        cur_dataset = dataset_class(
            csv_path=cur_path, image_folder=image_folder, preprocess=augs
        )
        datasets.append(cur_dataset)
    train_dataset, val_dataset, test_dataset = datasets
    return {"train": train_dataset, "val": val_dataset, "test": test_dataset}
