import argparse
import os
import sys

import pandas as pd
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def process_imagefolder(root, save_dir="./data"):
    # turn a dir of root/set/image into csv
    df = []
    for set in os.listdir(root):
        if os.path.isdir(f"{root}/{set}"):
            for img in os.listdir(f"{root}/{set}"):
                df.append({"subset": set, "path": f"{root}/{set}/{img}"})
    df = pd.DataFrame(df)
    return df


metadata = pd.read_csv("applications/imagenetV2/imagenetV2_meta.csv").drop_duplicates(
    subset=["wnid"]
)
wnid_to_class_name = {}
for i in range(len(metadata)):
    wnid_to_class_name[metadata.iloc[i]["wnid"]] = metadata.iloc[i]["class_name"]
class_num_to_wnid = {}
for i in range(len(metadata)):
    class_num_to_wnid[str(metadata.iloc[i]["class_num"])] = metadata.iloc[i]["wnid"]
wnid_to_class_num = {}
for i in range(len(metadata)):
    wnid_to_class_num[metadata.iloc[i]["wnid"]] = metadata.iloc[i]["class_num"]


def process_imagenet(imagenet_root, save_dir="./data"):
    print("Processing ImageNet dataset")
    df = process_imagefolder(f"{imagenet_root}/val", save_dir)
    df["group_name"] = "imagenet"
    df["class_name"] = df["subset"].apply(lambda x: wnid_to_class_name[x])
    df["class_num"] = df["subset"].apply(lambda x: wnid_to_class_num[x])
    df["wnid"] = df["subset"]
    return df


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_paths import CSV_SAVE_DIR, IMAGENET_PATH, IMAGENETV2_PATH

    # call function by add process_ to the dataset name
    df = process_imagenet(IMAGENET_PATH, CSV_SAVE_DIR)

    df.to_csv(f"{CSV_SAVE_DIR}/imagenet.csv", index=False)
