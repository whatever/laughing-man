#!/usr/bin/env python3


import os.path
import random
import shutil


from glob import glob


if __name__ == "__main__":

    # To make the the run immutable
    random.seed(420)

    images = [
        fname
        for fname in glob("images/*.jpg")
    ]

    labels = [
        fname
        for fname in glob("labels/*.json")
    ]

    n_train = int(0.7 * len(labels))
    n_test = int(0.15 * len(labels))
    n_validate = len(labels) - n_train - n_test


    random.shuffle(labels)


    train_labels = labels[:n_train]
    test_labels = labels[n_train:(n_train + n_test)]
    validate_labels = labels[n_train+n_test:]

    assert len(labels) == len(train_labels) + len(test_labels) + len(validate_labels)


    for fname in train_labels:
        label_dest = os.path.join("data", "train", "labels", os.path.basename(fname))
        image_dest = os.path.join("data", "train", "images", os.path.basename(fname).replace(".json", ".jpg"))
        shutil.copy(fname, label_dest)
        shutil.copy(fname.replace("labels", "images").replace(".json", ".jpg"), image_dest)

    for fname in test_labels:
        label_dest = os.path.join("data", "test", "labels", os.path.basename(fname))
        image_dest = os.path.join("data", "test", "images", os.path.basename(fname).replace(".json", ".jpg"))
        shutil.copy(fname, label_dest)
        shutil.copy(fname.replace("labels", "images").replace(".json", ".jpg"), image_dest)

    for fname in validate_labels:
        label_dest = os.path.join("data", "validate", "labels", os.path.basename(fname))
        image_dest = os.path.join("data", "validate", "images", os.path.basename(fname).replace(".json", ".jpg"))
        shutil.copy(fname, label_dest)
        shutil.copy(fname.replace("labels", "images").replace(".json", ".jpg"), image_dest)
