import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def list_images(dataset_dir: Path):
    classes = sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()], key=lambda x: int(x))
    filepaths = []
    labels = []
    for idx, class_name in enumerate(classes):
        class_dir = dataset_dir / class_name
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for path in class_dir.glob(ext):
                filepaths.append(str(path))
                labels.append(idx)
    return filepaths, labels, classes


def decode_and_resize(path, label, img_size):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=1, expand_animations=False)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def build_dataset(paths, labels, img_size, batch_size, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: decode_and_resize(p, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def augmentation_layers():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.GaussianNoise(0.02),
        ],
        name="augmentation",
    )


def build_dnn(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = augmentation_layers()(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="dnn")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = augmentation_layers()(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="cnn")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_and_evaluate(model, model_name, train_ds, val_ds, test_ds, class_names, output_dir, epochs):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / f"{model_name}_best.keras"),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    preds = model.predict(test_ds)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.concatenate([y for _, y in test_ds], axis=0)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    (output_dir / model_name).mkdir(parents=True, exist_ok=True)
    model.save(output_dir / model_name / "model.keras")

    with open(output_dir / model_name / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", cbar=False)
    plt.title(f"{model_name.upper()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_dir / model_name / "confusion_matrix.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title(f"{model_name.upper()} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / model_name / "accuracy_curve.png", dpi=200)
    plt.close()

    summary = {
        "model": model_name,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train DNN and CNN for Urdu alphabet recognition.")
    parser.add_argument("--dataset", default="dataset", help="Path to dataset folder")
    parser.add_argument("--img-size", type=int, default=64, help="Input image size (square)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_dir = Path(args.dataset)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    paths, labels, class_names = list_images(dataset_dir)
    if not paths:
        raise RuntimeError("No images found in dataset. Check folder structure and file extensions.")

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.3, stratify=labels, random_state=args.seed
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=args.seed
    )

    img_size = (args.img_size, args.img_size)
    input_shape = (args.img_size, args.img_size, 1)

    train_ds = build_dataset(train_paths, train_labels, img_size, args.batch_size, training=True)
    val_ds = build_dataset(val_paths, val_labels, img_size, args.batch_size)
    test_ds = build_dataset(test_paths, test_labels, img_size, args.batch_size)

    dnn = build_dnn(input_shape, len(class_names))
    cnn = build_cnn(input_shape, len(class_names))

    summaries = []
    summaries.append(
        train_and_evaluate(dnn, "dnn", train_ds, val_ds, test_ds, class_names, output_dir, args.epochs)
    )
    summaries.append(
        train_and_evaluate(cnn, "cnn", train_ds, val_ds, test_ds, class_names, output_dir, args.epochs)
    )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": class_names,
                "img_size": args.img_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "results": summaries,
            },
            f,
            indent=2,
        )

    print("Training complete. Results saved to outputs/ directory.")


if __name__ == "__main__":
    main()
