import argparse
import os

import pandas as pd
import torch
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Refine Stanford Cars dataset with car color predictions.")
    parser.add_argument("sc_data_root", help="Path to the Stanford Cars dataset root directory.")
    parser.add_argument("model_path", help="Path to the color classifier model.")
    parser.add_argument("output_dir", help="Path to the output directory.")

    return parser.parse_args()


def main():
    args = parse_args()

    train_img_dir = os.path.join(args.sc_data_root, "cars_train", "cars_train")
    label_path = os.path.join(args.sc_data_root, "train.csv")
    refined_label_path = os.path.join(args.output_dir, "refined_train.csv")

    refine(train_img_dir, label_path, refined_label_path, args.model_path)


def refine(train_img_dir, label_path, refined_label_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = torch.jit.load(model_path, map_location=device).to(device)
    model.eval()

    # Image transform
    img_transform = transforms.Compose([
        transforms.Resize((model.img_size, model.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=model.mean, std=model.std)
    ])

    # Load labels
    df = pd.read_csv(label_path)

    # Refine labels
    for i, row in tqdm(df.iterrows()):
        img_path = os.path.join(train_img_dir, row["image"])
        cls_label = row["Class"]

        try:
            img = Image.open(img_path)
            color = get_color_prediction(img, model, img_transform, device)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Not able to predict car color for: {os.path.basename(img_path)}")
            df.drop(i, inplace=True)
            continue

        instance_cls = f"{cls_label}_{color}"
        df.at[i, "Class"] = instance_cls

    df.to_csv(refined_label_path, index=False)


def get_color_prediction(input_img, classifier, transform, device):
    input_tensor = transform(input_img).unsqueeze(0)
    output = classifier(input_tensor.to(device))
    _, pred = torch.max(output, 1)
    return pred.item()


if __name__ == "__main__":
    main()


