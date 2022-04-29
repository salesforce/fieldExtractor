import argparse
import pickle as pkl
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img_dir",
        default='datasets/imgs',
        type=str,
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='output_vis'
    )

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.pred_path, 'rb') as f:
        preds = pkl.load(f)

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.figure(figsize=(30, 20))
    for im_name in tqdm(preds):
        image_path = os.path.join(args.img_dir, im_name+'.png')
        image = Image.open(image_path)

        fig, ax = plt.subplots(1,2)
        ax[0].imshow(image)
        column_labels = []
        data = []
        for i, fd in enumerate(preds[im_name]):
            value = preds[im_name][fd]
            column_labels.append(fd)
            data.append(value)

        ax[1].table(cellText=[data], colLabels=column_labels, loc="center")
        ax[0].axis('off')
        ax[1].axis('off')
        plt.savefig(os.path.join(args.output_dir, im_name+'.png'))
        plt.clf()
        plt.close()

if __name__ == "__main__":
    main()
