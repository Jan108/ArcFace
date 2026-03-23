import argparse
import os
from datetime import datetime, timedelta
from itertools import islice

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbones import get_model
from dataset import PetFaceVerification, PetFaceIdentification


@torch.no_grad()
def verification(params):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    net = get_model(params.network, fp16=False)
    net.load_state_dict(torch.load(params.weights))
    net.eval()
    net.to(device)

    # Load data
    dataset = PetFaceVerification(params.img_path, params.img_verification)
    test_loader = DataLoader(
        dataset=dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )

    # Predict similarity
    sim_list = []
    label_list = []
    for img1, img2, label in tqdm(test_loader, desc='Test image pairs'):
        vec1 = F.normalize(net(img1.to(device)))
        vec2 = F.normalize(net(img2.to(device)))

        sim = nn.CosineSimilarity()(vec1, vec2).cpu().data.numpy().tolist()
        sim_list += sim
        label_list += label.cpu().data.numpy().tolist()

    os.makedirs(os.path.dirname(params.output), exist_ok=True)
    pd.DataFrame(
        {'file1': dataset.image1_list, 'file2': dataset.image2_list,
         'sim': sim_list, 'label': label_list}).to_csv(
        os.path.join(params.output, 'verification.csv'), index=False)

    # Test latency
    test_loader_latency = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )
    inf_times = []
    for img1, img2, label in tqdm(islice(test_loader_latency, params.latency_test),
                                  desc='Test image pairs latency', total=params.latency_test):
        start_time = datetime.now()
        F.normalize(net(img1.to(device)))
        F.normalize(net(img2.to(device)))
        end_time = datetime.now()
        inf_times.append(end_time - start_time)
    avg_time = sum(inf_times, start=timedelta()) / len(inf_times)
    print(f'Inference for {params.weights} took {avg_time}')
    with open(os.path.join(params.output, 'timing.txt'), 'w') as file:
        file.write(str(avg_time))


def identification(params):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pool_loader = DataLoader(PetFaceIdentification(params.img_path, params.img_identification, pool_only=True), batch_size=64, shuffle=False)
    test_loader = DataLoader(PetFaceIdentification(params.img_path, params.img_identification, pool_only=False), batch_size=16, shuffle=False)

    # Load model
    net = get_model(params.network, fp16=False)
    net.load_state_dict(torch.load(params.weights))
    net.eval()
    net.to(device)

    features, labels = [], []
    net.eval()
    with torch.no_grad():
        for images, batch_labels in tqdm(pool_loader, desc="Extracting features"):
            images = images.to(device)
            batch_features = net(images)
            batch_features = F.normalize(batch_features, p=2, dim=1)
            features.append(batch_features)
            labels.extend(batch_labels.tolist())

    pool_features, pool_labels = torch.cat(features, dim=0), labels
    pool_features = pool_features.to(device)

    all_labels = []
    all_sims = []
    all_indices = []

    # Process test images in batches
    for test_images, test_labels in tqdm(test_loader, desc="Processing test images"):
        test_images = test_images.to(device)
        test_features = net(test_images)
        test_features = F.normalize(test_features, p=2, dim=1)

        # Compute cosine similarity (dot product of normalized vectors)
        sim = torch.matmul(test_features, pool_features.T)  # (batch_size, num_pool)
        # Get top 5 similarities and indices
        top5_sim, top5_indices = torch.topk(sim, k=5, dim=1)

        all_labels.extend(test_labels.tolist())
        all_sims.extend(top5_sim.tolist())
        all_indices.extend(top5_indices.tolist())

    # Open output file
    os.makedirs(os.path.dirname(params.output), exist_ok=True)
    with open(os.path.join(params.output, params.identification_file), 'w') as f:
        f.write(f"test_label,{','.join([f'predicted_label_{i}' for i in range(5)])},{','.join([f'similarity_{i}' for i in range(5)])}\n")
        for test_label, pred_label, sim in zip(all_labels, all_indices, all_sims):
            f.write(f"{test_label},{','.join([str(i) for i in pred_label])},{','.join([str(i) for i in sim])}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Evaluation')
    parser.add_argument("--network", type=str, default='r50', help="network name: r18, r34, r50, r100")
    parser.add_argument("--output", type=str, default='./work_dir', help="Output directory")
    parser.add_argument("--weights", type=str, default='./work_dir', help="Model weights")
    parser.add_argument("--img_path", type=str, default='./data/images', help="Image file rott directory")
    parser.add_argument("--img_verification", type=str, default='./data/split/verification.csv',
                        help="File containing a list of images files with corresponding label for verification")
    parser.add_argument("--img_identification", type=str, default='./data/split/identification.csv',
                        help="File containing a list of images files with corresponding label for identification")
    parser.add_argument("--latency_test", type=int, default=1000, help="Amount of images for the latency test")
    parser.add_argument("--ident-general", action='store_true', help="Generalized model evaluation")
    args = parser.parse_args()
    # verification(args)
    args.identification_file = 'identification.csv'
    if args.ident_general:
        base_path_ident = args.img_identification
        print('Loading and performing each class separately for generalized model')
        for cls in ['bird', 'cat', 'dog', 'small_animals']:
            args.img_identification = f'{base_path_ident}/{cls}/identification_img.csv'
            args.identification_file = f'identification_{cls}.csv'
            identification(args)
    else:
        identification(args)
