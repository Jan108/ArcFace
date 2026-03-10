import os
import queue as Queue
import threading
from functools import partial
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn


def get_dataloader(
    root_dir,
    img_list,
    local_rank,
    batch_size,
    seed = 2048,
    num_workers = 2,
    ) -> Iterable:

    train_set = PetFaceDataset(root_dir, img_list)

    # Synthetic
    if root_dir == "synthetic":
        train_set = SyntheticDataset()

    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader, train_set.num_classes, len(train_set)

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class PetFaceDataset(Dataset):
    def __init__(self, root_dir, img_list):
        super(PetFaceDataset, self).__init__()

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.image_list = []
        self.label_list = []

        with open(img_list, 'r') as f:
            for line in f:
                line = line.strip()
                img_path, img_label = line.split(',')
                if img_path == 'label':
                    continue

                img_path = os.path.join(root_dir, img_path)
                if not os.path.exists(img_path):
                    continue

                self.image_list.append(img_path)
                self.label_list.append(int(img_label))
        self.num_classes = len(set(self.label_list))
        print(f'Loaded Dataset from {img_list}: found {len(self.image_list)} images with {self.num_classes} classes')


    def __getitem__(self, index):
        path_img = self.image_list[index]
        label = self.label_list[index]
        img = Image.open(path_img).convert('RGB').crop((28, 28, 196, 169)).resize((112, 112), Image.LANCZOS)

        sample = self.transform(img)
        label = torch.tensor(label, dtype=torch.long)
        return sample, label

    def __len__(self):
        return len(self.image_list)


class SyntheticDataset(Dataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000
