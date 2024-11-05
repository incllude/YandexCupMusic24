from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchaudio.transforms import FrequencyMasking, TimeMasking
from typing import Dict, Literal, Tuple, TypedDict
import torch.nn.functional as F
from pathlib import Path
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import torch
import os


class Config:
    
    data_path = Path("...")
    file_ext = "npy"
    n_neg_samples = 5
    batch_size = 200
    epoch_size = 0.25
    random_state = 43
    epochs = 30
    
    def seed_everything(self):
        torch.backends.cudnn.deterministic = False
        torch.cuda.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        random.seed(self.random_state)


class BatchDict(TypedDict):
    anchor: torch.Tensor
    anchor_id: int
    anchor_clique: int
    positive: torch.Tensor
    negative: torch.Tensor
    negative_clique: int


def generate_random_binary_mask(image_height, image_width, mask_height, mask_width):

    mask = torch.zeros((1, image_height, image_width), dtype=torch.int32)

    max_y = image_height - mask_height
    max_x = image_width - mask_width
    top_left_y = torch.randint(0, max_y + 1, (1,)).item()
    top_left_x = torch.randint(0, max_x + 1, (1,)).item()

    mask[:, top_left_y:top_left_y + mask_height, top_left_x:top_left_x + mask_width] = 1
    return mask


class CoverDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        file_ext: str,
        dataset_path: str,
        data_split: Literal["train", "val", "test"],
        n_neg_samples: int,
        num_of_start_exploration: int = 0,
        exploration_prob: float = 0.5,
        l_mixup: Tuple[float, float] = (0., 1.),
        l_cutmix: Tuple[float, float] = (0., 1.),
    ) -> None:
        super().__init__()
        
        self.num_of_start_exploration = num_of_start_exploration
        self.p_explore = exploration_prob
        self.n_neg_samples = n_neg_samples
        self.data_path = Path(data_path)
        self.file_ext = file_ext
        self.dataset_path = Path(dataset_path)
        self.data_split = data_split
        self._load_data()
        
        if data_split == "train":
            self.l_mixup_min, self.l_mixup_max = l_mixup
            self.l_cutmix_min, self.l_cutmix_max = l_cutmix
            self.mask_aug = nn.Sequential(FrequencyMasking(freq_mask_param=25),
                                          TimeMasking(time_mask_param=25))
            self.neg_dist_matrix = self._generate_neg_dist_matrix()
            self.pos_dist_vector = self._generate_pos_dist_vector()
        
        if data_split != "test":
            self.value_counts = self.version2clique.clique.value_counts().sort_index().values
            self.num_classes = self.value_counts.shape[0]

    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, index: int) -> BatchDict:
        track_id = self.track_ids[index]
        anchor_cqt = self._load_cqt(track_id)
        
        if self.data_split in ["train", "val"]:
            anchor_idx = self.version2clique.loc[track_id, 'clique']
        else:
            anchor_idx = -1
        if self.data_split == "train":
            pos_idxs, neg_idxs, imposter = self._triplet_sampling(track_id, anchor_idx)
            neg_cliques = self.version2clique.loc[neg_idxs].clique.values
            positive_cqt = [self._load_cqt(pos_idx) for pos_idx in pos_idxs]
            negative_cqt = [self._load_cqt(neg_idx) for neg_idx in neg_idxs]
            if imposter:
                anchor_cqt = random.choice((self.cutmix_aug, self.mixup_aug))(anchor_cqt, positive_cqt[1])
            else:
                anchor_cqt = self.cutout_aug(anchor_cqt)
        else:
            positive_cqt = torch.empty(0)
            negative_cqt = torch.empty(0)
            neg_cliques = torch.empty(0)
                
        return dict(
            anchor=anchor_cqt,
            anchor_id=track_id,
            anchor_clique=anchor_idx,
            positive=positive_cqt,
            negative=negative_cqt,
            negative_clique=neg_cliques
        )

    def _make_file_path(self, track_id, file_ext):
        a = track_id % 10
        b = track_id // 10 % 10
        c = track_id // 100 % 10
        return os.path.join(str(c), str(b), str(a), f'{track_id}.{file_ext}')
    
    def cutout_aug(self, cqt: torch.Tensor) -> torch.Tensor:
        
        lam = np.sqrt(np.random.uniform(self.l_cutmix_min, self.l_cutmix_max))
        h, w = cqt.size(1), cqt.size(2)
        m_h, m_w = int(h * lam), int(w * lam)
        mask = generate_random_binary_mask(h, w, m_h, m_w)
        aug_cqt = cqt * (~mask) + cqt.mean() * mask
        
        return aug_cqt
    
    def cutmix_aug(self, cqt1: torch.Tensor, cqt2: torch.Tensor) -> torch.Tensor:
        
        lam = np.random.uniform(self.l_cutmix_min, self.l_cutmix_max)
        h, w = cqt1.size(1), cqt1.size(2)
        m_h, m_w = int(h * lam), int(w * lam)
        mask = generate_random_binary_mask(h, w, m_h, m_w)
        aug_cqt = cqt1 * (~mask) + cqt2 * mask
        
        return aug_cqt
    
    def mixup_aug(self, cqt1: torch.Tensor, cqt2: torch.Tensor) -> torch.Tensor:
        
        lam = np.random.uniform(self.l_mixup_min, self.l_mixup_max)
        aug_cqt = lam * cqt1 + (1 - lam) * cqt2

        return aug_cqt
    
    def _pitch_shift_aug(self, cqt: torch.Tensor) -> torch.Tensor:
        
        if np.random.random() < 0.5:
            return F.pad(cqt, pad=(0, 0, 0, 10), mode='reflect')[:, 10:]
        return F.pad(cqt, pad=(0, 0, 10, 0), mode='reflect')[:, :-10]
    
    def _time_shift_aug(self, cqt: torch.Tensor) -> torch.Tensor:
        
        if np.random.random() < 0.5:
            c = F.pad(cqt, pad=(6, 0, 0, 0), mode='reflect')[..., :-6]
            c[..., :6] = c[..., :6].flip(dims=(-1,))
        else:
            c = F.pad(cqt, pad=(0, 6, 0, 0), mode='reflect')[..., 6:]
            c[..., -6:] = c[..., -6:].flip(dims=(-1,))
        return c
    
    def _noise_aug(self, cqt: torch.Tensor) -> torch.Tensor:
        
        return torch.randn_like(cqt) * 4 + cqt

    def _triplet_sampling(self, track_id: int, clique_id: int) -> Tuple[int, int]:
        
        covers = self.versions.drop(clique_id)
        if self.num_of_start_exploration != 0:
            neg_covers = covers.index
            self.num_of_start_exploration -= 1
        elif np.random.random() < self.p_explore:
            neg_covers = covers.index
        else:
            n = int(self.n_neg_samples * 1.5)
            neg_covers = np.argsort(self.neg_dist_matrix[clique_id])[::-1][:n]
                
        neg_covers = np.random.choice(neg_covers, size=self.n_neg_samples, replace=False)    
        neg_ids = covers.loc[neg_covers, "versions"].apply(lambda x: np.random.choice(x, size=2, replace=False))
        neg_ids = np.concatenate(neg_ids.values)
        
        covers = self.versions.loc[clique_id, "versions"]
        covers = np.setdiff1d(covers, track_id)
        if len(covers) == 1:
            pos_covers = self.versions.drop(neg_covers).drop(clique_id)
            pos_cover = np.random.choice(pos_covers.sample(1).versions.values[0], size=1)
            pos_ids = [covers[0], pos_cover[0]]
            add_sample = True
        else:
            pos_ids = np.random.choice(covers, size=2, replace=False)
            add_sample = False
        
        return pos_ids, neg_ids, add_sample
    
    def _generate_neg_dist_matrix(self) -> np.array:
        
        n = self.versions.shape[0]
        neg_dist_matrix = np.random.uniform(low=1.25, high=1.75, size=(n, n)).astype(np.float16)
        np.fill_diagonal(neg_dist_matrix, -float("inf"))
            
        return neg_dist_matrix
    
    def _generate_pos_dist_vector(self) -> np.array:
        
        n = self.versions.shape[0]
        pos_dist_matrix = np.random.uniform(low=0.85, high=1.15, size=n).astype(np.float16)
        
        return pos_dist_matrix
    
    def update_neg_dist_matrix(self, clique_i: int, clique_j: int, cos: float) -> None:
        
        self.neg_dist_matrix[clique_i, clique_j] = 2/3 * cos + 1/3 * self.neg_dist_matrix[clique_i, clique_j]
        self.neg_dist_matrix[clique_j, clique_i] = self.neg_dist_matrix[clique_i, clique_j]
        
    def update_pos_dist_vector(self, clique_i: int, cos: float) -> None:
        
        self.pos_dist_vector[clique_i] = 2/3 * cos + 1/3 * self.pos_dist_vector[clique_i]

    def _load_data(self) -> None:
        if self.data_split in ['train', 'val']:
            cliques_subset = np.load(self.data_path.joinpath(f"splits/{self.data_split}_cliques.npy"))
            self.versions = pd.read_csv(
                self.data_path.joinpath("cliques2versions.tsv"), sep='\t', converters={"versions": eval}
            )
            self.versions = self.versions[self.versions["clique"].isin(set(cliques_subset))]
            sizes_of_cliques = self.versions.versions.apply(len) #.reset_index(drop=True)
            self.versions = self.versions.loc[sizes_of_cliques > 1]
            cliques_subset = self.versions.clique.values
            mapping = {}
            for k, clique in enumerate(sorted(cliques_subset)):
                mapping[clique] = k
            self.versions["clique"] = self.versions["clique"].map(lambda x: mapping[x])
            self.versions.set_index("clique", inplace=True)
            self.version2clique = pd.DataFrame(
                [{'version': version, 'clique': clique} for clique, row in self.versions.iterrows() for version in row['versions']]
            ).set_index('version')
            self.track_ids = self.version2clique.index.to_list()
        else:
            self.track_ids = np.load(self.data_path.joinpath(f"splits/{self.data_split}_ids.npy"))

    def _load_cqt(self, track_id: str) -> torch.Tensor:
        filename = self.dataset_path.joinpath(self._make_file_path(track_id, self.file_ext))
        cqt_spectrogram = np.load(filename)
        return torch.from_numpy(cqt_spectrogram).float().unsqueeze(0)
    
    def _load_label(self, track_id: int) -> torch.Tensor:
        idx = self.version2clique.loc[track_id, 'clique']
        return F.one_hot(torch.tensor(idx), num_classes=self.num_classes).float(), idx


def cover_dataloader(
    data_path: str,
    file_ext: str,
    dataset_path: str,
    data_split: Literal["train", "val", "test"],
    batch_size: int,
    n_neg_samples: int,
    num_of_start_exploration: int = 0,
    exploration_prob: float = 0.5,
    l_mixup: Tuple[float, float] = (0, 1),
    l_cutmix: Tuple[float, float] = (0, 1),
    **config: Dict,
) -> DataLoader:
    
    dataset = CoverDataset(data_path,
                           file_ext,
                           dataset_path,
                           data_split,
                           n_neg_samples=n_neg_samples,
                           exploration_prob=exploration_prob,
                           l_mixup=l_mixup,
                           l_cutmix=l_cutmix,
                           num_of_start_exploration=num_of_start_exploration)
    num_samples = len(dataset)
    if data_split == "train":
        num_samples = ((int(config["epoch_size"] * len(dataset)) // batch_size) * batch_size) if config["epoch_size"] <= 1 else config["epoch_size"]
        vc = dataset.value_counts
        weights = 1 / vc
        weights = weights[dataset.version2clique.values].reshape(-1)
        sampler = WeightedRandomSampler(weights=weights,
                                        num_samples=num_samples,
                                        replacement=config["replacement"])
        config = {"sampler": sampler}
        
    return num_samples, DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        **config
    )
