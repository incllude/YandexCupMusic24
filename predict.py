from sklearn.metrics import pairwise_distances
from argparse import ArgumentParser
from utils import cover_dataloader
from modules import MainModule
from utils import load_config
from models import CNNModel
from tqdm import tqdm
import numpy as np
import torch


parser = ArgumentParser()
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--config_path', type=str, required=True)
args = parser.parse_args()


config = load_config(args.config_path)

model = CNNModel(timm_model="timm/convnext_nano.d1h_in1k",
                 dropout=0.0)

module = MainModule.load_from_checkpoint(args.ckpt_path,
                                         model=model,
                                         A=2,
                                         temperature=0.025,
                                         num_classes=-1,
                                         train_dataloader=None)

_, test_dl = cover_dataloader(
    data_path=config.data_path,
    dataset_path=config.data_path.joinpath("test/test"),
    file_ext=config.file_ext,
    batch_size=config.batch_size,
    n_neg_samples=1,
    data_split="test",
    shuffle=False
)

idxs = []
embeddings = []
module = module.eval()

with torch.no_grad():
    for batch in tqdm(test_dl):

        idxs.extend(batch["anchor_id"].tolist())
        embeddings.extend(module.forward(batch["anchor"].to(module.device)).cpu().numpy())
        
idxs = np.array(idxs)

distances = pairwise_distances(embeddings, metric="cosine")


top_100 = []
for i, dists in enumerate(tqdm(distances)):
    top = np.argsort(dists)[:100]
    top = top[top != i]
    top = np.insert(top, 0, i)
    top = idxs[top]
    top_100.append(top)

with open('submission.txt', 'w') as f:
    f.seek(0)
    f.truncate()
    for top in tqdm(top_100):
        f.write(f"{' '.join(map(str, top.tolist()))}\n")
