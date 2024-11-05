from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from utils import cover_dataloader, Config
from modules import MainModule
from dotenv import load_dotenv
from models import CNNModel
import lightning as l
import wandb
import os


load_dotenv()
WANDB_KEY = os.getenv("WANDB_KEY")
if not WANDB_KEY:
    raise ValueError("WANDB_KEY not found in .env file")


config = Config()
config.seed_everything()
wandb.login(key=WANDB_KEY)


config.epoch_size, train_dl = cover_dataloader(
    data_path=config.data_path,
    dataset_path=config.data_path.joinpath("train/train"),
    file_ext=config.file_ext,
    batch_size=config.batch_size,
    data_split="train",
    n_neg_samples=config.n_neg_samples,
    epoch_size=config.epoch_size,
    replacement=False,
    l_mixup=(0.4, 0.6),
    l_cutmix=(0.4, 0.6),
    exploration_prob=0.125,
    num_of_start_exploration=200*300
)

_, val_dl = cover_dataloader(
    data_path=config.data_path,
    dataset_path=config.data_path.joinpath("train/train"),
    file_ext=config.file_ext,
    batch_size=config.batch_size,
    data_split="val",
    n_neg_samples=1,
    shuffle=False
)

num_classes = train_dl.dataset.num_classes


wandb_logger = WandbLogger(project="YaCup 2024",
                           name="GOAL")
checkpoint_callback = ModelCheckpoint(
    dirpath="./",
    filename="model_epoch={epoch:02}_ndcg={NDCG:.3f}",
    verbose=False,
    every_n_epochs=1,
    save_top_k=1,
    monitor="NDCG",
    mode="max"
)
lr_monitor = LearningRateMonitor(logging_interval='epoch')

model = CNNModel(timm_model="timm/convnext_nano.d1h_in1k",
                 dropout=0.0)

module = MainModule(model=model,
                    A=2,
                    temperature=0.025,
                    num_classes=num_classes,
                    train_dataloader=train_dl)
module.set_training_settings(optimizer_cfg={"type"  : "Adam",
                                            "params": {"lr": 0.0001,
                                                       "weight_decay": 0.}},
                             scheduler_cfg={"type"  : "ReduceLROnPlateau",
                                            "params": {"mode": "max",
                                                       "factor": 1/3,
                                                       "patience": 0,
                                                       "threshold": 0.02,
                                                       "threshold_mode": "rel",
                                                       "cooldown": 1}})

trainer = l.Trainer(accelerator="cuda",
                    max_epochs=config.epochs,
                    enable_progress_bar=True,
                    log_every_n_steps=config.epoch_size//config.batch_size,
                    logger=wandb_logger,
                    num_sanity_val_steps=0,
                    callbacks=[checkpoint_callback, lr_monitor])


trainer.fit(model=module, train_dataloaders=train_dl, val_dataloaders=val_dl)
wandb.finish()
