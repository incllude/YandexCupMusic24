from sklearn.metrics import average_precision_score
from sklearn.metrics import pairwise_distances
import torch.nn.functional as F
import lightning as l
import numpy as np
import torch


class MainModule(l.LightningModule):
    
    def __init__(self,
                 model,
                 A: float,
                 num_classes: int,
                 temperature: float,
                 train_dataloader=None) -> None:
        super(MainModule, self).__init__()
        
        self.A = A
        self.temperature = temperature
        self.num_classes = num_classes
        self.train_dataloader = train_dataloader
        
        self.model = model
        
        self.train_outputs = []
        self.val_outputs = []
        
    def set_training_settings(self, optimizer_cfg, scheduler_cfg):
        
        try:
            optimizer = getattr(torch.optim, optimizer_cfg["type"])
        except:
            optimizer = eval(optimizer_cfg["type"])
        self._optimizer = lambda x: optimizer(x, **optimizer_cfg["params"])
        self._scheduler = lambda x: getattr(torch.optim.lr_scheduler, scheduler_cfg["type"])(x, **scheduler_cfg["params"])
    
    def _update_train_dist_matrix(self, logits, anchor_cliques, negative_cliques):
        
        cos_dists = logits.detach().cpu().numpy() * self.temperature + 1
        for clique_i, cos_dist, negative_clique in zip(anchor_cliques, cos_dists, negative_cliques):
            for cos, clique_j in zip(cos_dist, negative_clique):
                self.train_dataloader.dataset.update_neg_dist_matrix(clique_i, clique_j, cos)
                
    def _update_train_dist_vector(self, logits, anchor_cliques):
        
        cos_dists = 1 - logits.detach().cpu().numpy() * self.temperature
        for clique_i, cos in zip(anchor_cliques, cos_dists):
            self.train_dataloader.dataset.update_pos_dist_vector(clique_i, cos[0])
    
    @staticmethod
    def ndcg_at_k(relevance_scores):

        k = len(relevance_scores)
        dcg = np.sum(relevance_scores / np.sqrt(np.arange(k) + 1))
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = np.sum(ideal_relevance / np.sqrt(np.arange(k) + 1))

        return dcg / idcg if idcg > 0 else 0.0    
    
    def forward(self, x: torch.Tensor):
        return self.model.forward(x)
    
    def pos_neg_cos_product(self, pos_batch, neg_batch):
        
        pos_batch_exp = pos_batch.repeat(neg_batch.size(0), 1, 1)
        logits = F.cosine_similarity(pos_batch_exp, neg_batch, dim=-1).T / self.temperature
        
        return logits
    
    def training_step(self, batch):
        batch["anchor_clique"] = batch["anchor_clique"].cpu()
        batch["negative_clique"] = batch["negative_clique"].cpu()
        l1, l2 = 0, 0

        anchor_emb = self.forward(batch["anchor"])
        positive_emb = torch.stack([self.forward(x) for x in batch["positive"]], dim=0)
        negative_emb = torch.stack([self.forward(x) for x in batch["negative"]], dim=0)
        
        anc_pos_logits = self.pos_neg_cos_product(anchor_emb, positive_emb)[:, [0]]
        anc_neg_logits = self.pos_neg_cos_product(anchor_emb, negative_emb)
        logits = torch.cat((anc_pos_logits, anc_neg_logits), dim=-1)
        probs = F.softmax(logits, dim=-1)
        pos_probs, neg_probs = probs[:, 0], probs[:, 1:]
        labels = torch.ones_like(pos_probs)
        l1 = F.binary_cross_entropy(pos_probs, labels)
        l2 = torch.mean(torch.sum(neg_probs, dim=-1))
        loss = l1 + l2
        
        self._update_train_dist_matrix(anc_neg_logits, batch["anchor_clique"], batch["negative_clique"])
        self._update_train_dist_vector(anc_pos_logits, batch["anchor_clique"])
        self.log("Contrastive Loss", l1.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("ReverseCrossEntropy Loss", l2.item(), on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
                
        emb = self.forward(batch["anchor"]).cpu().numpy()
        cover = batch["anchor_clique"].cpu().numpy()
        self.val_outputs.extend(zip(emb, cover))
        
    def on_train_epoch_end(self):
        
        weights = self.train_dataloader.dataset.pos_dist_vector.astype(np.float64)
        weights /= weights.max()
        weights = weights[self.train_dataloader.dataset.version2clique.clique.values]
        self.train_dataloader.sampler.weights = torch.from_numpy(weights / weights.sum())  
        
    def on_validation_epoch_end(self):
        k = 100
        
        embeddings = np.stack(list(map(lambda x: x[0], self.val_outputs)))
        covers = np.stack(list(map(lambda x: x[1], self.val_outputs)))
        
        distances = pairwise_distances(embeddings, metric="cosine")
        avg_prc, ndcg = [], []
        for sample_distances, cover_idx in zip(distances, covers):
            covers_idxs = np.where(covers == cover_idx)
            sample_distances = np.argsort(sample_distances.copy())[1:][:k]
            relevance = np.isin(sample_distances, covers_idxs)
            ndcg.append(self.ndcg_at_k(relevance))
            if relevance.astype(int).sum() == 0:
                avg_prc.append(0)
            else:
                avg_prc.append(average_precision_score(relevance, np.arange(k)[::-1]))
            
        self.log("NDCG", np.mean(ndcg))
        self.log("MAP", np.mean(avg_prc))
        self.val_outputs.clear()
        
    def configure_optimizers(self):
        
        optimizer = self._optimizer(self.parameters())
        scheduler = self._scheduler(optimizer)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "NDCG",
                "interval": "epoch",
                "frequency": 1
            }
        }
