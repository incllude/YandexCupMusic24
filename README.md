# Yandex Cup 2024 — ML: Music information retrieval

## Task
Create an algorithm that will find variations and covers that are closest to the original composition

## Data
The tracks are presented through a variation of the CQT spectrogram compressed in time dimension, which we built using 60 seconds taken from the central part of the track. So spectrogram's shape is $(84, 50)$

## Eval Metric

**Eval Metric** — $nDCG@100$, where $DCG@100 = \sum_{i=1}^{100} \frac{2^{rel_i} - 1}{\sqrt{i}}$, $rel_i \in \{0, 1\}$

## Proposed method

**Backbone** — $\text{ConvNeXt Nano}$ pretrained on $\text{ImageNet-1k}$

**Loss** — $\text{Contrastive Loss} + \text{Symmetric Cross Entropy}$

## Result

**Place** — $70/170$

$nDCG@100 = 0.21506$
