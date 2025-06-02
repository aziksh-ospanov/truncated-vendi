# Truncated Vendi Score Implementation

[Paper: On the Statistical Complexity of Estimating Vendi Scores from Empirical Data](https://arxiv.org/abs/2410.21719)
 
This repository provides an implementation of the truncated Vendi score, an entropy‐based diversity metric on kernel eigenvalues, along with an efficient Nyström‐approximation variant. It serves as the official code companion for the “On the Statistical Complexity of Estimating Vendi Scores from Empirical Data” paper.

## Abstract
Abstract: Evaluating the diversity of generative models without access to reference data poses methodological challenges. The reference-free Vendi score offers a solution by quantifying the diversity of generated data using matrix-based entropy measures. The Vendi score is usually computed via the eigendecomposition of an $n \times n$ kernel matrix for $n$ generated samples. However, the heavy computational cost of eigendecomposition for large $n$ often limits the sample size used in practice to a few tens of thousands. In this paper, we investigate the statistical convergence of the Vendi score. We numerically demonstrate that for kernel functions with an infinite feature map dimension, the score estimated from a limited sample size may exhibit a non-negligible bias relative to the population Vendi score, i.e., the asymptotic limit as the sample size approaches infinity. To address this, we introduce a truncation of the Vendi statistic, called the $t$-truncated Vendi statistic, which is guaranteed to converge to its asymptotic limit given $n=O(t)$ samples. We show that the existing Nyström method and the FKEA approximation method for approximating the Vendi score both converge to the population truncated Vendi score. We perform several numerical experiments to illustrate the concentration of the Nyström and FKEA-computed Vendi scores around the truncated Vendi and discuss how the truncated Vendi score correlates with the diversity of image and text data.

## Example Usage - Computing Original Vendi Score (with $\alpha=1$)
```python
from truncated_vendi import TruncatedVendi
import torch

features = torch.randn(2000, 768)   # 2,000 samples; 768-dim embeddings
vendi = TruncatedVendi(features)

score = vendi.compute_score(
    alpha=1.0,
    kernel="gaussian",
    sigma=5.0,
    use_nystrom=False,
    batch_size=128,
)
print(f"Vendi [alpha=1] (RBF): {score:.2f}")
```

## Example Usage - Computing Truncated Vendi Score (with $\alpha=1$)
```python
from truncated_vendi import TruncatedVendi
import torch

features = torch.randn(2000, 768)   # 2,000 samples; 768-dim embeddings
vendi = TruncatedVendi(features)

score = vendi.compute_score(
    alpha=1.0,
    truncation=500,     # Keep top 500 eigenvalues
    kernel="gaussian",
    sigma=5.0,
    use_nystrom=False,
    batch_size=128,
)
print(f"Vendi [alpha=1] [t=500] (RBF): {score:.2f}")
```

## Example Usage - Computing Efficient Nystorm-based Truncated Vendi Score (with $\alpha=2$)
```python
from truncated_vendi import TruncatedVendi
import torch

features = torch.randn(2000, 768)   # 2,000 samples; 768-dim embeddings
vendi = TruncatedVendi(features)

score = vendi.compute_score(
    alpha=2.0,
    truncation=500,     # Keep top 500 eigenvalues
    kernel="gaussian",
    sigma=5.0,
    use_nystrom=True,
    batch_size=128,
)
print(f"Vendi [alpha=2] [t=500] (RBF-Nystrom): {score:.2f}")
```


## Notes
- Ensure that loaded features are of dimension (n,d), where n is the number of evaluated samples and d is the embedding dimension
- The script is compatible with any embedding space across image, text, video and audio modailities

## Cite our work
