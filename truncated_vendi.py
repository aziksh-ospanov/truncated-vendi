"""
TruncatedVendi: A class to compute VENDI (exponentiated entropy) scores
using Gaussian (RBF) or cosine kernels, with optional Nyström approximation.
"""

import torch
import numbers
from typing import Optional
from sklearn.kernel_approximation import Nystroem


class TruncatedVendi:
    """
    A class to compute VENDI (exponentiated entropy) or truncated VENDI
    scores for a given feature matrix, using Gaussian (RBF) or cosine kernels.
    Supports full/exact kernel computation or Nyström approximation.

    Attributes:
        features (torch.Tensor): Feature matrix of shape (N, D).
    """

    def __init__(self, features: torch.Tensor):
        """
        Initialize the TruncatedVendi with a feature tensor.

        Args:
            features (torch.Tensor): Tensor of shape (N, D).
        """
        if not isinstance(features, torch.Tensor):
            raise TypeError("Features must be a torch.Tensor")
        if features.ndim != 2:
            raise ValueError("Features must be a 2D tensor of shape (N, D)")
        self.features = features

    def _normalized_gaussian_kernel(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sigma: float,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """
        Compute a normalized Gaussian (RBF) kernel matrix between x and y in batches.

        K[i, j] = exp(-||x[i] - y[j]||^2 / (2 * sigma^2)) / sqrt(N * M),
        where N = x.shape[0], M = y.shape[0].

        Args:
            x (torch.Tensor): Tensor of shape (N, D).
            y (torch.Tensor): Tensor of shape (M, D).
            sigma (float): Gaussian bandwidth parameter.
            batch_size (int, optional): Batch size for computing pairwise distances.

        Returns:
            torch.Tensor: Kernel matrix of shape (N, M), normalized by sqrt(N * M).
        """
        assert x.ndim == 2 and y.ndim == 2, "Inputs must be 2D tensors"
        assert x.shape[1] == y.shape[1], "Feature dimensions must match"

        n, m = x.shape[0], y.shape[0]
        inv_coeff = -1.0 / (2.0 * sigma * sigma)
        norm_factor = (n * m) ** 0.5
        device = x.device

        kernel_batches = []
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            y_batch = y[start:end]  # shape: (b, D)

            # Compute squared Euclidean distances: (N, b)
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
            x_norm_sq = (x ** 2).sum(dim=1, keepdim=True)           # (N, 1)
            y_norm_sq = (y_batch ** 2).sum(dim=1, keepdim=True).T    # (1, b)
            cross_term = x @ y_batch.T                               # (N, b)
            dist_sq = x_norm_sq + y_norm_sq - 2.0 * cross_term
            kernel_batch = torch.exp(inv_coeff * dist_sq)            # (N, b)
            kernel_batches.append(kernel_batch)

        K = torch.cat(kernel_batches, dim=1)  # shape: (N, M)
        return K.div_(norm_factor)

    def _cosine_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute a normalized cosine similarity kernel between x and y.

        K[i, j] = cosine_similarity(x[i], y[j]) / sqrt(N * M).

        Args:
            x (torch.Tensor): Tensor of shape (N, D).
            y (torch.Tensor): Tensor of shape (M, D).

        Returns:
            torch.Tensor: Kernel matrix of shape (N, M).
        """
        assert x.ndim == 2 and y.ndim == 2, "Inputs must be 2D tensors"

        # Normalize each vector to unit norm along dim=1
        x_norm = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
        y_norm = y / y.norm(dim=1, keepdim=True).clamp_min(1e-8)

        sim_matrix = x_norm @ y_norm.T  # shape: (N, M)
        n, m = x.shape[0], y.shape[0]
        return sim_matrix.div_((n * m) ** 0.5)

    def _nystrom_kernel(
        self,
        x_feats: torch.Tensor,
        kernel_name: str,
        n_components: int,
        sigma: Optional[float] = None,
        random_state: int = 1,
    ) -> torch.Tensor:
        """
        Compute an approximate kernel matrix via the Nyström method.

        Supports 'gaussian' (RBF) or 'cosine'.

        Args:
            x_feats (torch.Tensor): Data points shape (N, D). Must be convertible to numpy.
            kernel_name (str): One of ['gaussian', 'cosine'].
            n_components (int): Number of landmark points (<= N).
            sigma (float, optional): Bandwidth for Gaussian kernel.
            random_state (int, optional): Random seed for Nyström sampling.

        Returns:
            torch.Tensor: Approximated kernel matrix of shape (n_components, n_components),
                          normalized by dividing by N.
        """
        assert kernel_name in ("gaussian", "cosine"), "kernel_name must be 'gaussian' or 'cosine'"
        N, _ = x_feats.shape
        n_components = min(n_components, N)

        if kernel_name == "gaussian":
            assert isinstance(sigma, numbers.Number), "sigma must be provided for Gaussian kernel"
            gamma = 1.0 / (2.0 * sigma * sigma)  # sklearn expects gamma = 1/(2σ²)
            sklearn_kernel = "rbf"
        else:
            gamma = None
            sklearn_kernel = "cosine"

        # Convert to numpy for sklearn
        X_np = x_feats.cpu().numpy()
        nystroem = Nystroem(
            kernel=sklearn_kernel,
            gamma=gamma,
            n_components=n_components,
            random_state=random_state,
        )
        transformed = nystroem.fit_transform(X_np)  # shape: (N, n_components)

        # Compute low-rank approximation: (n_components, n_components)
        K_approx = transformed.T @ transformed  # numpy array
        K_approx = K_approx / float(N)
        return torch.from_numpy(K_approx)

    def _calculate_stats(
        self,
        eigenvalues: torch.Tensor,
        alpha: float = 2.0,
        truncation: Optional[int] = None,
    ) -> float:
        """
        Compute VENDI (exponentiated entropy) or truncated VENDI from eigenvalues.

        If alpha == 1: Shannon entropy; otherwise, Rényi entropy of order alpha.

        If truncation is provided and < len(eigenvalues), we add the tail mass uniformly.

        Args:
            eigenvalues (torch.Tensor): 1D tensor of eigenvalues (must be >= 0).
            alpha (float, optional): Order of Rényi entropy; alpha=1 uses Shannon.
            truncation (int, optional): Number of top eigenvalues to keep (t < len).

        Returns:
            float: Rounded VENDI score (to two decimals).
        """
        eps = 1e-10
        ev = eigenvalues.clamp(min=eps)
        ev, _ = ev.sort(descending=True)

        if isinstance(truncation, int) and 0 < truncation < ev.numel():
            top = ev[:truncation]
            tail_mass = 1.0 - top.sum()
            top = top + (tail_mass / truncation)
            ev = top

        log_ev = ev.log()
        if abs(alpha - 1.0) < 1e-8:
            # Shannon entropy
            entropy = - (ev * log_ev).sum()
        else:
            # Rényi entropy: (1 / (1 - alpha)) * log(sum(ev^alpha))
            entropy = (1.0 / (1.0 - alpha)) * (ev.pow(alpha).sum().log())

        vendi = torch.exp(entropy)
        return vendi.item()

    def compute_score(
        self,
        alpha: float,
        truncation: Optional[int] = None,
        sigma: Optional[float] = None,
        kernel: str = "gaussian",
        use_nystrom: bool = False,
        batch_size: int = 64,
    ) -> float:
        """
        Compute the VENDI or truncated VENDI score for the stored feature matrix,
        using the specified kernel and parameters.

        Args:
            alpha (float): Entropy order (1 for Shannon, >1 for Rényi).
            truncation (int, optional): Number of top eigenvalues to keep.
            sigma (float, optional): Bandwidth for Gaussian kernel.
            kernel (str, optional): One of ['gaussian', 'cosine'].
            use_nystrom (bool, optional): If True, use Nyström approximation.
            batch_size (int, optional): Batch size for computing Gaussian kernel.

        Returns:
            float: VENDI score (rounded to two decimals).
        """
        assert kernel in ("gaussian", "cosine"), "kernel must be 'gaussian' or 'cosine'"

        if use_nystrom and isinstance(truncation, int):
            K = self._nystrom_kernel(
                x_feats=self.features,
                kernel_name=kernel,
                n_components=truncation,
                sigma=sigma,
            )
        else:
            if kernel == "gaussian":
                assert isinstance(sigma, numbers.Number), "sigma must be provided for Gaussian kernel"
                K = self._normalized_gaussian_kernel(
                    self.features, self.features, sigma, batch_size
                )
            else:  # "cosine"
                K = self._cosine_kernel(self.features, self.features)

        # Ensure symmetry / numerical stability
        K = (K + K.T) / 2.0
        eigenvals = torch.linalg.eigvalsh(K)
        return self._calculate_stats(eigenvals, alpha=alpha, truncation=truncation)