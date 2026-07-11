"""Numerically careful linear-algebra helpers used throughout the belief memory."""
import torch


def symmetrize(mat: torch.Tensor) -> torch.Tensor:
    """Force symmetry after updates that accumulate float error."""
    return 0.5 * (mat + mat.transpose(-1, -2))


def add_jitter(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Add eps * I to the last two dims."""
    eye = torch.eye(mat.shape[-1], dtype=mat.dtype, device=mat.device)
    return mat + eps * eye


def chol_logdet(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """log|mat| via Cholesky. mat: (..., p, p) SPD."""
    chol = torch.linalg.cholesky(add_jitter(symmetrize(mat), eps))
    return 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)


def mahalanobis_sq(err: torch.Tensor, cov: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """err: (..., p), cov: (..., p, p) SPD. Returns e^T cov^{-1} e, shape (...)."""
    chol = torch.linalg.cholesky(add_jitter(symmetrize(cov), eps))
    sol = torch.cholesky_solve(err.unsqueeze(-1), chol)
    return (err.unsqueeze(-1) * sol).sum(dim=(-1, -2))


def gaussian_nll(err: torch.Tensor, cov: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Negative log-likelihood of residual `err` under N(0, cov), per element."""
    p = err.shape[-1]
    d2 = mahalanobis_sq(err, cov, eps)
    logdet = chol_logdet(cov, eps)
    return 0.5 * (d2 + logdet + p * torch.log(torch.tensor(2.0 * torch.pi)))
