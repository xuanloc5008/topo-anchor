from __future__ import annotations

from pathlib import Path

import numpy as np


def project_latents_2d(latents: np.ndarray, *, method: str = "pca", random_state: int = 0) -> np.ndarray:
    latents = np.asarray(latents, dtype=np.float32)
    if latents.ndim != 2:
        raise ValueError(f"Expected latents [N, D], got {latents.shape}.")
    method = method.lower()
    if method == "pca":
        centered = latents - latents.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        return centered @ vh[:2].T
    if method == "umap":
        import umap

        return umap.UMAP(n_components=2, random_state=random_state).fit_transform(latents)
    if method == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto").fit_transform(
            latents
        )
    raise ValueError(f"Unsupported latent projection method: {method}")


def save_latent_scatter(
    latents_2d: np.ndarray,
    *,
    labels: list[str] | np.ndarray | None,
    output_path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    latents_2d = np.asarray(latents_2d)
    if latents_2d.ndim != 2 or latents_2d.shape[1] != 2:
        raise ValueError(f"Expected 2D latent coordinates [N, 2], got {latents_2d.shape}.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    if labels is None:
        ax.scatter(latents_2d[:, 0], latents_2d[:, 1], s=12)
    else:
        labels = np.asarray(labels)
        unique = sorted(set(labels.tolist()))
        for label in unique:
            mask = labels == label
            ax.scatter(latents_2d[mask, 0], latents_2d[mask, 1], s=12, label=str(label))
        ax.legend(loc="best", fontsize="small")
    ax.set_xlabel("latent-1")
    ax.set_ylabel("latent-2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
