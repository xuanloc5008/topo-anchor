# Topology-Anchored, Context-Modulated MRI Segmentation

A professional research-to-production project layout for MRI segmentation with a **topology-anchored, context-modulated** latent prior and an explicit **topology tooling stack**.

This version fixes the mathematical notation so that all symbols are consistent across the model, loss functions, training workflow, inference workflow, and topology modules.

---

## 0. Notation convention

We use bold symbols for images, masks, tensors, vectors, and maps.

| Symbol | Meaning |
|---|---|
| $\mathbf{x}_i$ | MRI image or volume of sample $i$ |
| $\mathbf{y}_i$ | ground-truth segmentation mask of sample $i$ |
| $\hat{\mathbf{p}}_i$ | predicted probability map/logit-derived probability map |
| $\hat{\mathbf{y}}_i$ | final discrete predicted mask after threshold or argmax |
| $\mathbf{h}_i$ | shared deep visual representation from the encoder/bottleneck |
| $\mathbf{z}^{\mathrm{topo}}_i$ | topology-specific latent embedding |
| $\mathbf{c}^{\mathrm{app}}_i$ | residual context / appearance code |
| $\bar{\mathbf{z}}^{\mathrm{topo}}_i$ | detached topology code used by anchor heads, $\bar{\mathbf{z}}^{\mathrm{topo}}_i = \operatorname{sg}(\mathbf{z}^{\mathrm{topo}}_i)$ |
| $\boldsymbol{\mu}_i$ | topology-anchor center for sample $i$ |
| $\boldsymbol{\sigma}_i^2$ | diagonal anchor variance/spread for sample $i$ |
| $q_i(\mathbf{z})$ | Gaussian anchor density for sample $i$ |
| $\mathbf{T}^{\mathrm{anc}}_i$ | anchor-token set injected into D3 cross-attention |
| $\boldsymbol{\tau}_i$ | topology/morphology descriptor extracted from $\mathbf{y}_i$ |
| $\mathcal{P}_i$ | positive index set for anchor sample $i$ |
| $\mathcal{N}_i$ | negative index set for anchor sample $i$ |
| $\tau_m$ | temperature of metric/contrastive loss |
| $\epsilon_\sigma, \epsilon_d$ | small constants for numerical stability |

Important notation rules:

1. We use $\hat{\mathbf{p}}_i$ for the predicted probability map, not $p_i$, to avoid confusion with probability density.
2. We use $q_i(\mathbf{z})$ for the sample-wise Gaussian anchor density, not $p(z)$, to avoid overloading $p$.
3. We use $\mathcal{P}_i$ for the positive set, not $P(i)$, to avoid conflict with predicted probability maps.
4. In the distribution loss, the anchor heads receive $\bar{\mathbf{z}}^{\mathrm{topo}}_i = \operatorname{sg}(\mathbf{z}^{\mathrm{topo}}_i)$, where $\operatorname{sg}(\cdot)$ denotes stop-gradient / detach. This keeps the anchor topology-driven while reducing a trivial identity shortcut $\boldsymbol{\mu}_i \approx \mathbf{z}^{\mathrm{topo}}_i$.

---

## 1. Executive summary

### Final modeling choice

We adopt the compact but still **topology-anchored, context-modulated** design.

For each sample $i$, the encoder produces a shared representation:

$$
\mathbf{h}_i = E_\theta(\mathbf{x}_i).
$$

Then the model derives two codes:

$$
\mathbf{z}^{\mathrm{topo}}_i = H_{\psi}^{\mathrm{topo}}(\mathbf{h}_i),
\qquad
\mathbf{c}^{\mathrm{app}}_i = C_{\xi}(\mathbf{h}_i).
$$

The topology-anchor center is predicted only from the topology branch:

$$
\boldsymbol{\mu}_i = A_{\mu}(\bar{\mathbf{z}}^{\mathrm{topo}}_i),
\qquad
\bar{\mathbf{z}}^{\mathrm{topo}}_i = \operatorname{sg}(\mathbf{z}^{\mathrm{topo}}_i).
$$

The anchor spread is predicted from topology plus residual context:

$$
\boldsymbol{\rho}_i = A_{\sigma}\!\left([\bar{\mathbf{z}}^{\mathrm{topo}}_i,\mathbf{c}^{\mathrm{app}}_i]\right),
\qquad
\boldsymbol{\sigma}_i^2 = \operatorname{softplus}(\boldsymbol{\rho}_i)+\epsilon_\sigma.
$$

The corresponding sample-wise anchor density is:

$$
q_i(\mathbf{z})=
\mathcal{N}\!\left(\mathbf{z};\boldsymbol{\mu}_i,\operatorname{diag}(\boldsymbol{\sigma}_i^2)\right).
$$

This means:

- the **anchor center** is topology-driven
- the **anchor spread** is context-modulated
- the decoder is conditioned by anchor tokens built from $\mathbf{z}^{\mathrm{topo}}_i$, $\boldsymbol{\mu}_i$, and $\boldsymbol{\sigma}_i^2$

### Final training objective

We keep the training objective compact:

$$
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{seg}}
+
\lambda_{\mathrm{metric}}\mathcal{L}_{\mathrm{metric}}
+
\lambda_{\mathrm{dist}}\mathcal{L}_{\mathrm{dist}}.
$$

where:

- $\mathcal{L}_{\mathrm{seg}}$: segmentation loss
- $\mathcal{L}_{\mathrm{metric}}$: topology-aware metric loss on $\mathbf{z}^{\mathrm{topo}}$
- $\mathcal{L}_{\mathrm{dist}}$: topology-anchor regularization loss

### Topology stack policy

This project uses topology libraries in a modular way:

- **scikit-image**: lightweight topology/morphology descriptors, connected components, Euler-style proxies, contour hierarchy helpers, region properties
- **GUDHI**: persistent homology on masks/logits using cubical complexes, bottleneck/Wasserstein-style distances when needed
- **giotto-tda**: persistence vectorization and pipeline-friendly transforms such as persistence images, landscapes, silhouettes, entropy

The model core does **not** require direct online topology computation at inference. Topology is used for:

- preprocessing-time descriptor generation
- training-time pair/prototype building
- optional topology analysis and evaluation
- optional topology-explicit extensions later

---

## 2. Problem setting

This project targets MRI segmentation under appearance and domain shift, with emphasis on:

- robustness across scanner/vendor/style variation
- morphology- and topology-aware latent structuring
- topology-anchored plausibility estimation
- confidence calibration at inference

The core design question is:

> How do we keep the model sensitive to **structural differences** while reducing sensitivity to **appearance-only shifts**?

The answer is:

1. learn a topology-focused latent code $\mathbf{z}^{\mathrm{topo}}$
2. build a topology-anchored latent prior with center $\boldsymbol{\mu}$
3. let residual context $\mathbf{c}^{\mathrm{app}}$ modulate anchor spread $\boldsymbol{\sigma}^2$, not anchor center
4. condition the deep decoder with anchor tokens
5. supervise latent structure using topology-informed pair construction from ground-truth masks

---

## 3. Final architecture overview

### 3.1 High-level pipeline

```text
Input MRI x_i
  -> preprocessing
  -> nnU-Net encoder E1 -> E2 -> E3 -> E4
  -> Res-Mamba @ E4 / bottleneck
  -> shared representation h_i

h_i -> topology projection head H_topo -> z_topo_i
h_i -> context residual encoder C_app -> c_app_i

sg(z_topo_i) -> mean head A_mu -> mu_i
[sg(z_topo_i), c_app_i] -> variance head A_sigma -> sigma_i^2

(z_topo_i, mu_i, sigma_i^2) -> anchor-token generator -> T_anc_i
(h_i, T_anc_i) -> decoder with Res-Mamba @ D3 + Cross-Attention @ D3 -> p_hat_i
```

### 3.2 Design principle

This is **topology-anchored, context-modulated**:

- **topology** decides where the sample should be anchored
- **context** decides how tightly or loosely the sample should be anchored

Formally:

$$
\mathbf{z}^{\mathrm{topo}}_i = H^{\mathrm{topo}}_{\psi}(\mathbf{h}_i),
\qquad
\mathbf{c}^{\mathrm{app}}_i = C_{\xi}(\mathbf{h}_i).
$$

$$
\boldsymbol{\mu}_i=A_{\mu}(\operatorname{sg}(\mathbf{z}^{\mathrm{topo}}_i)),
\qquad
\boldsymbol{\sigma}_i^2=
\operatorname{softplus}\!\left(A_{\sigma}([\operatorname{sg}(\mathbf{z}^{\mathrm{topo}}_i),\mathbf{c}^{\mathrm{app}}_i])\right)+\epsilon_\sigma.
$$

---

## 4. Detailed architectural breakdown

## 4.1 Input and preprocessing

### Input

Each training sample is:

$$
(\mathbf{x}_i,\mathbf{y}_i), \qquad i\in\{1,\ldots,B\}.
$$

where:

- $\mathbf{x}_i$: MRI slice or volume
- $\mathbf{y}_i$: segmentation ground-truth mask

### Preprocessing goals

- unify orientation
- unify spacing / resolution
- normalize intensities
- crop to clinically relevant ROI
- verify image-mask pairing
- generate manifests and cached descriptors

### Recommended preprocessing stages

1. discover DICOM/NIfTI files
2. validate shape/orientation/spacing
3. reorient to canonical coordinate system
4. resample image and mask to target spacing
5. optional bias-field correction and clipping
6. intensity normalization
7. ROI extraction / foreground cropping
8. write dataset manifests
9. optionally precompute topology descriptors from GT masks

---

## 4.2 Encoder backbone

Use an nnU-Net-style backbone because it provides:

- a strong and reliable segmentation baseline
- clean skip connections
- compatibility with medical imaging tooling
- easy ablation against standard baselines

### Encoder stages

- `E1`: shallow texture and edge features
- `E2`: lower-resolution semantic features
- `E3`: mid-level structural features
- `E4`: deepest encoder feature map

### Bottleneck refinement

At the deepest stage, apply:

```text
Res-Mamba @ E4 / Bottleneck
```

This enriches the bottleneck with longer-range context before the representation branches split.

---

## 4.3 Shared representation $\mathbf{h}_i$

After the encoder and bottleneck refinement:

$$
\mathbf{h}_i = E_{\theta}(\mathbf{x}_i).
$$

$\mathbf{h}_i$ is the shared deep visual representation used by:

- the topology latent branch
- the context residual branch
- the decoder path

### What $\mathbf{h}_i$ contains

$\mathbf{h}_i$ is still a mixed representation. It may contain:

- anatomy semantics
- morphology cues
- global spatial organization
- style/vendor remnants
- noise and difficulty cues

Because $\mathbf{h}_i$ is mixed, we do **not** use it directly as the anchor representation.

---

## 4.4 Topology Projection Head $H^{\mathrm{topo}}_{\psi}$

### Definition

$$
\mathbf{z}^{\mathrm{topo}}_i = H^{\mathrm{topo}}_{\psi}(\mathbf{h}_i).
$$

### Design goal

Produce a compact latent code that is:

- more structural than stylistic
- topology- and morphology-sensitive
- stable under appearance-preserving shifts
- suitable for metric learning and anchor construction

### Recommended design

A compact design is:

1. global pooling over $\mathbf{h}_i$
2. 2-layer MLP projection
3. optional $\ell_2$-normalization before metric loss

### Practical implementation

```text
Input: h_i [B, C, H, W] or [B, C, H, W, D]
Pool: GAP(h_i) and GMP(h_i)
Concat: u_topo_i = [GAP(h_i), GMP(h_i)]
MLP: Linear -> LayerNorm -> GELU -> Dropout -> Linear
Output: z_topo_i [B, d_z]
Optional: L2 normalization for metric learning
```

### Suggested dimensions

If bottleneck channels = 256:

- pooled feature size: 512 with GAP+GMP
- hidden size: 256
- output size: 128

### Why it can learn topology-aware structure

The head does not know topology a priori. It becomes topology-aware because its output is optimized by:

- $\mathcal{L}_{\mathrm{metric}}$, which groups samples with similar topology/morphology
- $\mathcal{L}_{\mathrm{dist}}$, which regularizes the latent against a topology-driven anchor
- indirect gradients from $\mathcal{L}_{\mathrm{seg}}$, because anchor tokens affect decoder behavior

---

## 4.5 Context Residual Encoder $C_{\xi}$

### Definition

$$
\mathbf{c}^{\mathrm{app}}_i = C_{\xi}(\mathbf{h}_i).
$$

### Design goal

Capture residual context that should **not** define anchor center, such as:

- vendor/scanner appearance
- intensity style
- noise regime
- acquisition artifacts
- residual uncertainty cues

### Recommended design

Use a branch parallel to the topology head, but conceptually distinct:

```text
Input: h_i
Pool: GAP(h_i) or GAP+GMP
MLP: Linear -> LayerNorm -> GELU -> Dropout -> Linear
Output: c_app_i [B, d_app]
```

### Key rule

$\mathbf{c}^{\mathrm{app}}_i$ must **not** be used to determine the anchor center. It only modulates the anchor spread.

---

## 4.6 Topology-Anchored, Context-Modulated Anchor Distribution

This is the defining block of the project.

### Mean head

$$
\boldsymbol{\mu}_i = A_{\mu}(\bar{\mathbf{z}}^{\mathrm{topo}}_i),
\qquad
\bar{\mathbf{z}}^{\mathrm{topo}}_i = \operatorname{sg}(\mathbf{z}^{\mathrm{topo}}_i).
$$

- input: detached topology latent only
- output: topology anchor center

### Variance head

$$
\boldsymbol{\rho}_i = A_{\sigma}\!\left([\bar{\mathbf{z}}^{\mathrm{topo}}_i,\mathbf{c}^{\mathrm{app}}_i]\right),
\qquad
\boldsymbol{\sigma}_i^2=\operatorname{softplus}(\boldsymbol{\rho}_i)+\epsilon_\sigma.
$$

- input: detached topology latent + residual context
- output: positive diagonal anchor spread

### Interpretation

- $\boldsymbol{\mu}_i$: where sample $i$ should be anchored in latent topology space
- $\boldsymbol{\sigma}_i^2$: how much deviation is acceptable around that center

### Resulting Gaussian anchor density

$$
q_i(\mathbf{z})
=
\mathcal{N}\!\left(\mathbf{z};\boldsymbol{\mu}_i,\operatorname{diag}(\boldsymbol{\sigma}_i^2)\right).
$$

This gives a **topology-anchored latent prior** with **context-modulated uncertainty**.

---

## 4.7 Anchor-Token Generator $G_{\eta}$

From:

$$
\mathbf{z}^{\mathrm{topo}}_i,
\quad
\boldsymbol{\mu}_i,
\quad
\boldsymbol{\sigma}_i^2,
$$

produce:

$$
\mathbf{T}^{\mathrm{anc}}_i
=G_{\eta}\!\left(\mathbf{z}^{\mathrm{topo}}_i,\boldsymbol{\mu}_i,\boldsymbol{\sigma}_i^2\right),
$$

with tokens:

$$
\mathbf{T}^{\mathrm{anc}}_i
=
\{\mathbf{t}^{\mathrm{topo}}_i,\mathbf{t}^{\mu}_i,\mathbf{t}^{\sigma}_i\}.
$$

### Token semantics

- $\mathbf{t}^{\mathrm{topo}}_i$: token of the current topology latent state
- $\mathbf{t}^{\mu}_i$: token of the topology anchor center
- $\mathbf{t}^{\sigma}_i$: token of the anchor spread

These tokens are how latent priors are injected into the decoder.

---

## 4.8 Decoder with Res-Mamba @ D3 and Cross-Attention @ D3

The decoder follows a standard nnU-Net/U-Net style path with skip connections from:

- E3 -> D3
- E2 -> D2
- E1 -> D1

### Special deep decoder block

At D3:

1. fuse skip feature from E3 with bottleneck feature
2. apply `Res-Mamba @ D3`
3. apply `Cross-Attention @ D3` with anchor tokens

Let $\mathbf{f}^{\mathrm{rm}}_{D3,i}$ denote the feature after D3 Res-Mamba. Then:

$$
\mathbf{Q}_i = \operatorname{Proj}_{Q}(\mathbf{f}^{\mathrm{rm}}_{D3,i}),
$$

$$
\mathbf{K}_i = \operatorname{Proj}_{K}(\mathbf{T}^{\mathrm{anc}}_i),
\qquad
\mathbf{V}_i = \operatorname{Proj}_{V}(\mathbf{T}^{\mathrm{anc}}_i).
$$

$$
\operatorname{CA}_i
=
\operatorname{softmax}\!\left(\frac{\mathbf{Q}_i\mathbf{K}_i^{\top}}{\sqrt{d_k}}\right)\mathbf{V}_i.
$$

$$
\tilde{\mathbf{f}}_{D3,i}
=
\mathbf{f}^{\mathrm{rm}}_{D3,i}+\operatorname{CA}_i.
$$

### Why D3 only?

D3 is deep enough to carry structural semantics, but not so compressed that spatial detail has vanished. It is the best trade-off point for conditioning the decoder with topology-aware anchor tokens.

---

## 4.9 Output

The decoder produces a probability map:

$$
\hat{\mathbf{p}}_i = D_{\phi}(\mathbf{h}_i,\mathbf{T}^{\mathrm{anc}}_i).
$$

At inference, $\hat{\mathbf{p}}_i$ is converted into the final discrete prediction $\hat{\mathbf{y}}_i$ by thresholding or argmax.

---

## 5. Training workflow: step-by-step

Assume a mini-batch:

$$
\mathcal{B}=\{(\mathbf{x}_i,\mathbf{y}_i)\}_{i=1}^{B}.
$$

### Step 1: preprocess inputs

Each $\mathbf{x}_i$ is normalized, resampled, and cropped to ROI.

### Step 2: backbone forward

$$
\mathbf{x}_i \rightarrow \mathbf{h}_i.
$$

### Step 3: branch split

$$
\mathbf{h}_i \rightarrow \mathbf{z}^{\mathrm{topo}}_i,
\qquad
\mathbf{h}_i \rightarrow \mathbf{c}^{\mathrm{app}}_i.
$$

### Step 4: anchor distribution

$$
\bar{\mathbf{z}}^{\mathrm{topo}}_i = \operatorname{sg}(\mathbf{z}^{\mathrm{topo}}_i),
$$

$$
\boldsymbol{\mu}_i=A_{\mu}(\bar{\mathbf{z}}^{\mathrm{topo}}_i),
$$

$$
\boldsymbol{\sigma}_i^2=
\operatorname{softplus}\!\left(A_{\sigma}([\bar{\mathbf{z}}^{\mathrm{topo}}_i,\mathbf{c}^{\mathrm{app}}_i])\right)+\epsilon_\sigma.
$$

### Step 5: anchor tokens

$$
(\mathbf{z}^{\mathrm{topo}}_i,\boldsymbol{\mu}_i,\boldsymbol{\sigma}_i^2)
\rightarrow
\mathbf{T}^{\mathrm{anc}}_i.
$$

### Step 6: decoder

$$
(\mathbf{h}_i,\mathbf{T}^{\mathrm{anc}}_i)
\rightarrow
\hat{\mathbf{p}}_i.
$$

### Step 7: topology supervision branch, training only

From each ground-truth mask:

$$
\boldsymbol{\tau}_i = \mathcal{T}(\mathbf{y}_i).
$$

The pair/prototype builder receives $\{\boldsymbol{\tau}_i\}_{i=1}^{B}$ and constructs:

$$
\mathcal{P}_i = \{r\neq i:\boldsymbol{\tau}_r\text{ is similar to }\boldsymbol{\tau}_i\},
$$

$$
\mathcal{N}_i = \{r\neq i:\boldsymbol{\tau}_r\text{ is dissimilar to }\boldsymbol{\tau}_i\}.
$$

### Step 8: compute losses

#### Segmentation loss

$$
\mathcal{L}_{\mathrm{seg}}
=
\frac{1}{B}\sum_{i=1}^{B}
\left[
\mathcal{L}_{\mathrm{Dice}}(\hat{\mathbf{p}}_i,\mathbf{y}_i)
+
\lambda_{\mathrm{ce}}\mathcal{L}_{\mathrm{CE/BCE}}(\hat{\mathbf{p}}_i,\mathbf{y}_i)
\right].
$$

#### Metric loss

For supervised contrastive learning on $\mathbf{z}^{\mathrm{topo}}$:

$$
\mathcal{L}_{\mathrm{metric}}
=
\frac{1}{B}\sum_{i=1}^{B}
\left[
-
\frac{1}{|\mathcal{P}_i|}
\sum_{r\in\mathcal{P}_i}
\log
\frac{
\exp\left(\operatorname{sim}(\mathbf{z}^{\mathrm{topo}}_i,\mathbf{z}^{\mathrm{topo}}_r)/\tau_m\right)
}{
\sum_{a\in\mathcal{B}_{i}^{-}}
\exp\left(\operatorname{sim}(\mathbf{z}^{\mathrm{topo}}_i,\mathbf{z}^{\mathrm{topo}}_a)/\tau_m\right)
}
\right],
$$

where:

$$
\mathcal{B}_{i}^{-}=\{1,\ldots,B\}\setminus\{i\}.
$$

If $\mathcal{P}_i$ is empty for a sample, skip that sample in $\mathcal{L}_{\mathrm{metric}}$ or use a topology-aware sampler to avoid empty positives.

#### Distribution loss

The diagonal Gaussian negative log-likelihood is:

$$
\mathcal{L}_{\mathrm{dist}}
=
\frac{1}{B}\sum_{i=1}^{B}
\frac{1}{2}\sum_{j=1}^{d_z}
\left[
\frac{(z^{\mathrm{topo}}_{i,j}-\mu_{i,j})^2}{\sigma^2_{i,j}+\epsilon_d}
+
\log(\sigma^2_{i,j}+\epsilon_d)
\right].
$$

The constant $\frac{d_z}{2}\log(2\pi)$ is omitted because it does not affect optimization.

### Step 9: total loss

$$
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{seg}}
+
\lambda_{\mathrm{metric}}\mathcal{L}_{\mathrm{metric}}
+
\lambda_{\mathrm{dist}}\mathcal{L}_{\mathrm{dist}}.
$$

### Step 10: backpropagation

The total loss updates:

- backbone
- topology projection head
- context residual encoder
- mean head
- variance head
- anchor-token generator
- decoder

---

## 6. Inference workflow

At inference, there is no ground-truth mask and no topology pair building.

### Step 1

$$
\mathbf{x}_i \rightarrow \mathbf{h}_i.
$$

### Step 2

$$
\mathbf{h}_i \rightarrow \mathbf{z}^{\mathrm{topo}}_i,
\qquad
\mathbf{h}_i \rightarrow \mathbf{c}^{\mathrm{app}}_i.
$$

### Step 3

$$
\bar{\mathbf{z}}^{\mathrm{topo}}_i = \operatorname{sg}(\mathbf{z}^{\mathrm{topo}}_i),
$$

$$
\boldsymbol{\mu}_i=A_{\mu}(\bar{\mathbf{z}}^{\mathrm{topo}}_i),
\qquad
\boldsymbol{\sigma}_i^2=
\operatorname{softplus}\!\left(A_{\sigma}([\bar{\mathbf{z}}^{\mathrm{topo}}_i,\mathbf{c}^{\mathrm{app}}_i])\right)+\epsilon_\sigma.
$$

### Step 4

$$
(\mathbf{z}^{\mathrm{topo}}_i,\boldsymbol{\mu}_i,\boldsymbol{\sigma}_i^2)
\rightarrow
\mathbf{T}^{\mathrm{anc}}_i.
$$

### Step 5

$$
(\mathbf{h}_i,\mathbf{T}^{\mathrm{anc}}_i)
\rightarrow
\hat{\mathbf{p}}_i
\rightarrow
\hat{\mathbf{y}}_i.
$$

### Step 6: topology-anchor plausibility score

$$
d^{\mathrm{mah}}_i
=
\sum_{j=1}^{d_z}
\frac{(z^{\mathrm{topo}}_{i,j}-\mu_{i,j})^2}{\sigma^2_{i,j}+\epsilon_d}.
$$

### Step 7: self-calibrated confidence

Let $s^{\mathrm{raw}}_i$ be a raw segmentation confidence summary extracted from $\hat{\mathbf{p}}_i$. Then:

$$
s^{\mathrm{cal}}_i
=
s^{\mathrm{raw}}_i\exp(-\gamma d^{\mathrm{mah}}_i).
$$

where $\gamma>0$ controls calibration strength.

---

## 7. Professional project structure

```text
project_root/
├── README.md
├── pyproject.toml
├── requirements/
│   ├── base.txt
│   ├── train.txt
│   ├── topology.txt
│   ├── eval.txt
│   ├── viz.txt
│   └── dev.txt
├── configs/
│   ├── config.yaml
│   ├── data/
│   │   ├── cardiac_mri.yaml
│   │   ├── prostate_mri.yaml
│   │   └── brain_mri.yaml
│   ├── model/
│   │   ├── topology_anchored.yaml
│   │   ├── backbone_nnunet_resmamba.yaml
│   │   └── decoder_cross_attention.yaml
│   ├── train/
│   │   ├── default.yaml
│   │   ├── optimizer.yaml
│   │   ├── scheduler.yaml
│   │   └── augmentations.yaml
│   ├── loss/
│   │   └── default.yaml
│   ├── eval/
│   │   └── default.yaml
│   └── paths/
│       └── local.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── manifests/
│   ├── topology_cache/
│   └── splits/
├── reports/
│   ├── figures/
│   ├── tables/
│   ├── logs/
│   └── topology_analysis/
├── scripts/
│   ├── verify_environment.py
│   ├── verify_data.py
│   ├── preprocess_dataset.py
│   ├── build_manifests.py
│   ├── precompute_topology.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── plot_training_curves.py
│   ├── plot_calibration.py
│   ├── run_topology_analysis.py
│   └── export_predictions.py
├── src/
│   ├── data/
│   │   ├── manifests/
│   │   ├── preprocessing/
│   │   ├── transforms/
│   │   ├── datasets/
│   │   ├── samplers/
│   │   ├── datamodules/
│   │   └── loaders/
│   ├── models/
│   │   ├── backbones/
│   │   ├── heads/
│   │   ├── anchors/
│   │   ├── decoder/
│   │   ├── modules/
│   │   └── topology_anchored_model.py
│   ├── topology/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── skimage_descriptors.py
│   │   ├── gudhi_persistence.py
│   │   ├── gtda_vectorization.py
│   │   ├── targets.py
│   │   ├── pair_builder.py
│   │   ├── prototype_builder.py
│   │   ├── cache.py
│   │   ├── manifest_writer.py
│   │   └── sanity.py
│   ├── losses/
│   │   ├── segmentation_losses.py
│   │   ├── metric_loss.py
│   │   ├── anchor_distribution_loss.py
│   │   └── loss_factory.py
│   ├── evaluation/
│   ├── visualize/
│   ├── analysis/
│   ├── training/
│   └── utils/
├── tests/
└── notebooks/
```

---

## 8. The `src/topology/` module: full design

### Design principle

`src/topology/` must support three modes:

1. **offline preprocessing mode** for descriptor caching
2. **training supervision mode** for pair/prototype construction
3. **analysis/evaluation mode** for topology statistics and validation

### File-by-file design

#### `src/topology/schema.py`

Defines data structures for topology descriptors.

Example contents:

- `TopologyDescriptor`
- `BettiDescriptor`
- `PersistenceDescriptor`
- `TopologyCacheRecord`

No heavy backend dependency here.

---

#### `src/topology/skimage_descriptors.py`

**Uses:** `scikit-image`

Purpose:

- fast topology/morphology proxies from masks
- connected-component counts
- Euler characteristic proxies
- regionprops-based morphology summaries
- contour/label-derived structure counts

Typical functionality:

- count connected components
- estimate number of holes in 2D masks
- compute area, perimeter, compactness, eccentricity
- derive fast descriptors for pair building

Why `scikit-image` here?

- lightweight
- stable
- useful for mask-level structural summaries
- fast enough for preprocessing and cache generation

---

#### `src/topology/gudhi_persistence.py`

**Uses:** `gudhi`

Purpose:

- persistent homology on masks or logit maps
- cubical complex construction for 2D/3D images
- persistence diagram extraction
- optional bottleneck / Wasserstein-style comparisons

Typical functionality:

- convert binary mask or scalar field to cubical complex
- compute persistence intervals
- extract top-k lifetimes
- export diagram summaries for caching

Use this file when formal persistence descriptors are needed.

---

#### `src/topology/gtda_vectorization.py`

**Uses:** `giotto-tda`

Purpose:

- vectorize persistence diagrams into ML-friendly features
- persistence images
- persistence landscapes
- silhouettes
- entropy summaries

Recommended workflow:

1. compute diagrams with **GUDHI**
2. vectorize with **giotto-tda**
3. store vectors in topology cache
4. use cached vectors for pair/prototype building and analysis

---

#### `src/topology/targets.py`

Purpose:

- unify topology targets from multiple sources
- combine `scikit-image` descriptors and persistence vectors
- expose one consistent target interface for pair/prototype building

Example target:

```text
tau_i = {
  betti_like,
  euler_like,
  morphology_stats,
  persistence_vector
}
```

This corresponds mathematically to:

$$
\boldsymbol{\tau}_i=\mathcal{T}(\mathbf{y}_i).
$$

---

#### `src/topology/pair_builder.py`

Purpose:

- construct $\mathcal{P}_i$ and $\mathcal{N}_i$ for $\mathcal{L}_{\mathrm{metric}}$
- use topology targets, not raw style information

Inputs:

- cached topology descriptors from manifests or cache files
- thresholds / similarity settings from config

Typical policy:

- positives: similar topology and morphology
- negatives: clearly different topology or morphology
- optional hard-negative mining

---

#### `src/topology/prototype_builder.py`

Purpose:

- optional prototype construction for topology clusters
- can be used instead of pair-only supervision

Useful when:

- batch size is limited
- you want stable topology cluster centers
- you want later ablations with prototype-based metric learning

---

#### `src/topology/cache.py`

Purpose:

- cache computed descriptors to avoid recomputation
- store per-sample topology records
- support incremental preprocessing

Recommended storage:

- JSONL for human-readable metadata
- Parquet/CSV for tables
- NPZ/PT for dense vectors

---

#### `src/topology/manifest_writer.py`

Purpose:

- inject topology descriptors into data manifests
- keep preprocessing outputs synchronized with training inputs

---

#### `src/topology/sanity.py`

Purpose:

- validate descriptor ranges
- detect NaN / degenerate persistence results
- check consistency between cached descriptors and dataset manifests

This file should be called by `scripts/verify_data.py` and `scripts/precompute_topology.py`.

---

## 9. Which topology library is used where?

### `scikit-image`

Use in:

- `src/topology/skimage_descriptors.py`
- `src/data/preprocessing/mask_checks.py`
- optionally `src/evaluation/topology_metrics.py`

Best for:

- connected components
- hole-counting proxies in 2D
- Euler-style morphology checks
- fast region shape statistics

### `gudhi`

Use in:

- `src/topology/gudhi_persistence.py`
- optionally `src/evaluation/topology_metrics.py`
- optionally `src/analysis/topology/descriptor_stats.py`

Best for:

- cubical complex persistent homology
- persistence diagrams
- topological distances

### `giotto-tda`

Use in:

- `src/topology/gtda_vectorization.py`
- optionally `src/analysis/topology/descriptor_correlation.py`
- optionally `src/visualize/topology_plots.py`

Best for:

- persistence image / landscape / silhouette vectorization
- making persistence usable inside ML workflows

### Recommended division of labor

- **scikit-image** = fast structural descriptors
- **GUDHI** = formal persistence computation
- **giotto-tda** = persistence vectorization and analysis-ready features

---

## 10. Data manifests and data verification

A professional AI project should never rely on ad hoc file discovery during training.

### Required manifest fields

Each manifest row should contain at least:

- `sample_id`
- `image_path`
- `mask_path`
- `split`
- `dataset_name`
- `spacing`
- `shape`
- `orientation`
- `roi_bbox`
- `topology_cache_path`
- `topology_descriptor_version`

### Verification checks

Implement `scripts/verify_data.py` to validate:

- all image and mask files exist
- image-mask shapes align after preprocessing
- topology cache files exist when required
- no duplicated sample IDs
- all splits are disjoint
- no NaN or degenerate descriptors

---

## 11. Transforms, dataset, and dataloader design

### `src/data/transforms/`

Separate training and evaluation transforms.

#### Training transforms

- spatial flips/rotations if anatomy-safe
- intensity augmentations
- mild elastic or affine transforms
- appearance perturbations that preserve morphology as much as possible

#### Evaluation transforms

- deterministic only
- no stochastic augmentation

### `src/data/datasets/manifest_dataset.py`

Use manifest-driven loading.

Each `__getitem__` should return:

- image tensor
- mask tensor if available
- sample ID
- optional topology cache record
- optional metadata dict

### `src/data/samplers/topology_sampler.py`

Optional advanced sampler for metric learning:

- encourages same-topology positives in batch
- improves $\mathcal{L}_{\mathrm{metric}}$ stability

---

## 12. Loss implementation design

### `src/losses/segmentation_losses.py`

Implement:

- Dice
- BCE / CE
- combined segmentation loss

### `src/losses/metric_loss.py`

Implement supervised contrastive loss for $\mathbf{z}^{\mathrm{topo}}$.

Inputs:

- `z_topo`: tensor of shape `[B, d_z]`
- `positive_mask`: boolean tensor `[B, B]`, where `positive_mask[i, r] = True` iff $r\in\mathcal{P}_i$
- optional `valid_anchor_mask` for anchors with at least one positive

### `src/losses/anchor_distribution_loss.py`

Implements:

$$
\mathcal{L}_{\mathrm{dist}}
=
\frac{1}{B}\sum_{i=1}^{B}\frac{1}{2}\sum_{j=1}^{d_z}
\left[
\frac{(z^{\mathrm{topo}}_{i,j}-\mu_{i,j})^2}{\sigma^2_{i,j}+\epsilon_d}
+
\log(\sigma^2_{i,j}+\epsilon_d)
\right].
$$

Inputs:

- `z_topo`: `[B, d_z]`
- `mu`: `[B, d_z]`
- `var`: `[B, d_z]`, strictly positive

This file should be numerically stable and thoroughly tested.

### `src/losses/loss_factory.py`

Creates the final composite loss:

$$
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{seg}}
+
\lambda_{\mathrm{metric}}\mathcal{L}_{\mathrm{metric}}
+
\lambda_{\mathrm{dist}}\mathcal{L}_{\mathrm{dist}}.
$$

---

## 13. Evaluation design

### Standard segmentation metrics

In `src/evaluation/segmentation_metrics.py`:

- Dice
- IoU / Jaccard
- Hausdorff distance
- ASSD / boundary metrics

### Calibration metrics

In `src/evaluation/calibration_metrics.py`:

- ECE
- Brier score
- reliability diagrams
- confidence-error correlation

### Topology-aware evaluation

In `src/evaluation/topology_metrics.py`:

- compare structural descriptor distributions
- compare persistence summaries between prediction and GT
- count connected components and holes where relevant
- correlate topology deviation with segmentation error

### Robustness metrics

In `src/evaluation/robustness_metrics.py`:

- performance under vendor/style split
- performance under domain shift
- confidence degradation vs. performance degradation

---

## 14. Visualization and plotting

### `src/visualize/overlay.py`

- image/mask overlays
- false positive / false negative maps

### `src/visualize/curves.py`

- training loss curves
- metric trajectories

### `src/visualize/latent_space.py`

- t-SNE / UMAP of $\mathbf{z}^{\mathrm{topo}}$
- color by split, anatomy regime, topology label, or error

### `src/visualize/calibration.py`

- reliability diagrams
- confidence histograms
- Mahalanobis vs. error plots

### `src/visualize/topology_plots.py`

- descriptor histograms
- persistence summary distributions
- topology cluster visualizations

---

## 15. Topology analysis module

This project should have a dedicated analysis branch because topology claims must be verified.

### `src/analysis/topology/descriptor_stats.py`

- summarize descriptor distributions by split/domain
- report topological statistics of the dataset

### `src/analysis/topology/descriptor_correlation.py`

- correlate topology descriptors with segmentation difficulty, confidence, and latent distance

### `src/analysis/topology/topology_vs_error.py`

- analyze how topology deviation relates to Dice drop / Hausdorff increase

### `src/analysis/topology/topology_vs_confidence.py`

- analyze how topology plausibility relates to confidence calibration

---

## 16. Requirements and dependency management

Use layered requirements files.

### `requirements/base.txt`

Core runtime and utilities:

- numpy
- scipy
- pandas
- pyyaml
- hydra-core
- rich
- pydantic
- tqdm

### `requirements/train.txt`

Training stack:

- torch
- torchvision
- torchaudio
- lightning
- monai
- timm
- einops
- xformers, optional

### `requirements/topology.txt`

Topology stack:

- scikit-image
- gudhi
- giotto-tda
- networkx, optional

### `requirements/eval.txt`

Evaluation stack:

- scikit-learn
- statsmodels
- medpy, optional
- surface-distance, optional

### `requirements/viz.txt`

Visualization stack:

- matplotlib
- plotly
- seaborn, optional for analysis only
- umap-learn

### `requirements/dev.txt`

Developer tooling:

- pytest
- pytest-cov
- ruff
- black
- isort
- mypy
- pre-commit

### Dependency policy

- keep README generic
- pin exact versions in lockfiles or CI-tested environment files
- use `uv`, `pip-tools`, Poetry, or PDM locking in production research workflows

---

## 17. Config system

Use Hydra or an equivalent hierarchical config system.

### Core config groups

- `data/`
- `model/`
- `train/`
- `loss/`
- `eval/`
- `paths/`

### Example model config

```yaml
model:
  backbone: nnunet_resmamba
  topo_dim: 128
  app_dim: 64
  use_gap_gmp: true
  d3_cross_attention: true
  anchor:
    mean_from: z_topo_detached
    variance_from: [z_topo_detached, c_app]
    variance_activation: softplus
```

### Example loss config

```yaml
loss:
  lambda_metric: 0.2
  lambda_dist: 0.05
  metric:
    type: supcon
    temperature: 0.1
  seg:
    dice_weight: 1.0
    ce_weight: 1.0
  dist:
    epsilon: 1.0e-6
```

---

## 18. Professional scripts and phases

### Phase 1: environment verification

```bash
python scripts/verify_environment.py
```

Checks:

- Python and package availability
- GPU / CUDA visibility
- topology backends import correctly

### Phase 2: data verification

```bash
python scripts/verify_data.py --config-name config
```

Checks:

- raw files exist
- mask alignment
- manifest integrity
- topology cache readiness

### Phase 3: preprocessing

```bash
python scripts/preprocess_dataset.py --config-name config
python scripts/build_manifests.py --config-name config
python scripts/precompute_topology.py --config-name config
```

Outputs:

- processed images/masks
- manifests
- cached topology descriptors

### Phase 4: training

```bash
python scripts/train.py --config-name config
```

Outputs:

- checkpoints
- logs
- curves
- validation metrics

### Phase 5: evaluation

```bash
python scripts/evaluate.py --config-name config
```

Outputs:

- test metrics
- calibration metrics
- robustness tables
- topology analysis summaries

### Phase 6: plotting and reporting

```bash
python scripts/plot_training_curves.py --config-name config
python scripts/plot_calibration.py --config-name config
python scripts/run_topology_analysis.py --config-name config
```

Outputs:

- publication figures
- latent plots
- topology statistics figures

---

## 19. Minimal recommended implementation roadmap

### Milestone 1

Implement the compact topology-anchored model with:

- preprocessing
- manifests
- model core
- $\mathcal{L}_{\mathrm{seg}}$
- $\mathcal{L}_{\mathrm{metric}}$
- $\mathcal{L}_{\mathrm{dist}}$
- scikit-image-based topology descriptors

### Milestone 2

Add topology cache and offline persistent homology:

- `gudhi_persistence.py`
- `gtda_vectorization.py`
- richer pair building

### Milestone 3

Add deeper topology analysis and paper figures:

- latent clustering
- correlation with errors and confidence
- robustness under domain shift

---

## 20. Final interpretation of the architecture

This project is not a pure topology-explicit end-to-end differentiable topology network. It is a professional, practical, topology-anchored medical segmentation system that:

- learns a topology-focused latent representation $\mathbf{z}^{\mathrm{topo}}$
- anchors it through a topology-driven center $\boldsymbol{\mu}$
- modulates uncertainty with residual context through $\boldsymbol{\sigma}^2$
- injects the anchor into the decoder through anchor tokens $\mathbf{T}^{\mathrm{anc}}$
- validates topology claims with a dedicated offline topology module stack

That is exactly why the project needs both:

- a clean model core
- and a clear topology toolkit layer in `src/topology/`

---

## 21. Final checklist for a professional implementation

Before calling the project production-quality research code, confirm that you have:

- [ ] manifest-driven data loading
- [ ] deterministic preprocessing pipeline
- [ ] cached topology descriptors
- [ ] verified environment and data scripts
- [ ] separated config groups
- [ ] layered requirements files
- [ ] tested loss modules
- [ ] evaluation and calibration metrics
- [ ] topology analysis scripts
- [ ] plotting scripts for curves and latent analysis
- [ ] unit tests for preprocessing, topology descriptors, losses, and model forward

---

## 22. One-sentence summary

This repository implements a **topology-anchored, context-modulated MRI segmentation framework** in which the topology-specific latent code $\mathbf{z}^{\mathrm{topo}}$ defines the anchor center $\boldsymbol{\mu}$, residual context $\mathbf{c}^{\mathrm{app}}$ modulates the anchor spread $\boldsymbol{\sigma}^2$, and a dedicated topology stack based on **scikit-image**, **GUDHI**, and **giotto-tda** supports reproducible topology supervision, caching, and analysis.
