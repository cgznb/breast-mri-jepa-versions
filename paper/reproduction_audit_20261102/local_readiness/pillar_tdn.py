import torch
import torch.nn as nn


class Projection(nn.Module):
    """Project a frozen per-timepoint embedding (input_dim -> proj_dim) with a LayerNorm."""

    def __init__(self, input_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.LayerNorm(proj_dim))
        self.out_dim = proj_dim

    def forward(self, x):                       # x: [..., input_dim]
        return self.net(x)                      # [..., proj_dim]


class T0Head(nn.Module):
    """Single-timepoint classifier on the baseline (T0) embedding, optionally concatenated with the
    clinical vector. No temporal modeling and no clinical prior (the single-scan baseline)."""

    def __init__(self, config):
        super().__init__()
        ds = config["downstream"]
        input_dim = ds.get("input_dim", 1152)
        clinical_dim = ds.get("clinical_dim", 0)
        proj_dim = ds.get("proj_dim", 128)
        h = ds.get("hidden_dim", 64)
        dropout = ds.get("dropout", 0.2)
        self.proj = Projection(input_dim, proj_dim)
        self.net = nn.Sequential(
            nn.Linear(self.proj.out_dim + clinical_dim, h), nn.LayerNorm(h), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(h, 1),
        )

    def forward(self, embeddings, mask, clinical, prior_logit=None):
        B = embeddings.shape[0]
        idx0 = torch.argmax((mask > 0).float(), dim=1)          # first present TP (baseline)
        z0 = embeddings[torch.arange(B, device=embeddings.device), idx0]
        z = self.proj(z0)
        f = torch.cat([z, clinical], dim=-1) if clinical.shape[-1] > 0 else z
        return self.net(f).squeeze(-1)


class TDN(nn.Module):
    """Temporal Dynamics Network (TDN).

    Projects each timepoint's frozen embedding into a compact disease signature, adds an elapsed-time
    positional encoding, and reads the sequence with a Transformer over a learned [pCR] query, the
    per-timepoint tokens, and a clinical token (a key-padding mask excludes missing timepoints). The
    query output gives a residual logit that is fused with a fixed logistic-regression clinical prior.

    Config keys (under config["downstream"]): input_dim, proj_dim, clinical_dim (set by the runner);
    sig_dim, n_layers, n_heads, dropout, use_prior.
    """

    def __init__(self, config):
        super().__init__()
        ds = config["downstream"]
        input_dim = ds.get("input_dim", 1152)
        proj_dim = ds.get("proj_dim", 128)
        clinical_dim = ds.get("clinical_dim", 0)
        d = ds.get("sig_dim", 64)
        n_layers = ds.get("n_layers", 1)
        n_heads = ds.get("n_heads", 4)
        dropout = ds.get("dropout", 0.2)
        self.d = d
        self.use_prior = ds.get("use_prior", False)

        self.proj = Projection(input_dim, proj_dim)
        self.sig = nn.Sequential(
            nn.Linear(self.proj.out_dim, d), nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, d), nn.LayerNorm(d),
        )
        self.pos = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.clin = (nn.Sequential(nn.Linear(clinical_dim, d), nn.GELU(), nn.Linear(d, d))
                     if clinical_dim > 0 else None)
        self.q = nn.Parameter(torch.randn(d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=2 * d, dropout=dropout,
                                         batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, n_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d, 1)
        if self.use_prior:
            self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, embeddings, mask, clinical, days=None, prior_logit=None):
        B, T, _ = embeddings.shape
        d = self.d
        s = self.sig(self.proj(embeddings))                       # [B,T,d] disease signature
        present = mask > 0

        # time descriptor: linearly scaled days, log-scaled days, and the normalized scan order
        tnorm = (torch.arange(T, device=embeddings.device).float()
                 / max(T - 1, 1)).view(1, T).expand(B, T)
        dd = days if days is not None else torch.zeros_like(tnorm)
        pe = torch.stack([(dd / 180.0).clamp(0, 2), torch.log1p(dd) / 6.0, tnorm], dim=-1)
        x = s + self.pos(pe)                                      # tokens [B,T,d]

        q = self.q.view(1, 1, d).expand(B, 1, d)
        toks = [q, x]
        pad = [torch.zeros(B, 1, dtype=torch.bool, device=mask.device), ~present]   # True = ignore
        if self.clin is not None:
            toks.append(self.clin(clinical).unsqueeze(1))
            pad.append(torch.zeros(B, 1, dtype=torch.bool, device=mask.device))
        H = self.tr(torch.cat(toks, dim=1), src_key_padding_mask=torch.cat(pad, dim=1))
        h = self.dropout(H[:, 0, :])                              # [pCR] query output
        logit = self.head(h).squeeze(-1)
        if self.use_prior and prior_logit is not None:
            logit = prior_logit + self.alpha * logit
        return logit
