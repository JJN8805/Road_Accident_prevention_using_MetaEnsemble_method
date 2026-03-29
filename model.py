import torch
import torch.nn as nn

class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_model, dropout=0.3):
        super().__init__()
        self.num_proj = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.xavier_uniform_(self.num_proj)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        tokens = torch.einsum("bf,fd->bfd", x, self.num_proj)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        return self.dropout(torch.cat([cls, tokens], dim=1))


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_features,
        d_model=256,
        nhead=16,
        ff_dim=128,
        num_layers=3,
        dropout=0.3,
        out_dim=1,
    ):
        super().__init__()

        self.tok = FeatureTokenizer(n_features, d_model, dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x):
        x = self.tok(x)
        x = self.encoder(x)
        cls = self.norm(x[:, 0, :])
        return self.head(cls).squeeze(1)
