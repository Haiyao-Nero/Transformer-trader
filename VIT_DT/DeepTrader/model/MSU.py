import torch
import torch.nn as nn

class MSU(nn.Module):
    def __init__(self, in_features, window_len, hidden_dim=128):
        super(MSU, self).__init__()
        self.in_features = in_features
        self.window_len = window_len
        self.hidden_dim = hidden_dim

        # Linear Layer (2->128), Relu activation
        self.embedding = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()

        # Linear (128->768) to pass to transformer
        self.pre_transformer_linear = nn.Linear(hidden_dim, 768)

        # Transformer setup
        encoder_layers = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=6)

        # Post-attention layers
        self.linear1 = nn.Linear(768, 10)

    def forward(self, X):
        """
        :X: [batch_size(B), window_len(L), in_features(I)]
        :return: Parameters: [batch, 2]
        """
        X = X.permute(1, 0, 2)

        # Apply the embedding transformation
        X = self.embedding(X)  # 2->128
        X = self.relu(X)
        X = self.pre_transformer_linear(X)  # 128->768

        # Transformer encoder
        attn_embed = self.transformer_encoder(X)

        embed = attn_embed.permute(1, 0, 2).mean(dim=1)  # average pooling over time steps

        # Post-attention processing
        parameters = torch.softmax(self.linear1(embed), dim=-1)

        parameters = parameters.squeeze(-1)
        return parameters

    def freeze(self):
      for param in self.parameters():
        param.requires_grad = False

    def unfreeze(self):
      for param in self.parameters():
        param.requires_grad = True

if __name__ == '__main__':
  a = torch.randn((1, 20, 2))
  net = MSU(2, 20, 128)
  b = net(a)
  print("MSU_b:",b)