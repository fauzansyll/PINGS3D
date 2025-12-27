import torch.nn as nn

class PINGS_Generator(nn.Module):
    def __init__(self, input_dim=4, output_dim=3, hidden_dim=128, num_layers=6):
        """
        MLP Architecture matching the PINGS paper.
        input_dim: 1 (time) + 3 (latent) = 4
        """
        super().__init__()
        
        layers = []
        # Input Layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden Layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        # Output Layer (Linear, no activation)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Explicit Initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, t, z):
        # Concatenate time t and latent z
        x = torch.cat([t, z], dim=1)
        return self.net(x)