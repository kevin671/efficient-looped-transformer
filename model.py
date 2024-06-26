import torch
import torch.nn as nn
import torch.nn.functional as F

class LoopedTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes, num_loops):
        super(LoopedTransformer, self).__init__()
        
        self.model_dim = model_dim
        self.num_loops = num_loops
        
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, model_dim))
        
        #self.transformer_layers = nn.ModuleList(
        #    [nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads) for _ in range(num_layers)]
        #)
        self.layers = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(model_dim, num_classes)
        
    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        
        for _ in range(self.num_loops):
            #for layer in self.transformer_layers:
            #    x = layer(x)
            for layer in self.layers:
                x = x + layer(x)
        
        x = x.mean(dim=1)
        x = self.fc_out(x)
        return x
    
    def parallel_forward(self, x, n_step):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]

        xs = [x.clone() for _ in range(self.num_loops+1)]

        for _ in range(n_step):
            for k in range(self.num_loops):
                sum_f_xs = sum([self._apply_layers(xi) for xi in xs[:k+1]])
                xs[k+1] = xs[0] + (1 / self.num_loops) * sum_f_xs
        
        x = x.mean(dim=1)
        x = self.fc_out(x)
        return x
    
    def _apply_layers(self, x):
        for layer in self.layers:
            x = layer(x)
        return x