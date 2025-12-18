import numpy as np
import torch
from torch import nn

def main():
    # Task 1.1: tensor
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)

    x_ones = torch.ones_like(x_data)
    x_rand = torch.rand_like(x_data, dtype=torch.float)

    print("x_data:\n", x_data)
    print("x_np:\n", x_np)
    print("x_ones:\n", x_ones)
    print("x_rand:\n", x_rand)
    print("shape/dtype/device:", x_rand.shape, x_rand.dtype, x_rand.device)

    # Task 2.1: autograd
    x = torch.ones(1, requires_grad=True)
    y = x + 2
    z = y * y * 3
    z.backward()
    print("x.grad =", x.grad)  # dz/dx

    # Task 3: nn.Linear, nn.Embedding, nn.Module
    linear_layer = nn.Linear(5, 2)
    input_tensor = torch.randn(3, 5)
    out = linear_layer(input_tensor)
    print("Linear out shape:", out.shape)

    embedding_layer = nn.Embedding(num_embeddings=10, embedding_dim=3)
    input_indices = torch.LongTensor([1, 5, 0, 8])
    embeds = embedding_layer(input_indices)
    print("Embeds shape:", embeds.shape)

    class MyFirstModel(nn.Module):
        def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.linear = nn.Linear(embedding_dim, hidden_dim)
            self.act = nn.ReLU()
            self.out = nn.Linear(hidden_dim, output_dim)

        def forward(self, indices: torch.Tensor) -> torch.Tensor:
            x = self.embedding(indices)
            h = self.act(self.linear(x))
            return self.out(h)

    model = MyFirstModel(vocab_size=100, embedding_dim=16, hidden_dim=8, output_dim=2)
    sample = torch.LongTensor([[1, 2, 5, 9]])
    print("Model output shape:", model(sample).shape)

if __name__ == "__main__":
    main()
