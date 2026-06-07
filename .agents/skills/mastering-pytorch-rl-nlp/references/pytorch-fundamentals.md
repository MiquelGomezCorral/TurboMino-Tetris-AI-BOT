# PyTorch Fundamentals Reference

Comprehensive guide to PyTorch 2.7+ fundamentals including tensors, autograd, nn.Module, and device management.

## Device Management (Multi-Platform)

### Universal Device Detection

```python
import torch

def get_device():
    """Get the best available device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    elif hasattr(torch.backends, 'xpu') and torch.backends.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")
```

### Apple Silicon (M2/M3/M4) MPS Backend

**Status (December 2025):** MPS backend is officially supported in PyTorch 1.12+.

```python
# Check MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device available")
else:
    print("MPS not available, using CPU")
```

**Known MPS Limitations:**
- Not all operations implemented - some fall back to CPU
- No distributed training support
- No float64 or fp16 tensor core support
- SDPA (Scaled Dot-Product Attention) can be unstable

**MPS Best Practices:**
```python
import os

# Enable MPS fallback for unsupported operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# For transformer models with attention issues
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Prevents memory spikes
```

**Performance Notes:**
- ~3x slower than RTX 4090 for large training
- 80% lower energy consumption
- Best for: prototyping, inference, light-medium training

### CUDA Device Management

```python
import torch

# Check CUDA availability
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Multi-GPU selection
device = torch.device("cuda:0")  # First GPU
device = torch.device("cuda:1")  # Second GPU

# Memory management
torch.cuda.empty_cache()
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

## Tensors

### Creating Tensors

```python
import torch

# From Python data
x = torch.tensor([1, 2, 3], dtype=torch.float32)
x = torch.tensor([[1, 2], [3, 4]], device=device)

# Common creation functions
zeros = torch.zeros(3, 4, device=device)
ones = torch.ones(3, 4, dtype=torch.float16)
rand = torch.rand(3, 4)  # Uniform [0, 1)
randn = torch.randn(3, 4)  # Normal distribution
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # 5 evenly spaced

# From NumPy (shares memory!)
import numpy as np
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)
```

### Tensor Operations

```python
# Basic math
a = torch.tensor([1, 2, 3], dtype=torch.float32)
b = torch.tensor([4, 5, 6], dtype=torch.float32)

c = a + b  # Element-wise addition
c = a * b  # Element-wise multiplication
c = a @ b  # Dot product (1D) or matrix multiply (2D+)
c = torch.matmul(a.unsqueeze(0), b.unsqueeze(1))  # Explicit matmul

# In-place operations (append _)
a.add_(1)  # Modifies a in place
a.zero_()  # Zeros in place

# Reduction operations
x = torch.randn(3, 4)
x.sum()  # Total sum
x.sum(dim=0)  # Sum along rows (result: shape [4])
x.sum(dim=1)  # Sum along columns (result: shape [3])
x.mean(), x.std(), x.var()
x.max(), x.min(), x.argmax(), x.argmin()
```

### Reshaping and Indexing

```python
x = torch.randn(2, 3, 4)

# Reshaping
x.view(6, 4)  # Must be contiguous
x.reshape(6, 4)  # Works on non-contiguous
x.flatten()  # To 1D
x.squeeze()  # Remove dims of size 1
x.unsqueeze(0)  # Add dim at position 0

# Indexing
x[0]  # First element along dim 0
x[:, 1]  # All rows, second column
x[..., -1]  # Last element along last dim
x[x > 0]  # Boolean indexing

# Advanced indexing
indices = torch.tensor([0, 2])
x.index_select(dim=1, index=indices)
```

## Autograd

### Gradient Computation

```python
import torch

# Enable gradient tracking
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2 + 3 * x
z = y.sum()

# Compute gradients
z.backward()
print(x.grad)  # dz/dx = 2x + 3 = [7.0, 9.0]

# Disable gradient tracking
with torch.no_grad():
    y = x * 2  # No gradient computation

# Detach from computation graph
y = x.detach()  # Same data, no gradient tracking
```

### Gradient Control

```python
# Zero gradients (important before each backward!)
optimizer.zero_grad()
# Or manually:
x.grad.zero_()

# Gradient clipping (prevents exploding gradients)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# Gradient accumulation
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## nn.Module

### Building Models

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Using nn.Sequential
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10)
)
```

### Model Operations

```python
# Move to device
model = model.to(device)

# Training vs evaluation mode
model.train()  # Enable dropout, batch norm training mode
model.eval()   # Disable dropout, use running stats for batch norm

# Parameter access
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Freeze parameters
for param in model.fc1.parameters():
    param.requires_grad = False
```

### Saving and Loading

```python
# Save model state dict (recommended)
torch.save(model.state_dict(), "model.pth")

# Load model state dict
model = SimpleNet(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("model.pth", map_location=device))

# Save entire model (less portable)
torch.save(model, "model_full.pth")
model = torch.load("model_full.pth")

# Save checkpoint with optimizer state
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, "checkpoint.pth")
```

## Common Layers

### Linear and Activation

```python
import torch.nn as nn

# Linear layers
nn.Linear(in_features, out_features, bias=True)

# Activations
nn.ReLU()
nn.LeakyReLU(negative_slope=0.01)
nn.GELU()  # Popular in transformers
nn.SiLU()  # Swish activation
nn.Tanh()
nn.Sigmoid()
nn.Softmax(dim=-1)
```

### Normalization

```python
# Batch normalization (for CNNs, MLPs)
nn.BatchNorm1d(num_features)
nn.BatchNorm2d(num_features)

# Layer normalization (for transformers, RNNs)
nn.LayerNorm(normalized_shape)

# Group normalization
nn.GroupNorm(num_groups, num_channels)
```

### Regularization

```python
# Dropout
nn.Dropout(p=0.5)
nn.Dropout2d(p=0.5)  # For conv layers

# Weight decay (L2 regularization) - applied via optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

## Data Loading

### Dataset and DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = CustomDataset(X, y)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,  # Faster GPU transfer
    drop_last=True    # Drop incomplete batches
)

for batch_data, batch_labels in dataloader:
    batch_data = batch_data.to(device)
    batch_labels = batch_labels.to(device)
    # Train...
```

## Training Loop Pattern

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Setup
model = SimpleNet(input_dim, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_data, batch_labels in train_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()

        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        correct = 0
        for batch_data, batch_labels in val_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_data)
            val_loss += criterion(outputs, batch_labels).item()
            correct += (outputs.argmax(dim=1) == batch_labels).sum().item()

        accuracy = correct / len(val_loader.dataset)

    print(f"Epoch {epoch}: Train Loss={total_loss:.4f}, Val Acc={accuracy:.4f}")
```

## torch.compile (PyTorch 2.0+)

```python
import torch

# Basic compilation (recommended for most cases)
model = torch.compile(model)

# Compilation modes
model = torch.compile(model, mode="default")      # Balanced
model = torch.compile(model, mode="reduce-overhead")  # Fast, less memory
model = torch.compile(model, mode="max-autotune")     # Slowest compile, fastest run

# Note: torch.compile has limited MPS support as of Dec 2025
# For Apple Silicon, test thoroughly or use without compile
```

## Best Practices Summary

1. **Always use device-agnostic code** - check CUDA, MPS, XPU availability
2. **Use `torch.no_grad()` for inference** - saves memory and computation
3. **Zero gradients before backward** - prevent gradient accumulation bugs
4. **Use `model.train()` and `model.eval()`** - affects dropout and batch norm
5. **Pin memory for GPU training** - faster data transfer
6. **Save state_dict, not full model** - more portable
7. **Use AMP for faster training** - see optimization guide
