# Optimization and Deployment Reference

Comprehensive guide to torch.compile, quantization, distributed training, and production deployment.

## torch.compile (PyTorch 2.0+)

### Basic Usage

```python
import torch

model = MyModel()
model = model.to(device)

# Compile for faster execution
model = torch.compile(model)

# Now use normally
output = model(input)
```

### Compilation Modes

```python
# Default - balanced compile time and performance
model = torch.compile(model, mode="default")

# Reduce overhead - faster compile, good for smaller models
model = torch.compile(model, mode="reduce-overhead")

# Max autotune - slowest compile, fastest runtime
model = torch.compile(model, mode="max-autotune")
```

### Platform Support

| Platform | torch.compile Support | Notes |
|----------|----------------------|-------|
| CUDA | Full | Best performance |
| MPS (Apple) | Limited | Some operations fallback |
| XPU (Intel) | Partial | Improving |
| CPU | Yes | Useful for inference |

## Mixed Precision Training (AMP)

### Automatic Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler
import torch

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # Forward pass in mixed precision
    with autocast():
        output = model(batch)
        loss = criterion(output, targets)

    # Scaled backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### BFloat16 (Ampere+ GPUs)

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(input)
```

## Quantization

### Dynamic Quantization

```python
import torch

quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

### HuggingFace Quantization

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
```

## Distributed Training

### DistributedDataParallel (DDP)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

### Launch with torchrun

```bash
torchrun --nproc_per_node=4 train.py
```

## ONNX Export

```python
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=17,
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)
```

## TorchServe Deployment

```bash
pip install torchserve torch-model-archiver

torch-model-archiver \
    --model-name my_model \
    --version 1.0 \
    --serialized-file model.pth \
    --handler handler.py

torchserve --start --model-store model_store --models my_model=my_model.mar
```

## Inference Optimization Checklist

1. Use torch.compile - 10-30% speedup
2. Enable mixed precision - Faster and less memory
3. Batch inputs - Better GPU utilization
4. Use torch.no_grad() - No gradient computation
5. Quantize if possible - Smaller, faster
6. Use model.train(False)

```python
model.train(False)
model = model.to(device)
model = torch.compile(model)

@torch.no_grad()
def predict(inputs):
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        return model(inputs.to(device))
```
