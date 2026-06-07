# Advanced Topics Reference

Multi-modal learning, CLIP, RLHF, and responsible AI practices.

## Multi-Modal Learning

### CLIP (Contrastive Language-Image Pre-training)

```python
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Process image and text
image = Image.open("image.jpg")
texts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]

inputs = processor(
    text=texts,
    images=image,
    return_tensors="pt",
    padding=True
)

# Get similarity scores
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
print(f"Probabilities: {probs}")
```

### Image-Text Matching

```python
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Image captioning
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

image = Image.open("image.jpg")
pixel_values = processor(images=image, return_tensors="pt").pixel_values

output_ids = model.generate(pixel_values, max_length=16)
caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

## RLHF (Reinforcement Learning from Human Feedback)

### Overview

RLHF trains language models to align with human preferences:
1. **Supervised Fine-Tuning (SFT)** - Fine-tune on demonstration data
2. **Reward Modeling** - Train model to predict human preferences
3. **RL Fine-Tuning** - Use PPO to optimize for reward model

### Using TRL Library

```bash
pip install trl
```

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

# Load model with value head
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Configure PPO
ppo_config = PPOConfig(
    batch_size=16,
    learning_rate=1e-5,
    mini_batch_size=4,
    ppo_epochs=4
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer
)
```

### Reward Model Training

```python
from transformers import AutoModelForSequenceClassification, Trainer

# Load reward model
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=1  # Scalar reward
)

# Train on preference data
# Dataset format: (prompt, chosen_response, rejected_response)
```

## Ethics and Responsible AI

### Bias Detection

```python
# Check for bias in embeddings
def measure_bias(model, tokenizer, word_pairs):
    """Measure association bias between word pairs."""
    results = {}
    for target, attributes in word_pairs.items():
        target_emb = get_embedding(model, tokenizer, target)
        attr_embs = [get_embedding(model, tokenizer, a) for a in attributes]
        results[target] = compute_association(target_emb, attr_embs)
    return results

# Example word pairs for gender bias
word_pairs = {
    "programmer": ["he", "she"],
    "nurse": ["he", "she"],
}
```

### Fairness Metrics

```python
from sklearn.metrics import confusion_matrix

def demographic_parity(predictions, sensitive_attribute):
    """Check if positive prediction rates are equal across groups."""
    groups = {}
    for pred, attr in zip(predictions, sensitive_attribute):
        if attr not in groups:
            groups[attr] = []
        groups[attr].append(pred)
    
    rates = {g: sum(p) / len(p) for g, p in groups.items()}
    return rates

def equalized_odds(predictions, labels, sensitive_attribute):
    """Check if TPR and FPR are equal across groups."""
    # Implementation for equalized odds metric
    pass
```

### Model Cards

Always document your models with:
- Intended use cases
- Training data sources
- Performance metrics by demographic
- Known limitations and biases
- Ethical considerations

```markdown
# Model Card: MyClassifier

## Model Details
- Model type: BERT-based classifier
- Training data: [Dataset name and description]
- Training procedure: Fine-tuned for 3 epochs

## Intended Use
- Primary: Sentiment analysis for product reviews
- Out of scope: Medical or legal advice

## Limitations
- May not perform well on non-English text
- Biased toward certain demographics

## Ethical Considerations
- Tested for fairness across demographic groups
- Should not be used for automated decisions without human review
```

### Robustness Testing

```python
# Adversarial examples
def perturb_text(text, perturbation_type="typo"):
    """Generate adversarial perturbations."""
    if perturbation_type == "typo":
        # Add random typos
        pass
    elif perturbation_type == "synonym":
        # Replace with synonyms
        pass
    return perturbed_text

# Test model robustness
def test_robustness(model, test_cases):
    original_preds = model(test_cases)
    perturbed = [perturb_text(t) for t in test_cases]
    perturbed_preds = model(perturbed)
    consistency = sum(o == p for o, p in zip(original_preds, perturbed_preds))
    return consistency / len(test_cases)
```

## Advanced Transformer Architectures

### Efficient Attention

```python
# Scaled Dot-Product Attention (SDPA) - PyTorch native
import torch.nn.functional as F

# Uses Flash Attention when available
output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=mask,
    dropout_p=0.1 if training else 0.0,
    is_causal=True  # For autoregressive models
)
```

### FlexAttention (PyTorch 2.5+)

```python
from torch.nn.attention.flex_attention import flex_attention

# Custom attention patterns
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

output = flex_attention(query, key, value, score_mod=causal_mask)
```

## Multi-Agent RL

### PettingZoo Environments

```python
from pettingzoo.mpe import simple_spread_v3
from torchrl.envs import PettingZooEnv

# Create multi-agent environment
env = PettingZooEnv(
    env=simple_spread_v3.parallel_env(),
    categorical_actions=True
)

# Each agent gets observations and takes actions
td = env.reset()
for agent in env.agents:
    print(f"{agent}: {td[agent]['observation'].shape}")
```

### Independent Learners

```python
# Simple approach: train each agent independently
agents = {name: PPOAgent(obs_dim, act_dim) for name in env.agents}

for episode in range(num_episodes):
    td = env.reset()
    while not td["done"].all():
        actions = {}
        for name, agent in agents.items():
            actions[name] = agent.act(td[name]["observation"])
        td = env.step(actions)
        
        for name, agent in agents.items():
            agent.update(td[name])
```

## Best Practices Summary

1. **Multi-Modal**: Use pre-trained models like CLIP, fine-tune for specific tasks
2. **RLHF**: Start with SFT, then reward modeling, finally PPO tuning
3. **Ethics**: Always test for bias, document limitations, use model cards
4. **Robustness**: Test with adversarial examples, edge cases
5. **Efficiency**: Use native SDPA, FlexAttention for custom patterns
6. **Multi-Agent**: Start with independent learners, then centralized training
