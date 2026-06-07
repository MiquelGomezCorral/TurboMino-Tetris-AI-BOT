# NLP and Transformers Reference

Comprehensive guide to NLP with HuggingFace Transformers 4.56+, BERT, GPT, fine-tuning, and PEFT/LoRA.

## HuggingFace Transformers Setup

### Installation

```bash
pip install transformers datasets peft accelerate
```

### Core Imports

```python
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    pipeline,
    DataCollatorWithPadding
)
from datasets import load_dataset
```

## Pipeline API (Quick Inference)

The fastest way to use pre-trained models:

```python
from transformers import pipeline

# Sentiment Analysis
classifier = pipeline("sentiment-analysis")
result = classifier("I love PyTorch!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Specific model
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Text Generation
generator = pipeline("text-generation", model="gpt2")
result = generator("PyTorch is", max_length=50)

# Question Answering
qa = pipeline("question-answering")
result = qa(
    question="What is PyTorch?",
    context="PyTorch is a machine learning framework based on Torch."
)

# Named Entity Recognition
ner = pipeline("ner", grouped_entities=True)
result = ner("Hugging Face is based in New York City.")

# Zero-shot Classification
classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a great movie!",
    candidate_labels=["positive", "negative", "neutral"]
)

# Summarization
summarizer = pipeline("summarization")
result = summarizer(long_text, max_length=100, min_length=30)

# Translation
translator = pipeline("translation_en_to_fr")
result = translator("Hello, how are you?")
```

### Pipeline with GPU

```python
import torch

device = 0 if torch.cuda.is_available() else -1  # -1 for CPU
classifier = pipeline("sentiment-analysis", device=device)

# Or with specific device
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device="cuda:0"
)
```

## Tokenizers

### Basic Tokenization

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Single text
tokens = tokenizer("Hello, world!")
print(tokens)
# {'input_ids': [...], 'attention_mask': [...]}

# Batch processing
texts = ["Hello, world!", "PyTorch is great!"]
tokens = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"  # Return PyTorch tensors
)

# Decode back to text
decoded = tokenizer.decode(tokens["input_ids"][0])
```

### Advanced Tokenization

```python
# For sequence classification (single text)
tokens = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

# For sequence pair tasks (e.g., NLI, QA)
tokens = tokenizer(
    text_a,
    text_b,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# Token-level tasks (NER)
tokens = tokenizer(
    text,
    is_split_into_words=True,  # Pre-tokenized input
    return_offsets_mapping=True
)
```

## Models

### Loading Models

```python
from transformers import AutoModel, AutoModelForSequenceClassification
import torch

# Base model (embeddings only)
model = AutoModel.from_pretrained("bert-base-uncased")

# Classification head
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Causal LM (text generation)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### Model Inference

```python
model.eval()
with torch.no_grad():
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # Classification
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    # Get hidden states
    hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
```

## Fine-Tuning with Trainer

### Complete Fine-Tuning Example

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
import torch

# Load dataset
dataset = load_dataset("imdb")

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512
    )

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments - NOTE: use "epoch" for when to run validation
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # When to validate
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,  # Mixed precision
    push_to_hub=False
)

# Compute metrics
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred_tuple):
    logits, labels = pred_tuple
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Validate
results = trainer.evaluate()
print(results)

# Save model
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")
```

## PEFT and LoRA

Parameter-Efficient Fine-Tuning dramatically reduces trainable parameters.

### Installation

```bash
pip install peft
```

### LoRA Fine-Tuning

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification

# Load base model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,  # Rank
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1,
    target_modules=["query", "value"],  # Which layers to adapt
    bias="none"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
# trainable params: 294,912 || all params: 109,778,690 || trainable%: 0.27%

# Train as normal with Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer
)
trainer.train()

# Save LoRA weights only
model.save_pretrained("./lora_model")

# Load LoRA model
from peft import PeftModel
base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model = PeftModel.from_pretrained(base_model, "./lora_model")
```

### QLoRA (Quantized LoRA)

For larger models with limited memory:

```python
from transformers import BitsAndBytesConfig
import torch

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply LoRA on quantized model
model = get_peft_model(model, lora_config)
```

## Token Classification (NER)

```python
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from datasets import load_dataset

# Load dataset
dataset = load_dataset("conll2003")

# Load model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(dataset["train"].features["ner_tags"].feature.names)
)

# Tokenize with label alignment
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore in loss
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # Subword tokens
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Train
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./ner_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3
    ),
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer
)
trainer.train()
```

## Text Generation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text
input_text = "PyTorch is"
inputs = tokenizer(input_text, return_tensors="pt")

# Generation parameters
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Popular Models

### Classification
- `bert-base-uncased` - General purpose
- `distilbert-base-uncased` - Faster, smaller
- `microsoft/deberta-v3-base` - Best accuracy
- `roberta-base` - Robust BERT

### Generation
- `gpt2`, `gpt2-medium`, `gpt2-large`
- `meta-llama/Llama-2-7b-hf`
- `mistralai/Mistral-7B-v0.1`

### Embeddings
- `sentence-transformers/all-MiniLM-L6-v2` - Fast
- `sentence-transformers/all-mpnet-base-v2` - Quality

### Multilingual
- `bert-base-multilingual-cased`
- `xlm-roberta-base`

## Best Practices

### Memory Optimization

```python
# Gradient checkpointing
model.gradient_checkpointing_enable()

# Mixed precision
training_args = TrainingArguments(
    fp16=True,  # NVIDIA GPUs
    bf16=True,  # Ampere+ GPUs, better for transformers
)

# Gradient accumulation
training_args = TrainingArguments(
    gradient_accumulation_steps=4,  # Effective batch = batch_size * 4
)
```

### Efficient Inference

```python
# Batch inference
texts = ["text1", "text2", "text3"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs.to(device))

# Use torch.compile
model = torch.compile(model)
```

### Data Loading

```python
from datasets import load_dataset

# Stream large datasets
dataset = load_dataset("wikipedia", streaming=True)

# Efficient preprocessing
dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,  # Parallel processing
    remove_columns=["text"]
)
```

## Common Issues

### Out of Memory
- Reduce batch size
- Enable gradient checkpointing
- Use gradient accumulation
- Use PEFT/LoRA
- Use mixed precision (fp16/bf16)

### Slow Training
- Use DataLoader with num_workers > 0
- Use pin_memory=True
- Ensure GPU utilization is high
- Use torch.compile()

### Poor Results
- Verify data quality
- Try different learning rates (1e-5 to 5e-5)
- Increase training epochs
- Use learning rate warmup
- Check for label imbalance
