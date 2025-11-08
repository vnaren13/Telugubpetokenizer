# Telugu BPE Tokenizer

A Byte Pair Encoding (BPE) tokenizer implementation for Telugu language, following the algorithmic approach from Andrej Karpathy's tokenization lectures.

## 📊 Statistics

- **Vocabulary Size:** 5,000 tokens ✅
- **Compression Ratio:** 3.24x ✅
- **Training Corpus:** 2,000 Telugu texts
- **Total Bytes Processed:** 4,776,832
- **Algorithm:** Byte Pair Encoding (BPE)
- **Training Time:** ~34 minutes

## 🎯 Assignment Requirements Met

✅ **Vocabulary Size:** 5000+ tokens (exactly 5000)
✅ **Compression Ratio:** ~3.2 (achieved 3.24)
✅ **Dataset:** salmankhanpm/Corpous_Telugu_Tokenizer
✅ **Upload to HuggingFace:** [vnaren13/telugu-bpe-tokenizer](https://huggingface.co/vnaren13/telugu-bpe-tokenizer)
✅ **Working Examples:** Provided with Gradio interface

## 🚀 Live Demo

**Gradio Interface:** https://13082b7aad2033bae1.gradio.live (expires in 1 week)

Try the tokenizer live with an interactive web interface!

## 📦 Repository Contents

- `Telugu_BPE_final.ipynb` - Complete implementation notebook
- `telugu_merges.json` - BPE merge rules (4,744 merges)
- `telugu_vocab.json` - Token vocabulary mappings
- `README.md` - This file

## 🔧 Implementation Details

### Core Algorithm (Professor's Pattern)

The implementation follows the exact pattern from the reference notebook (Session_20_01.ipynb):

#### 1. **get_stats(ids)** - Pair Frequency Counting
```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts
```
Uses `zip()` to iterate through consecutive token pairs.

#### 2. **merge(ids, pair, idx)** - Pair Replacement
```python
def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2  # Skip both tokens
        else:
            newids.append(ids[i])
            i += 1
    return newids
```
Manual index control: `i += 2` when matched, `i += 1` otherwise.

#### 3. **Training Loop** - Iterative Merging
```python
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)  # Most frequent pair
    idx = 256 + i
    ids = merge(ids, pair, idx)
    merges[pair] = idx
```
Starts token IDs at 256 (after UTF-8 base bytes).

#### 4. **Encoding** - Uses min() NOT max()
```python
def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        # KEY: min() finds earliest trained pair
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens
```

#### 5. **Decoding** - Vocabulary Lookup
```python
def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    return tokens.decode("utf-8", errors="replace")
```

## 📝 Usage

### Loading the Tokenizer

```python
import json

# Load tokenizer files
with open('telugu_merges.json', 'r') as f:
    merges = {eval(k): v for k, v in json.load(f).items()}

with open('telugu_vocab.json', 'r') as f:
    vocab = {int(k): bytes.fromhex(v) for k, v in json.load(f).items()}
```

### Helper Functions

```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        tokens = merge(tokens, pair, merges[pair])
    return tokens

def decode(ids):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
```

### Examples

```python
# Example 1: Simple greeting
text = "నమస్కారం"
encoded = encode(text)
decoded = decode(encoded)

print(f"Original: {text}")
print(f"Tokens: {encoded}")  # 4 tokens
print(f"Decoded: {decoded}")
print(f"Match: {text == decoded}")  # True

# Example 2: Sentence
text = "తెలుగు భాష"
encoded = encode(text)
print(f"Tokens: {len(encoded)}")  # 5 tokens

# Example 3: Longer text
text = "నేను తెలుగు నేర్చుకుంటున్నాను"
encoded = encode(text)
print(f"Tokens: {len(encoded)}")  # 8 tokens
compression = len(text) / len(encoded)
print(f"Compression: {compression:.2f}x")
```

## 🧪 Test Results

| Telugu Text | Tokens | Match |
|-------------|--------|-------|
| నమస్కారం | 4 | ✓ |
| తెలుగు భాష | 5 | ✓ |
| నేను తెలుగు నేర్చుకుంటున్నాను | 8 | ✓ |

All encodings successfully decode back to original text!

## 🎓 Training Process

1. **Dataset Loading:** Streamed 2,000 Telugu texts from HuggingFace
2. **Corpus Creation:** Combined texts into single corpus (4.7M bytes)
3. **BPE Training:** 4,744 merge operations (5000 - 256 base bytes)
4. **Vocabulary Building:** Created byte mappings for all tokens
5. **Validation:** Tested on multiple Telugu examples

### Training Progress
```
Merge  100/4744 | 91.4s  | ETA: 70.7 min
Merge 1000/4744 | 532.3s | ETA: 33.2 min
Merge 2000/4744 | 961.0s | ETA: 22.0 min
Merge 3000/4744 | 1369s  | ETA: 13.3 min
Merge 4000/4744 | 1770s  | ETA: 5.5 min
Merge 4744/4744 | DONE in 34.2 minutes
```

## 🌐 HuggingFace Repository

**Live at:** https://huggingface.co/vnaren13/telugu-bpe-tokenizer

Repository includes:
- `telugu_merges.json` - Merge rules
- `telugu_vocab.json` - Vocabulary
- `README.md` - Usage instructions

## 📊 Performance Metrics

- **Characters (no spaces):** Processed from 300 sample texts
- **Total Tokens:** Encoded count
- **Compression Ratio:** 3.24x (each token ≈ 3.24 characters)
- **Vocabulary Coverage:** 5,000 subword units
- **UTF-8 Base:** 256 byte-level tokens + 4,744 learned merges

## 🎨 Gradio Frontend

Interactive web interface for testing the tokenizer:

```python
import gradio as gr

def tokenize_ui(text):
    if not text.strip():
        return "Enter text!", "", ""
    enc = encode(text)
    dec = decode(enc)
    stats = f"Chars: {len(text)} | Tokens: {len(enc)} | Compression: {len(text)/len(enc):.2f}x"
    return stats, str(enc[:30]) + "...", dec

demo = gr.Interface(
    fn=tokenize_ui,
    inputs=gr.Textbox(label="Telugu Text", lines=3),
    outputs=[
        gr.Textbox(label="Stats"),
        gr.Textbox(label="Tokens"),
        gr.Textbox(label="Decoded")
    ],
    title="Telugu BPE Tokenizer",
    examples=[["నమస్కారం"], ["తెలుగు భాష"]]
)

demo.launch(share=True)
```

## 🔍 Technical Details

### Algorithm Flow
1. **Input:** Telugu text string
2. **UTF-8 Encoding:** Convert to byte sequence
3. **BPE Application:** Apply learned merges iteratively
4. **Output:** Token ID sequence

### Key Features
- **Subword tokenization:** Handles unknown words gracefully
- **Compression:** Reduces sequence length by ~3.24x
- **Reversible:** Perfect decode back to original text
- **Language-specific:** Optimized for Telugu Unicode patterns



## 👤 Author

**vnaren13**
- HuggingFace: [@vnaren13](https://huggingface.co/vnaren13)
- Repository: [telugu-bpe-tokenizer](https://huggingface.co/vnaren13/telugu-bpe-tokenizer)

## 📄 License

Apache 2.0

