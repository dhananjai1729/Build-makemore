# Copilot Instructions for Build-makemore

## Project Overview
Build-makemore is an educational project implementing bigram language models for generating synthetic names. It includes two parallel implementations:
1. **Build_makemore_YEPPY.ipynb** - English names generation using raw PyTorch tensors
2. **malayalam_bigram.ipynb** - Malayalam names generation using neural networks

Inspired by Andrej Karpathy's "Neural Networks: Zero to Hero" course.

## Architecture & Data Flow

### Bigram Model Core Pattern
Both notebooks follow the same core pipeline:
1. **Load names dataset** - Names from `names.txt` (English) or HuggingFace `santhosh/english-malayalam-names` (Malayalam)
2. **Character encoding** - Build `stoi` (string-to-index) and `itos` (index-to-string) mappings for vocabulary
3. **Bigram counting** - Create co-occurrence matrix `N` where `N[ch1][ch2]` = frequency of character pairs
4. **Probability modeling** - Convert counts to probabilities (with Laplace smoothing `+1`)
5. **Generation** - Sample next character using multinomial distribution

### Key Data Structures
- **stoi/itos**: Character-level vocabularies (0 = special token for `.`, indices 1+ = characters)
- **N matrix**: 2D tensor tracking bigram frequencies (shape: `[vocab_size, vocab_size]`)
- **P matrix**: Normalized probability matrix derived from `N` with smoothing
- **W matrix**: Neural network weight matrix (Malayalam notebook only) trained via gradient descent

### Language-Specific Differences
- **English**: Simple ASCII characters with static `names.txt` dataset (32,032 names)
- **Malayalam**: Unicode characters (0x0D00-0x0D80 range), dataset loaded from HuggingFace with filtering

## Critical Conventions

### Tensor Operations
- Use `torch.multinomial()` with explicit seed for reproducible generation: `generator=torch.Generator().manual_seed(seed_value)`
- Normalize probabilities with `sum(dim=1, keepdim=True)` to maintain 2D shape
- One-hot encoding uses `F.one_hot(input, num_classes=vocab_size).float()` for compatibility with matrix multiplication

### Character Boundaries
- Start token: `.` at index 0
- Words encoded as `['.'] + list(word) + ['.']` to mark boundaries
- Generator loops check `if ix == 0: break` to detect end-of-word

### Loss Computation (Neural Model)
- Negative log-likelihood: `-probs[indices, targets].log().mean()`
- L2 regularization: `0.01 * (W**2).mean()` to prevent extreme weights
- Gradient updates: `-learning_rate * W.grad` (no optimizer class used, manual updates)

## Running & Debugging

### Notebook Execution
Both notebooks are designed to run cell-by-cell. Key checkpoints:
1. **Data loading** - Verify `words` list is populated and non-empty
2. **Vocabulary building** - Confirm `itos` dict maps all indices (should be ~27 for English, ~100+ for Malayalam)
3. **Matrix visualization** - `plt.imshow(N)` shows bigram heatmap (visual sanity check)
4. **Generation** - Sample 50 names to verify learned patterns

### Common Issues
- **Character encoding errors (Malayalam)**: Ensure dataset is properly filtered with `all(ch in malayalam_characters for ch in word)`
- **Empty vocab**: Check if `stoi['.']` is always set to 0 before other characters
- **Loss not decreasing**: Increase learning rate in gradient descent loop or verify `W.grad` is being reset to None each iteration

## External Dependencies
- `torch` - Tensor operations and neural computation
- `datasets` - HuggingFace dataset loading (Malayalam notebook only)
- `pandas` - Data manipulation (Malayalam notebook only)
- `matplotlib` - Visualization

Install with: `pip install -r requirements.txt`

## Extension Points
When adding features:
- New datasets: Follow the character encoding pattern (`stoi`/`itos` then bigram counting)
- Larger context: Modify to n-grams by changing zip patterns (`zip(chs, chs[1:])` → `zip(chs, chs[n:])`)
- Different languages: Use appropriate Unicode ranges and dataset sources
