# Debug Patch Injection Guide

Complete guide for adding debug capabilities to your existing Hermes code **without rewriting everything**.

---

## 🎯 Overview

The debug patch is a **wrapper** that adds comprehensive debugging to your existing trainer with minimal code changes. No need to rewrite your training loop!

### What It Does:
- ✅ Validates data loading before training
- ✅ Tracks GPU memory usage
- ✅ Times every operation
- ✅ Catches common data issues
- ✅ Provides detailed logging
- ✅ Works with **your existing code**

---

## 📦 Installation

### Option 1: Download the Patch

```bash
# Download debug_patch.py to your project directory
wget https://your-repo/debug_patch.py
```

### Option 2: Copy-Paste

Just copy `debug_patch.py` into your project folder.

---

## 🚀 Quick Start (30 seconds)

### Minimal Changes Required

**BEFORE (Your existing code):**
```python
# Your existing training script
trainer = Trainer(model, config)
trainer.train(flood_loader, damage_loader)
```

**AFTER (With debug patch):**
```python
from debug_patch import DebugPatch

# Just wrap your trainer!
trainer = Trainer(model, config)
debug_patch = DebugPatch(trainer, debug_level=2)
debug_patch.train(flood_loader, damage_loader)
```

That's it! 🎉

---

## 📖 Detailed Usage Examples

### Example 1: Basic Injection

```python
from debug_patch import DebugPatch

# Your existing setup (don't change this)
model = EnhancedDisasterModel(...)
trainer = DisasterTrainer(model, config)

# Wrap with debug
debug_patch = DebugPatch(trainer, debug_level=2)

# Train normally
debug_patch.train(
    flood_train_loader=flood_loader,
    damage_train_loader=damage_loader,
    flood_val_loader=flood_val_loader,
    damage_val_loader=damage_val_loader
)
```

### Example 2: One-Line Injection

```python
from debug_patch import inject_debug

# Your existing setup
trainer = DisasterTrainer(model, config)

# One-line wrap
trainer = inject_debug(trainer, debug_level=2)

# Use normally
trainer.train(flood_loader, damage_loader)
```

### Example 3: Per-Epoch Training

```python
from debug_patch import DebugPatch

trainer = DisasterTrainer(model, config)
debug_patch = DebugPatch(trainer, debug_level=2)

# Train epoch by epoch for more control
for epoch in range(config.num_epochs):
    debug_patch.train_epoch(flood_loader, damage_loader, epoch)
    
    # Your custom logic here
    if epoch % 5 == 0:
        save_checkpoint(model, epoch)
```

### Example 4: Quick Dataset Diagnosis

```python
from debug_patch import quick_diagnose

# Diagnose issues BEFORE training
quick_diagnose(flood_train_dataset, flood_train_loader, num_samples=5)
quick_diagnose(damage_train_dataset, damage_train_loader, num_samples=5)

# Then train normally
```

---

## 🎛️ Debug Levels

### Level 0: Off
```python
debug_patch = DebugPatch(trainer, debug_level=0)
```
- No debugging output
- Minimal overhead
- Use for production

### Level 1: Basic
```python
debug_patch = DebugPatch(trainer, debug_level=1)
```
**Output:**
```
🔍 Testing DataLoaders...
  ✅ Flood loader OK: torch.Size([16, 3, 256, 256])
  ✅ Damage loader OK: torch.Size([16, 3, 256, 256])

EPOCH 1 COMPLETE
Time: 156.3s (2.6m)
Avg step time: 0.782s
```

### Level 2: Verbose (Recommended)
```python
debug_patch = DebugPatch(trainer, debug_level=2)
```
**Output:**
```
🔍 Testing DataLoaders...
  ✅ Flood loader OK: torch.Size([16, 3, 256, 256])

Iteration 0
  Flood loss: 0.8234
  Damage loss: 1.1234
  Time: 0.234s
  GPU: 8.23GB / 10.45GB

Iteration 10
  Flood loss: 0.7891
  Damage loss: 1.0567
  Time: 0.228s
  GPU: 8.25GB / 10.45GB

MEMORY USAGE SUMMARY
════════════════════════════════════════
Epoch 1 start           | 2.34GB / 5.12GB
Iter 0                  | 8.23GB / 10.45GB
Iter 10                 | 8.25GB / 10.45GB
════════════════════════════════════════
```

### Level 3: Extreme
```python
debug_patch = DebugPatch(trainer, debug_level=3)
```
**Output:**
- Everything from Level 2
- Data validation for every batch
- NaN/Inf detection
- Value range checks
- Per-operation timing

---

## 🔧 Configuration Options

```python
debug_patch = DebugPatch(
    trainer,
    debug_level=2,      # 0-3, controls verbosity
    log_every_n=10      # Log every N iterations
)
```

### Adjust Logging Frequency

```python
# Log every iteration (very verbose)
debug_patch = DebugPatch(trainer, debug_level=2, log_every_n=1)

# Log every 50 iterations (less verbose)
debug_patch = DebugPatch(trainer, debug_level=2, log_every_n=50)
```

---

## 📊 What Gets Tracked

### Automatically Monitored:

1. **Data Loading**
   - Loader validation before training
   - Batch shape verification
   - Data range checks
   - NaN/Inf detection

2. **Memory Usage**
   - GPU allocation per iteration
   - Memory growth over time
   - Peak memory usage

3. **Timing**
   - Per-iteration time
   - Average step time
   - Total epoch time
   - Throughput (steps/sec)

4. **Training Metrics**
   - Loss values
   - Gradient flow
   - Learning rate (if available)

---

## 🐛 Troubleshooting with Debug Patch

### Problem: Training Stuck at 0%

**Diagnosis:**
```python
from debug_patch import quick_diagnose

# Test your dataloaders
quick_diagnose(flood_dataset, flood_loader)
```

**What to look for:**
```
📊 DataLoader:
   Load time: 45.3s  ← PROBLEM! Should be <1s
   ⚠️  Issues found:
       Batch images out of expected range
```

### Problem: GPU Memory Issues

**Use debug patch to track:**
```python
debug_patch = DebugPatch(trainer, debug_level=2)
# Watch MEMORY USAGE SUMMARY after each epoch
```

**Example output showing leak:**
```
MEMORY USAGE SUMMARY
════════════════════════════════════════
Iter 0     | 8.23GB / 10.45GB
Iter 100   | 12.34GB / 14.56GB  ← Growing!
Iter 200   | 16.78GB / 18.23GB  ← Memory leak
```

### Problem: Slow Training

**Track bottlenecks:**
```python
debug_patch = DebugPatch(trainer, debug_level=2, log_every_n=1)
```

**Look for:**
```
Iteration 50
  Time: 2.345s  ← Slow!
  
Iteration 51
  Time: 0.234s  ← Normal
```

This tells you iteration 50 had an issue.

---

## 💡 Advanced Usage

### Custom Validation

Add your own validation logic:

```python
from debug_patch import DebugPatch, validate_batch

class MyDebugPatch(DebugPatch):
    def _log_iteration(self, iteration, metrics, elapsed):
        # Call parent
        super()._log_iteration(iteration, metrics, elapsed)
        
        # Add your custom checks
        if metrics.get('flood_loss', 0) > 2.0:
            print("⚠️  Flood loss too high!")

# Use your custom patch
debug_patch = MyDebugPatch(trainer, debug_level=2)
```

### Save Debug Logs

```python
import sys

# Redirect output to file
with open('debug_log.txt', 'w') as f:
    sys.stdout = f
    
    debug_patch = DebugPatch(trainer, debug_level=2)
    debug_patch.train(flood_loader, damage_loader)
    
    sys.stdout = sys.__stdout__
```

### Conditional Debugging

```python
# Only debug first few epochs
for epoch in range(config.num_epochs):
    if epoch < 3:
        # Debug enabled
        debug_patch = DebugPatch(trainer, debug_level=2)
        debug_patch.train_epoch(flood_loader, damage_loader, epoch)
    else:
        # Normal training
        trainer.train_epoch(flood_loader, damage_loader, epoch)
```

---

## 🔄 Integration with Existing Code

### Works With:

✅ **Your custom trainer classes**
```python
class MyTrainer:
    def train_epoch(self, loader1, loader2, epoch):
        # Your implementation
        pass

trainer = MyTrainer(model, config)
debug_patch = DebugPatch(trainer)  # Just works!
```

✅ **Existing training loops**
```python
# Your loop
for epoch in range(50):
    trainer.train_epoch(flood_loader, damage_loader, epoch)

# Becomes
debug_patch = DebugPatch(trainer)
for epoch in range(50):
    debug_patch.train_epoch(flood_loader, damage_loader, epoch)
```

✅ **Complex multi-task setups**
```python
# Multiple tasks? No problem
debug_patch = DebugPatch(trainer, debug_level=2)
debug_patch.train(task1_loader, task2_loader, task3_loader)
```

### Doesn't Interfere With:

- Your model architecture
- Loss functions
- Optimizers
- Schedulers
- Checkpointing
- Validation logic

---

## 📋 Complete Example

Here's a full example showing before/after:

### BEFORE (Original Code)

```python
# Original training script
import torch
from torch.utils.data import DataLoader

# Setup
config = Config()
model = EnhancedDisasterModel(...)
trainer = DisasterTrainer(model, config)

# Data
flood_loader = DataLoader(flood_dataset, batch_size=16)
damage_loader = DataLoader(damage_dataset, batch_size=16)

# Train
for epoch in range(config.num_epochs):
    trainer.train_epoch(flood_loader, damage_loader, epoch)
    
print("Training complete!")
```

### AFTER (With Debug Patch)

```python
# Training script with debug patch
import torch
from torch.utils.data import DataLoader
from debug_patch import DebugPatch, quick_diagnose  # ← Added

# Setup (unchanged)
config = Config()
model = EnhancedDisasterModel(...)
trainer = DisasterTrainer(model, config)

# Data (unchanged)
flood_loader = DataLoader(flood_dataset, batch_size=16)
damage_loader = DataLoader(damage_dataset, batch_size=16)

# Quick diagnosis (optional)  ← Added
quick_diagnose(flood_dataset, flood_loader, num_samples=3)

# Wrap trainer  ← Added
debug_patch = DebugPatch(trainer, debug_level=2)

# Train with debugging  ← Modified
for epoch in range(config.num_epochs):
    debug_patch.train_epoch(flood_loader, damage_loader, epoch)
    
print("Training complete!")
```

**Changes made:** 3 lines added, 1 line modified. That's it!

---

## 🎓 Best Practices

### 1. Start with Quick Diagnose

```python
# Before training, always check your data
quick_diagnose(dataset, loader)
```

### 2. Use Level 2 for Development

```python
# Good balance of information vs. speed
debug_patch = DebugPatch(trainer, debug_level=2)
```

### 3. Adjust Logging Frequency

```python
# Fast training? Log less often
debug_patch = DebugPatch(trainer, debug_level=2, log_every_n=50)

# Debugging specific issue? Log more
debug_patch = DebugPatch(trainer, debug_level=3, log_every_n=1)
```

### 4. Remove for Production

```python
# Development
if config.debug:
    debug_patch = DebugPatch(trainer, debug_level=2)
    debug_patch.train(flood_loader, damage_loader)
else:
    # Production
    trainer.train(flood_loader, damage_loader)
```

---

## ⚡ Performance Impact

| Debug Level | Overhead | When to Use |
|-------------|----------|-------------|
| 0 | <1% | Production |
| 1 | ~2-3% | Light debugging |
| 2 | ~5-8% | Development (recommended) |
| 3 | ~10-15% | Deep debugging |

**Note:** Overhead is mostly from printing, not computation.

---

## 🆘 Common Issues

### Issue: "AttributeError: 'Trainer' object has no attribute 'train_epoch'"

**Solution:** Your trainer uses a different method name. Use the manual loop:

```python
debug_patch = DebugPatch(trainer, debug_level=2)

# Use train() instead of train_epoch()
debug_patch.train(flood_loader, damage_loader)
```

### Issue: "Debug output too verbose"

**Solution:** Reduce logging frequency:

```python
debug_patch = DebugPatch(trainer, debug_level=2, log_every_n=100)
```

### Issue: "Want to debug just one epoch"

**Solution:**

```python
# Debug first epoch only
for epoch in range(config.num_epochs):
    if epoch == 0:
        debug_patch = DebugPatch(trainer, debug_level=3)
        debug_patch.train_epoch(flood_loader, damage_loader, epoch)
    else:
        trainer.train_epoch(flood_loader, damage_loader, epoch)
```

---

## 📚 API Reference

### DebugPatch Class

```python
DebugPatch(trainer, debug_level=2, log_every_n=10)
```

**Methods:**
- `train(flood_loader, damage_loader, flood_val_loader=None, damage_val_loader=None, num_epochs=None)`
- `train_epoch(flood_loader, damage_loader, epoch)`

### Helper Functions

```python
quick_diagnose(dataset, dataloader, num_samples=3)
```
Quick diagnostic check for datasets/dataloaders.

```python
inject_debug(trainer, debug_level=2)
```
Convenience function for one-line injection.

```python
validate_batch(images, masks, name="batch")
```
Validate a batch for common issues.

---

## ✅ Checklist

Before training:
- [ ] Copy `debug_patch.py` to your project
- [ ] Import: `from debug_patch import DebugPatch`
- [ ] Wrap trainer: `debug_patch = DebugPatch(trainer, debug_level=2)`
- [ ] Run quick_diagnose on your datasets
- [ ] Start training with `debug_patch.train(...)`

During training:
- [ ] Monitor memory usage in summary
- [ ] Check for increasing iteration times
- [ ] Watch for data validation warnings
- [ ] Verify loss values are reasonable

After training:
- [ ] Review memory summary for leaks
- [ ] Check average step time
- [ ] Look for any warnings that appeared

---

## 🎉 That's It!

The debug patch is designed to be:
- **Easy to add** (3 lines of code)
- **Non-invasive** (doesn't change your code)
- **Comprehensive** (tracks everything important)
- **Flexible** (adjustable debug levels)

Try it out and happy debugging! 🐛

---

**Questions?** Open an issue on GitHub or check the examples above.
