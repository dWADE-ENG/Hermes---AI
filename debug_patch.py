"""
Hermes Debug Patch v1.0 - Injectable Debugging Module
=====================================================

This module provides drop-in debug capabilities for existing Hermes training code.
Simply import and wrap your trainer to enable comprehensive debugging.

Usage:
    from debug_patch import DebugPatch
    
    # Wrap your existing trainer
    debug_patch = DebugPatch(trainer, debug_level=2)
    debug_patch.train(flood_loader, damage_loader)

Author: Hermes Team
Date: 2025-01-17
"""

import time
import sys
import torch
import numpy as np
from contextlib import contextmanager
from collections import defaultdict


class DebugTimer:
    """Simple timer for profiling code blocks"""
    def __init__(self, name, verbose=True):
        self.name = name
        self.verbose = verbose
        
    def __enter__(self):
        self.start = time.time()
        if self.verbose:
            print(f"⏱️  {self.name}...")
        return self
        
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.verbose:
            print(f"   Completed in {self.elapsed:.3f}s")


class MemoryTracker:
    """Track GPU memory usage"""
    def __init__(self):
        self.snapshots = []
        
    def snapshot(self, label=""):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            self.snapshots.append({
                'label': label,
                'allocated_gb': allocated,
                'reserved_gb': reserved
            })
            return allocated, reserved
        return 0, 0
        
    def print_summary(self):
        if not self.snapshots:
            return
            
        print("\n" + "="*60)
        print("MEMORY USAGE SUMMARY")
        print("="*60)
        for snap in self.snapshots:
            print(f"{snap['label']:30s} | {snap['allocated_gb']:.2f}GB / {snap['reserved_gb']:.2f}GB")
        print("="*60 + "\n")
        
    def clear(self):
        self.snapshots.clear()


@contextmanager
def monitored_timeout(duration, operation_name="Operation"):
    """
    Context manager that monitors operation time
    Works in both notebook and script environments
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    
    if elapsed > duration:
        print(f"⚠️  {operation_name} took {elapsed:.1f}s (expected <{duration}s)")


def validate_batch(images, masks, name="batch"):
    """Validate batch data for common issues"""
    issues = []
    
    # Check tensors
    if not isinstance(images, torch.Tensor):
        issues.append(f"{name} images not a tensor")
    if not isinstance(masks, torch.Tensor):
        issues.append(f"{name} masks not a tensor")
        
    if isinstance(images, torch.Tensor):
        # Check for NaN/Inf
        if torch.isnan(images).any():
            issues.append(f"{name} images contain NaN")
        if torch.isinf(images).any():
            issues.append(f"{name} images contain Inf")
            
        # Check value range
        if images.min() < -1 or images.max() > 2:
            issues.append(f"{name} images out of expected range [{images.min():.2f}, {images.max():.2f}]")
    
    if isinstance(masks, torch.Tensor):
        # Check mask values
        unique_vals = torch.unique(masks)
        if len(unique_vals) > 10:
            issues.append(f"{name} masks have {len(unique_vals)} unique values (expected <10)")
    
    return issues


class DebugPatch:
    """
    Injectable debug wrapper for existing Hermes trainers
    
    This class wraps your existing trainer and adds comprehensive debugging
    without requiring code rewrites.
    """
    
    def __init__(self, trainer, debug_level=2, log_every_n=10):
        """
        Initialize debug patch
        
        Args:
            trainer: Your existing trainer instance
            debug_level: 0=off, 1=basic, 2=verbose, 3=extreme
            log_every_n: Log every N iterations
        """
        self.trainer = trainer
        self.debug_level = debug_level
        self.log_every_n = log_every_n
        
        # Tracking
        self.memory_tracker = MemoryTracker() if debug_level >= 1 else None
        self.step_times = defaultdict(list)
        self.iteration_times = []
        
        print(f"\n{'='*60}")
        print(f"🐛 DEBUG PATCH ACTIVATED (Level {debug_level})")
        print(f"{'='*60}")
        print(f"Wrapping trainer: {type(trainer).__name__}")
        print(f"Log frequency: Every {log_every_n} iterations")
        print(f"Memory tracking: {'Enabled' if self.memory_tracker else 'Disabled'}")
        print(f"{'='*60}\n")
    
    def _test_dataloaders(self, flood_loader, damage_loader):
        """Test dataloaders before training"""
        if self.debug_level < 1:
            return
            
        print("\n🔍 Testing DataLoaders...")
        
        # Test flood loader
        try:
            with monitored_timeout(30, "Flood loader test"):
                batch = next(iter(flood_loader))
                print(f"  ✅ Flood loader OK: {batch[0].shape}")
                
                # Validate data
                issues = validate_batch(batch[0], batch[1], "Flood")
                if issues:
                    print(f"  ⚠️  Flood loader issues:")
                    for issue in issues:
                        print(f"      - {issue}")
        except Exception as e:
            print(f"  ❌ Flood loader FAILED: {e}")
            raise
        
        # Test damage loader
        try:
            with monitored_timeout(30, "Damage loader test"):
                batch = next(iter(damage_loader))
                print(f"  ✅ Damage loader OK: {batch[0].shape}")
                
                # Validate data
                issues = validate_batch(batch[0], batch[1], "Damage")
                if issues:
                    print(f"  ⚠️  Damage loader issues:")
                    for issue in issues:
                        print(f"      - {issue}")
        except Exception as e:
            print(f"  ❌ Damage loader FAILED: {e}")
            raise
        
        print()
    
    def _log_iteration(self, iteration, metrics, elapsed):
        """Log iteration metrics"""
        if self.debug_level < 2:
            return
            
        if iteration % self.log_every_n == 0:
            print(f"\nIteration {iteration}")
            print(f"  Flood loss: {metrics.get('flood_loss', 0):.4f}")
            print(f"  Damage loss: {metrics.get('damage_loss', 0):.4f}")
            print(f"  Time: {elapsed:.3f}s")
            
            if self.memory_tracker:
                alloc, reserved = self.memory_tracker.snapshot(f"Iter {iteration}")
                print(f"  GPU: {alloc:.2f}GB / {reserved:.2f}GB")
    
    def _wrap_train_step(self, original_step_fn):
        """Wrap the training step function with debugging"""
        def wrapped_step(*args, **kwargs):
            step_start = time.time()
            
            # Call original step
            result = original_step_fn(*args, **kwargs)
            
            # Track timing
            step_time = time.time() - step_start
            self.step_times['total'].append(step_time)
            
            return result
        
        return wrapped_step
    
    def train_epoch(self, flood_loader, damage_loader, epoch):
        """
        Debug-wrapped epoch training
        
        This wraps your trainer's existing train_epoch method
        """
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1} (Debug Mode)")
        print(f"{'='*60}")
        
        # Test loaders first
        if self.debug_level >= 1 and epoch == 0:
            self._test_dataloaders(flood_loader, damage_loader)
        
        # Memory snapshot at start
        if self.memory_tracker:
            self.memory_tracker.snapshot(f"Epoch {epoch} start")
        
        # Call original trainer's train_epoch
        epoch_start = time.time()
        
        try:
            result = self.trainer.train_epoch(flood_loader, damage_loader, epoch)
        except AttributeError:
            # If train_epoch doesn't exist, try manual training
            print("⚠️  No train_epoch method found, attempting manual training loop...")
            result = self._manual_train_epoch(flood_loader, damage_loader, epoch)
        
        epoch_time = time.time() - epoch_start
        
        # Summary
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1} COMPLETE")
        print(f"{'='*60}")
        print(f"Time: {epoch_time:.1f}s ({epoch_time/60:.1f}m)")
        
        if self.step_times['total']:
            avg_step = np.mean(self.step_times['total'])
            print(f"Avg step time: {avg_step:.3f}s")
            print(f"Steps per second: {1/avg_step:.2f}")
        
        if self.memory_tracker:
            self.memory_tracker.print_summary()
            self.memory_tracker.clear()
        
        print(f"{'='*60}\n")
        
        return result
    
    def _manual_train_epoch(self, flood_loader, damage_loader, epoch):
        """
        Fallback manual training loop with debugging
        Use this if your trainer doesn't have a train_epoch method
        """
        self.trainer.model.train()
        
        from itertools import cycle
        from tqdm import tqdm
        
        flood_cycle = cycle(flood_loader)
        damage_cycle = cycle(damage_loader)
        
        num_iterations = max(len(flood_loader), len(damage_loader))
        epoch_metrics = defaultdict(float)
        
        self.trainer.optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(range(num_iterations), desc=f"Epoch {epoch+1}")
        
        for i in pbar:
            iter_start = time.time()
            
            try:
                # Get batches
                flood_batch = next(flood_cycle)
                damage_batch = next(damage_cycle)
                
                # Validate if needed
                if self.debug_level >= 3 and i % self.log_every_n == 0:
                    flood_issues = validate_batch(flood_batch[0], flood_batch[1], f"Flood iter {i}")
                    damage_issues = validate_batch(damage_batch[0], damage_batch[1], f"Damage iter {i}")
                    
                    if flood_issues or damage_issues:
                        print(f"\n⚠️  Data validation issues at iteration {i}:")
                        for issue in flood_issues + damage_issues:
                            print(f"    {issue}")
                
                # Training step - try to call existing method
                if hasattr(self.trainer, '_simultaneous_disaster_step'):
                    metrics = self.trainer._simultaneous_disaster_step(flood_batch, damage_batch, i)
                else:
                    # Need to implement basic step
                    metrics = self._basic_training_step(flood_batch, damage_batch, i)
                
                # Accumulate metrics
                for k, v in metrics.items():
                    epoch_metrics[k] += v
                
                # Track timing
                iter_time = time.time() - iter_start
                self.iteration_times.append(iter_time)
                
                # Logging
                self._log_iteration(i, metrics, iter_time)
                
                # Update progress bar
                if i % 10 == 0:
                    avg_flood = epoch_metrics['flood_loss'] / (i + 1)
                    avg_damage = epoch_metrics['damage_loss'] / (i + 1)
                    pbar.set_postfix({
                        'flood': f'{avg_flood:.4f}',
                        'damage': f'{avg_damage:.4f}'
                    })
            
            except Exception as e:
                print(f"\n❌ Error at iteration {i}: {e}")
                if self.debug_level >= 1:
                    import traceback
                    traceback.print_exc()
                raise
        
        return epoch_metrics
    
    def _basic_training_step(self, flood_batch, damage_batch, step):
        """
        Basic training step implementation
        Override this if your trainer has a different structure
        """
        device = next(self.trainer.model.parameters()).device
        
        flood_images, flood_masks = flood_batch
        flood_images = flood_images.to(device, non_blocking=True)
        flood_masks = flood_masks.to(device, non_blocking=True)
        
        damage_images, damage_masks = damage_batch
        damage_images = damage_images.to(device, non_blocking=True)
        damage_masks = damage_masks.to(device, non_blocking=True)
        
        # Forward passes
        with torch.amp.autocast('cuda', enabled=True):
            flood_out, _ = self.trainer.model(flood_images)
            _, damage_out = self.trainer.model(damage_images)
            
            # Losses
            flood_loss, _ = self.trainer.flood_loss_fn(flood_out, flood_masks)
            damage_loss, _ = self.trainer.damage_loss_fn(damage_out, damage_masks)
        
        # Backward
        total_loss = (flood_loss + damage_loss) / 2
        
        if hasattr(self.trainer, 'scaler') and self.trainer.scaler:
            self.trainer.scaler.scale(total_loss).backward()
            
            if (step + 1) % getattr(self.trainer.config, 'accumulation_steps', 1) == 0:
                self.trainer.scaler.step(self.trainer.optimizer)
                self.trainer.scaler.update()
                self.trainer.optimizer.zero_grad(set_to_none=True)
        else:
            total_loss.backward()
            
            if (step + 1) % getattr(self.trainer.config, 'accumulation_steps', 1) == 0:
                self.trainer.optimizer.step()
                self.trainer.optimizer.zero_grad(set_to_none=True)
        
        return {
            'flood_loss': flood_loss.item(),
            'damage_loss': damage_loss.item()
        }
    
    def train(self, flood_train_loader, damage_train_loader, 
              flood_val_loader=None, damage_val_loader=None, num_epochs=None):
        """
        Debug-wrapped full training loop
        
        Args:
            flood_train_loader: DataLoader for flood training data
            damage_train_loader: DataLoader for damage training data
            flood_val_loader: Optional validation loader for flood
            damage_val_loader: Optional validation loader for damage
            num_epochs: Number of epochs (uses config if not provided)
        """
        if num_epochs is None:
            num_epochs = getattr(self.trainer.config, 'num_epochs', 50)
        
        print(f"\n{'#'*60}")
        print(f"🐛 DEBUG TRAINING START")
        print(f"{'#'*60}")
        print(f"Epochs: {num_epochs}")
        print(f"Debug level: {self.debug_level}")
        print(f"{'#'*60}\n")
        
        for epoch in range(num_epochs):
            try:
                # Training
                self.train_epoch(flood_train_loader, damage_train_loader, epoch)
                
                # Validation
                if flood_val_loader and damage_val_loader:
                    if hasattr(self.trainer, 'validate'):
                        print("\n📊 Running validation...")
                        self.trainer.validate(flood_val_loader, 'flood', 2)
                        self.trainer.validate(damage_val_loader, 'damage', 4)
                
                # Cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except KeyboardInterrupt:
                print("\n⚠️  Training interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Fatal error in epoch {epoch}: {e}")
                if self.debug_level >= 1:
                    import traceback
                    traceback.print_exc()
                break
        
        print(f"\n{'#'*60}")
        print(f"🐛 DEBUG TRAINING COMPLETE")
        print(f"{'#'*60}\n")


def quick_diagnose(dataset, dataloader, num_samples=3):
    """
    Quick diagnostic function for datasets and dataloaders
    
    Usage:
        from debug_patch import quick_diagnose
        quick_diagnose(my_dataset, my_loader)
    """
    print(f"\n{'='*60}")
    print(f"QUICK DIAGNOSTIC")
    print(f"{'='*60}")
    
    # Test dataset
    print(f"\n📦 Dataset: {type(dataset).__name__}")
    print(f"   Length: {len(dataset)}")
    
    for i in range(min(num_samples, len(dataset))):
        try:
            start = time.time()
            img, mask = dataset[i]
            elapsed = time.time() - start
            
            print(f"\n   Sample {i}:")
            print(f"     Image: {img.shape} {img.dtype}")
            print(f"     Mask: {mask.shape} {mask.dtype}")
            print(f"     Load time: {elapsed:.3f}s")
            
            # Validate
            issues = validate_batch(img.unsqueeze(0), mask.unsqueeze(0), f"Sample {i}")
            if issues:
                for issue in issues:
                    print(f"     ⚠️  {issue}")
        except Exception as e:
            print(f"   ❌ Sample {i} failed: {e}")
    
    # Test dataloader
    print(f"\n📊 DataLoader:")
    print(f"   Batch size: {dataloader.batch_size}")
    print(f"   Num workers: {dataloader.num_workers}")
    
    try:
        start = time.time()
        batch = next(iter(dataloader))
        elapsed = time.time() - start
        
        print(f"\n   First batch:")
        print(f"     Images: {batch[0].shape}")
        print(f"     Masks: {batch[1].shape}")
        print(f"     Load time: {elapsed:.3f}s")
        
        # Validate
        issues = validate_batch(batch[0], batch[1], "Batch")
        if issues:
            print(f"   ⚠️  Issues found:")
            for issue in issues:
                print(f"       {issue}")
        else:
            print(f"   ✅ Batch validation passed")
            
    except Exception as e:
        print(f"   ❌ DataLoader failed: {e}")
    
    print(f"\n{'='*60}\n")


# Convenience function
def inject_debug(trainer, debug_level=2):
    """
    Convenience function to quickly inject debug capabilities
    
    Usage:
        trainer = inject_debug(trainer, debug_level=2)
        trainer.train(flood_loader, damage_loader)
    """
    return DebugPatch(trainer, debug_level=debug_level)


if __name__ == "__main__":
    print("""
    Hermes Debug Patch v1.0
    =======================
    
    This is an injectable debug module. Import it in your training script:
    
        from debug_patch import DebugPatch
        
        # Wrap your trainer
        debug_patch = DebugPatch(trainer, debug_level=2)
        debug_patch.train(flood_loader, damage_loader)
    
    Or use the convenience function:
    
        from debug_patch import inject_debug
        
        trainer = inject_debug(trainer, debug_level=2)
        trainer.train(flood_loader, damage_loader)
    """)
