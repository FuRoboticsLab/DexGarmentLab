# Simplified Multi-Stage Residual RL Environment

## Overview

This is a **simplified version** of the multi-stage residual RL environment that removes complex mechanisms to focus on core learning. It's designed to be easier to train, debug, and understand.

## What Was Removed?

1. ❌ **Phase-aware control** - No complex gating based on manipulation phases
2. ❌ **Early termination** - No aggressive termination conditions
3. ❌ **Complex residual gating** - No sparse application or phase-based blocking
4. ❌ **Complex reward shaping** - Fewer reward components to tune

## What Was Kept?

1. ✅ **Multi-stage IL integration** - Still uses 3 SADP_G models
2. ✅ **Proactive advance** - Automatically advances when IL is done
3. ✅ **Simple residuals** - Arm-only residuals (hands are 100% IL)
4. ✅ **Core rewards** - Fold progress, compactness, stage completion

## Files

- **`multi_stage_residual_env_simple.py`** - Simplified environment
- **`train_multi_stage_ppo_simple.py`** - Training script
- **`ALGORITHM_WALKTHROUGH.md`** - Complete algorithm explanation

## Usage

### Training

```bash
# With dummy IL (for testing)
python Env_RL/train_multi_stage_ppo_simple.py --use-dummy-il

# With actual SADP_G checkpoints
python Env_RL/train_multi_stage_ppo_simple.py \
    --training-data-num 100 \
    --stage-1-checkpoint 1500 \
    --stage-2-checkpoint 1500 \
    --stage-3-checkpoint 1500 \
    --arm-residual-scale 0.05 \
    --total-timesteps 200000
```

### Key Parameters

- `--arm-residual-scale`: Max residual for arm joints (default: 0.05)
- `--total-timesteps`: Total training steps (default: 200000)
- `--learning-rate`: PPO learning rate (default: 3e-4)
- `--device`: Device (default: cuda:0)

## Comparison: Simplified vs Original

| Feature | Original | Simplified |
|---------|----------|------------|
| **Phase-aware control** | ✅ Yes | ❌ No |
| **Early termination** | ✅ Yes | ❌ No |
| **Residual gating** | ✅ Sparse + phase-based | ❌ Every step, simple |
| **Reward components** | 10+ | 5 |
| **Proactive advance** | ✅ Yes | ✅ Yes |
| **Arm-only residuals** | ✅ Yes | ✅ Yes |
| **Safe transitions** | ✅ Yes | ✅ Yes |

## Expected Behavior

1. **IL executes stages** - Each stage's SADP_G model runs
2. **RL applies small corrections** - Arm positioning adjustments
3. **Proactive advance** - Automatically advances after 85% of IL duration
4. **RL can learn** - Can advance earlier if beneficial

## Monitoring

The environment prints:
- Step-by-step progress (every 50 steps)
- Stage transitions (proactive vs RL)
- Episode summaries (stages completed, quality, reward)

## Algorithm Flow

See **`ALGORITHM_WALKTHROUGH.md`** for complete details. Quick summary:

```
1. Observation → RL sees state + IL guidance
2. Action → RL outputs residuals + stage signal
3. Execution → IL action + RL residuals → robot moves
4. Reward → Progress, compactness, penalties
5. Update → PPO updates policy using GAE
```

## Troubleshooting

**Problem:** No stages completing
- **Solution:** Check IL models are loading correctly
- **Solution:** Increase `--arm-residual-scale` (try 0.1)

**Problem:** Training unstable
- **Solution:** Reduce `--arm-residual-scale` (try 0.02)
- **Solution:** Check reward scaling in environment

**Problem:** RL not learning
- **Solution:** Check observation space (NaN values?)
- **Solution:** Check reward signal (should be positive for progress)

## Next Steps

1. **Train simplified version** - See if it learns better
2. **Compare with original** - Which performs better?
3. **Add complexity back** - Only if simplified version works

## Questions?

See **`ALGORITHM_WALKTHROUGH.md`** for detailed explanations of:
- How PPO updates work
- How rewards are computed
- How stage transitions happen
- Complete step-by-step flow

