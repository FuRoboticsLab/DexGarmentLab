# Complete Codebase Walkthrough: Multi-Stage Residual RL

## Table of Contents

1. [Evaluation Methods](#1-evaluation-methods)
2. [RL Control Mechanisms](#2-rl-control-mechanisms)
3. [Early Stopping & Anomaly Detection](#3-early-stopping--anomaly-detection)
4. [Training Script Analysis](#4-training-script-analysis)
5. [Issues & Analysis](#5-issues--analysis)

---

## 1. Evaluation Methods

### 1.1 IL (SADP_G) Evaluation

**Location**: `eval_multi_stage.py` → `evaluate_il_only()`

**How it works:**
```python
# IL-only evaluation (baseline)
def evaluate_il_only(env, n_episodes):
    for episode in range(n_episodes):
        obs, info = env.reset()
        while not done:
            # Use ZERO residuals - pure IL action
            action = np.zeros(61, dtype=np.float32)  # [60D residuals=0, 1D advance]
            
            # Simple stage advance heuristic (after N steps)
            if stage_steps >= max_steps:
                action[60] = 1.0  # Advance stage
            
            obs, reward, done, info = env.step(action)
            metrics.step(info, action, reward)
        
        metrics.end_episode(info, obs)
```

**Metrics Collected:**
- Success rate (all 3 stages completed + final success check)
- Stage completion rates (stage 1, 2, 3 individually)
- Mean reward per episode
- Episode length (steps)
- Stage transition steps (when stage 1→2, 2→3 happened)

**Final Success Criteria** (`_check_final_success()`):
```python
def _check_final_success(obs):
    pcd = obs["garment_pcd"]
    current_bbox = compute_bbox(pcd)
    current_size = current_bbox[3:6] - current_bbox[0:3]
    initial_size = initial_bbox[3:6] - initial_bbox[0:3]
    
    x_ratio = current_size[0] / initial_size[0]  # Must be < 0.5 (50% reduction)
    y_ratio = current_size[1] / initial_size[1]  # Must be < 0.7 (30% reduction)
    height_var = np.var(pcd[:, 2])               # Must be < 0.02 (flat)
    
    return (x_ratio < 0.5) and (y_ratio < 0.7) and (height_var < 0.02)
```

**Key Points:**
- IL evaluation uses **zero residuals** (pure IL actions)
- Stage advances happen automatically after fixed step count
- Success requires **all 3 stages completed** AND final success criteria met

---

### 1.2 RL Evaluation

**Location**: `eval_multi_stage.py` → `evaluate_rl_policy()`

**How it works:**
```python
def evaluate_rl_policy(env, model, n_episodes):
    for episode in range(n_episodes):
        obs, info = env.reset()
        while not done:
            # Get action from trained RL policy
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment (RL residuals + IL actions)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            metrics.step(info, action, reward)
        
        metrics.end_episode(info, obs)
```

**Metrics Collected:**
- Same as IL evaluation
- **Additional**: Residual magnitude (how much RL is correcting)
- **Additional**: RL vs proactive stage advances

**Comparison:**
- RL evaluation uses **RL policy** to generate residuals
- Stage advances can be **RL-initiated** or **proactive**
- Success criteria same as IL

---

### 1.3 Quality Evaluation (`_evaluate_current_stage_quality()`)

**Location**: `multi_stage_residual_env.py:1138`

**How it works:**
```python
def _evaluate_current_stage_quality():
    # Get bounding box at stage start
    initial_bbox = _stage_initial_bbox  # Set when stage starts
    current_bbox = compute_bbox(current_garment_pcd)
    
    # Compute area reduction
    initial_area = (initial_bbox[3] - initial_bbox[0]) * \
                  (initial_bbox[4] - initial_bbox[1])
    current_area = (current_bbox[3] - current_bbox[0]) * \
                   (current_bbox[4] - current_bbox[1])
    
    area_reduction = (initial_area - current_area) / initial_area
    return np.clip(area_reduction, 0.0, 1.0)  # [0, 1]
```

**Key Points:**
- Quality = **area reduction** from stage start
- Used for:
  - Stage advance decisions (proactive & RL)
  - Reward computation
  - Plateau detection

**Limitations:**
- Only measures **area**, not fold quality (wrinkles, alignment)
- Doesn't account for **height** (could be folded but lifted)
- Doesn't detect **misalignment** (wrong fold direction)

---

## 2. RL Control Mechanisms

### 2.1 When Does RL Intervene?

**Location**: `multi_stage_residual_env.py:900` → `_should_apply_residual()`

**Decision Flow:**
```python
def _should_apply_residual():
    # Check 1: Sparse application
    if current_step % residual_apply_interval != 0:
        return False  # Only apply every N steps (default: 5)
    
    # Check 2: Phase-aware gating (if enabled)
    if enable_phase_aware_control:
        phase = _detect_manipulation_phase()
        if phase == "manipulation":
            return False  # Pure IL during manipulation
    
    # Check 3: Gripper-based check (fallback)
    elif disable_residual_during_manipulation:
        if _is_gripper_closed():
            return False  # No residuals when grasping
    
    return True  # Apply residual
```

**Current Settings:**
- `residual_apply_interval = 5` → Apply every 5 steps (sparse)
- `enable_phase_aware_control = False` → **DISABLED** (too complex)
- `disable_residual_during_manipulation = True` → Block when gripper closed
- `arm_only_residuals = True` → Only arm joints get residuals (hands = pure IL)

**Key Issue**: RL intervention is **too sparse** (every 5 steps) and **too restricted** (blocked during manipulation). This means RL can't help IL when IL is struggling!

---

### 2.2 Phase Detection

**Location**: `multi_stage_residual_env.py:935` → `_detect_manipulation_phase()`

**How it works:**
```python
def _detect_manipulation_phase():
    stage_steps = il_wrapper.stage_step_count
    gripper_closed = _is_gripper_closed()
    
    # Early in stage + gripper open = approach
    if stage_steps < approach_phase_steps (50) and not gripper_closed:
        return "approach"
    
    # Gripper closed = manipulation
    if gripper_closed:
        return "manipulation"
    
    # Late in stage + gripper open = release
    if stage_steps > approach_phase_steps and not gripper_closed:
        return "release"
    
    return "manipulation"  # Default
```

**Gripper Detection** (`_is_gripper_closed()`):
```python
def _is_gripper_closed():
    left_joints = robot.dexleft.get_joint_positions()
    right_joints = robot.dexright.get_joint_positions()
    
    left_hand = left_joints[6:30]   # Hand joints
    right_hand = right_joints[6:30]
    
    left_avg_flexion = np.mean(np.abs(left_hand))
    right_avg_flexion = np.mean(np.abs(right_hand))
    
    return (left_avg_flexion > gripper_threshold (0.3) or 
            right_avg_flexion > gripper_threshold)
```

**Current Settings:**
- `approach_phase_steps = 50` → First 50 steps = approach
- `gripper_threshold = 0.3` → Threshold for detecting closed gripper
- `enable_phase_aware_control = False` → **DISABLED** (not used)

**Key Issues:**
1. **Too simplistic**: Only uses step count + gripper state
2. **Not accurate**: Gripper detection based on average flexion (unreliable)
3. **Disabled**: Phase-aware control is turned off anyway!

---

### 2.3 Stage Transition Detection

**Location**: `multi_stage_residual_env.py:1167` → `_check_proactive_advance()`

**Proactive Advance (Automatic):**
```python
def _check_proactive_advance():
    stage_steps = il_wrapper.stage_step_count
    stage_quality = _evaluate_current_stage_quality()
    
    # Condition 1: Duration-based
    steps_threshold = il_steps_per_stage (200) * proactive_advance_after_ratio (0.85)
    # = 170 steps
    
    if stage_steps >= steps_threshold:
        if stage_quality >= min_quality_for_proactive (0.1):
            return True
        # Even with low quality, advance after full duration
        if stage_steps >= il_steps_per_stage (200):
            return True
    
    # Condition 2: Quality plateau
    if _detect_quality_plateau() and stage_quality >= min_quality_for_proactive:
        return True
    
    return False
```

**Quality Plateau Detection:**
```python
def _detect_quality_plateau():
    if len(quality_history) < quality_plateau_window (30):
        return False
    
    recent = quality_history[-30:]  # Last 30 steps
    improvement = recent[-1] - recent[0]
    
    return improvement < quality_plateau_threshold (0.02)  # < 2% improvement
```

**RL-Initiated Advance:**
```python
def _handle_stage_advance():
    # Check minimum steps
    if stage_steps < min_steps_before_advance (10):
        return False, -premature_advance_penalty (1.0)
    
    # Check quality threshold
    stage_quality = _evaluate_current_stage_quality()
    if stage_quality < 0.15:  # Must have at least 15% area reduction
        return False, -premature_advance_penalty
    
    # Safe transition (open hands first)
    _safe_release_before_transition()
    
    # Advance
    success = il_wrapper.advance_stage()
    if success:
        return True, stage_advance_bonus (10.0) + stage_completion_bonus (5.0)
    
    return False, 0.0
```

**Current Settings:**
- `proactive_advance_after_ratio = 0.85` → Advance after 85% of IL steps (170/200)
- `min_quality_for_proactive = 0.1` → Only need 10% area reduction
- `min_steps_before_advance = 10` → RL can't advance before 10 steps
- `stage_advance_threshold = 0.5` → RL signal > 0.5 triggers advance attempt
- `quality_plateau_window = 30` → Check last 30 steps for plateau
- `quality_plateau_threshold = 0.02` → < 2% improvement = plateau

**Key Issues:**
1. **Too lenient**: `min_quality_for_proactive = 0.1` is very low (10% area reduction)
2. **Duration-based only**: Advances even with low quality after full duration
3. **Plateau detection weak**: 30 steps window, 2% threshold is too sensitive
4. **RL advance threshold low**: 0.15 quality is too low for good transitions

---

### 2.4 Reward Structure

**Location**: `multi_stage_residual_env.py:1593` → `_compute_reward()`

**Reward Components:**
```python
def _compute_reward(obs, residual, stage_advanced):
    reward = 0.0
    
    # 1. Fold progress (area reduction)
    fold_progress = (initial_area - current_area) / initial_area
    reward += fold_progress * 2.0  # Weight: 2.0
    
    # 2. Compactness (volume reduction)
    compactness = 1.0 - current_vol / initial_vol
    reward += compactness * 1.0  # Weight: 1.0
    
    # 3. Height penalty (should stay flat)
    height_var = np.var(pcd[:, 2])
    reward -= height_var * 0.2  # Weight: 0.2
    
    # 4. Residual penalty (encourage small corrections)
    residual_magnitude = np.sum(residual ** 2)
    reward -= residual_magnitude * 0.01  # Weight: 0.01
    
    # 5. Smoothness penalty (penalize jerky movements)
    action_change = np.sum((current_action - last_action) ** 2)
    reward -= action_change * 0.02  # Weight: 0.02
    
    # 6. Late advance penalty (if should advance but hasn't)
    if should_advance_hint > 0.8 and not stage_advanced:
        reward -= 0.05  # Weight: 0.05
    
    # 7. Task completion bonus
    if all_stages_done and final_success:
        reward += 30.0  # Weight: 30.0
    
    return reward
```

**Stage Advance Rewards:**
- **Proactive advance**: `stage_completion_bonus = 5.0`
- **RL-initiated advance**: `stage_advance_bonus (10.0) + stage_completion_bonus (5.0) + 2.0 bonus = 17.0`
- **Premature advance penalty**: `-1.0`

**Current Reward Weights:**
```python
reward_weights = {
    "fold_progress": 2.0,           # Main reward
    "compactness": 1.0,             # Volume reduction
    "height_penalty": 0.2,          # Keep flat
    "residual_penalty": 0.01,       # Small corrections
    "smoothness_penalty": 0.02,     # Smooth movements
    "stage_advance_bonus": 10.0,    # RL advance bonus
    "premature_advance_penalty": 1.0,  # Too early penalty
    "late_advance_penalty": 0.05,   # Too late penalty
    "stage_completion_bonus": 5.0,  # Stage done bonus
    "task_success_bonus": 30.0,     # Final success bonus
}
```

**Key Issues:**
1. **Reward scale mismatch**: Progress rewards (2.0) vs completion bonus (30.0) is huge
2. **No intermediate rewards**: Only rewards at stage completion, not during folding
3. **Residual penalty too small**: 0.01 encourages large residuals
4. **No quality-based rewards**: Doesn't reward good fold quality, only area reduction

---

## 3. Early Stopping & Anomaly Detection

### 3.1 Early Termination

**Location**: `multi_stage_residual_env.py:1714` → `_check_early_termination()`

**Current Status**: **DISABLED** (`enable_early_termination = False`)

**If Enabled, Checks:**
```python
def _check_early_termination(obs):
    # 1. Garment anomaly (stretched, lifted, out of bounds)
    is_anomaly, anomaly_type = _check_garment_anomaly(obs)
    if is_anomaly:
        return True, f"anomaly_{anomaly_type}"
    
    # 2. No progress for too long
    stage_steps = il_wrapper.stage_step_count
    max_allowed = stage_config.max_inference_steps * 10  # 10x normal (2000 steps!)
    if stage_steps > max_allowed:
        quality = _evaluate_current_stage_quality()
        if quality < 0.0:  # Negative quality (impossible check)
            return True, "no_stage_progress"
    
    # 3. Regression (DISABLED - commented out)
    # if recent_trend < -0.15:
    #     return True, "regressing"
    
    # 4. Very negative progress
    current_progress = recent_fold_progress[-1]
    if current_progress < -0.5:  # Garment 50% larger than initial
        return True, "negative_progress"
    
    return False, ""
```

**Current Settings:**
- `enable_early_termination = False` → **DISABLED**
- `early_termination_penalty = 0.0` → No penalty if enabled

**Key Issues:**
1. **Disabled**: Early termination is turned off, so bad episodes run to completion
2. **Too lenient if enabled**: 10x normal steps (2000 steps!) before checking
3. **Regression check disabled**: Won't catch quality getting worse
4. **Negative progress check too extreme**: -0.5 is very rare (50% larger)

---

### 3.2 Anomaly Detection

**Location**: `multi_stage_residual_env.py:1666` → `_check_garment_anomaly()`

**Checks:**
```python
def _check_garment_anomaly(obs):
    pcd = obs["garment_pcd"]
    
    # 1. Height anomaly: lifted too high
    max_height = np.max(pcd[:, 2])
    if max_height > 1.0:  # More than 1.0m above ground
        return True, "garment_lifted"
    
    # 2. Spread anomaly: stretched too much
    current_size = current_bbox[3:6] - current_bbox[0:3]
    initial_size = initial_bbox[3:6] - initial_bbox[0:3]
    
    if current_size[0] > initial_size[0] * 2.0:  # 100% larger
        return True, "garment_stretched_x"
    if current_size[1] > initial_size[1] * 2.0:
        return True, "garment_stretched_y"
    
    # 3. Position anomaly: out of workspace
    center = np.mean(pcd, axis=0)
    if abs(center[0]) > 2.0:  # Way too far left/right
        return True, "garment_out_of_bounds_x"
    if center[1] < -0.5 or center[1] > 2.5:  # Way too far forward/back
        return True, "garment_out_of_bounds_y"
    
    # 4. Dispersion anomaly: points too spread out
    point_std = np.std(pcd, axis=0)
    if np.max(point_std) > 1.0:  # Very extreme variance
        return True, "garment_dispersed"
    
    return False, ""
```

**Current Thresholds (RELAXED):**
- Height: `1.0m` (was 0.5m) - very high
- Stretch: `2.0x` (was 1.3x) - 100% larger
- Out of bounds: `2.0m` (was 1.0m) - very far
- Dispersion: `1.0` (was 0.5) - very spread out

**Key Issues:**
1. **Too relaxed**: Thresholds are very high, won't catch moderate problems
2. **Only extreme cases**: Only catches catastrophic failures
3. **No intermediate warnings**: No soft penalties for approaching thresholds

---

## 4. Issues & Analysis

### 4.1 Issue: RL Not Leveraging IL Good Enough

**Root Causes:**

1. **Residual Application Too Sparse**
   - `residual_apply_interval = 5` → Only every 5 steps
   - RL can't provide continuous guidance
   - **Fix**: Reduce to 1-2 steps, or make it adaptive

2. **Residuals Blocked During Manipulation**
   - `disable_residual_during_manipulation = True`
   - When IL is struggling (gripper closed), RL can't help
   - **Fix**: Allow small residuals even during manipulation, or better phase detection

3. **Residual Scale Too Small**
   - `arm_residual_scale = 0.05` → Very small corrections
   - RL can't make meaningful adjustments
   - **Fix**: Increase to 0.1-0.15, or make adaptive based on IL confidence

4. **No IL Confidence Signal**
   - RL doesn't know when IL is uncertain
   - Can't increase residuals when IL needs help
   - **Fix**: Add IL action variance or confidence to observation

5. **Reward Doesn't Encourage RL-IL Cooperation**
   - Residual penalty (0.01) is too small
   - No reward for helping IL succeed
   - **Fix**: Add reward for IL success, reduce residual penalty when IL struggling

---

### 4.2 Issue: Phase Detection and Stage Transition Not Good Enough

**Root Causes:**

1. **Phase Detection Too Simplistic**
   - Only uses: step count + gripper state
   - Doesn't consider: garment state, IL action patterns, actual manipulation progress
   - **Fix**: Use IL action patterns, garment deformation, actual contact detection

2. **Gripper Detection Unreliable**
   - Based on average finger flexion
   - Threshold (0.3) may not match actual grasp state
   - **Fix**: Use contact sensors, force feedback, or more sophisticated detection

3. **Stage Transition Quality Threshold Too Low**
   - `min_quality_for_proactive = 0.1` → Only 10% area reduction
   - `RL advance threshold = 0.15` → Only 15% area reduction
   - Allows transitions when fold is incomplete
   - **Fix**: Increase to 0.3-0.4 for good transitions

4. **Duration-Based Advance Too Aggressive**
   - Advances after 85% duration even with low quality
   - Doesn't wait for IL to finish properly
   - **Fix**: Require quality threshold even after duration, or increase duration threshold

5. **Plateau Detection Weak**
   - 30-step window may miss longer trends
   - 2% threshold too sensitive (noise can trigger)
   - **Fix**: Longer window (50-100 steps), higher threshold (5-10%)

6. **No Visual Quality Assessment**
   - Only uses area reduction, not actual fold quality
   - Can't detect misalignment, wrinkles, wrong fold direction
   - **Fix**: Add visual quality metrics (VLA features, wrinkle detection, alignment)

---

### 4.3 Issue: Bad Examples Marked as Done, Bad Examples Not Stopped Early

**Root Causes:**

1. **Final Success Criteria Too Lenient**
   ```python
   x_ratio < 0.5  # 50% reduction (OK)
   y_ratio < 0.7  # 30% reduction (too lenient!)
   height_var < 0.02  # Flat (OK)
   ```
   - `y_ratio < 0.7` allows 70% of original Y size (not well folded)
   - **Fix**: Tighten to `y_ratio < 0.5` or add visual quality check

2. **Early Termination Disabled**
   - `enable_early_termination = False`
   - Bad episodes run to completion, marked as "done"
   - **Fix**: Enable with better thresholds, or add soft penalties

3. **Anomaly Detection Too Relaxed**
   - Thresholds are 2x-4x what they should be
   - Only catches catastrophic failures
   - **Fix**: Lower thresholds, add intermediate warnings

4. **No Quality-Based Termination**
   - Doesn't check if fold quality is actually good
   - Can complete all stages with poor quality
   - **Fix**: Add quality check to final success, or intermediate quality checks

5. **Stage Completion Doesn't Check Quality**
   - Stages marked complete based on duration/plateau, not quality
   - `min_quality_for_proactive = 0.1` is too low
   - **Fix**: Require minimum quality (0.3-0.4) for stage completion

6. **No Regression Detection**
   - Regression check is disabled
   - Can't catch quality getting worse
   - **Fix**: Re-enable with better thresholds, or add trend-based penalties

---

## Summary of Key Problems

### RL Not Leveraging IL:
- ✅ Residuals too sparse (every 5 steps)
- ✅ Blocked during manipulation (can't help when needed)
- ✅ Scale too small (0.05)
- ✅ No IL confidence signal
- ✅ Reward doesn't encourage cooperation

### Phase/Stage Detection:
- ✅ Phase detection too simplistic
- ✅ Gripper detection unreliable
- ✅ Quality thresholds too low (0.1, 0.15)
- ✅ Duration-based advance too aggressive
- ✅ Plateau detection weak
- ✅ No visual quality assessment

### Early Stopping:
- ✅ Early termination disabled
- ✅ Anomaly detection too relaxed
- ✅ Final success criteria too lenient
- ✅ No quality-based termination
- ✅ Stage completion doesn't check quality
- ✅ No regression detection

---

## Recommended Fixes

### Priority 1: Enable RL to Help IL
1. Reduce `residual_apply_interval` to 1-2
2. Allow small residuals during manipulation (scale down, don't block)
3. Increase `arm_residual_scale` to 0.1-0.15
4. Add IL confidence to observation
5. Reward RL for IL success

### Priority 2: Improve Stage Transitions
1. Increase quality thresholds (0.3-0.4)
2. Require quality even after duration
3. Improve plateau detection (longer window, higher threshold)
4. Add visual quality assessment (VLA features)
5. Better phase detection (IL action patterns, garment state)

### Priority 3: Better Early Stopping
1. Enable early termination with better thresholds
2. Lower anomaly detection thresholds
3. Tighten final success criteria
4. Add quality-based termination
5. Re-enable regression detection

---

This walkthrough covers all the key mechanisms. The main issues are: **RL is too restricted**, **quality thresholds are too low**, and **early stopping is disabled/too lenient**.

