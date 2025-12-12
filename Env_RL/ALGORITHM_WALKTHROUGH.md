# Multi-Stage Residual RL Algorithm: Complete Walkthrough

This document explains the **entire algorithm** from observation → action → reward → PPO update, including all detection mechanisms and reward computation.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Observation Flow](#observation-flow)
3. [Action Flow](#action-flow)
4. [Reward Computation](#reward-computation)
5. [PPO Update Process](#ppo-update-process)
6. [Stage Transition Logic](#stage-transition-logic)
7. [Complete Step-by-Step Flow](#complete-step-by-step-flow)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RL Policy (PPO)                           │
│  Input: Observation (Dict)                                   │
│  Output: Action [residuals(60D), stage_advance(1D)]         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Multi-Stage Residual Environment                │
│                                                               │
│  1. Get IL action from current stage's SADP_G               │
│  2. Apply RL residuals to IL action                         │
│  3. Execute combined action                                  │
│  4. Compute reward                                           │
│  5. Check stage transitions                                  │
│  6. Return (obs, reward, done, info)                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Isaac Sim Physics Simulation                    │
│  - Robot joints move                                         │
│  - Garment deforms                                           │
│  - Physics updates                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Observation Flow

### 1. **Observation Collection** (`_get_observation()`)

The environment collects multiple data sources:

```python
obs = {
    # Garment state
    "garment_pcd": (2048, 3)      # Point cloud from camera
    "gam_keypoints": (6, 3)       # Manipulation keypoints from GAM
    
    # Robot state
    "joint_positions": (60,)       # Current joint angles
    "ee_poses": (14,)              # End-effector positions/orientations
    
    # IL guidance
    "il_action": (60,)            # What IL wants to do (from SADP_G)
    
    # Stage information
    "current_stage": (4,)          # One-hot [stage1, stage2, stage3, completed]
    "stage_progress": (1,)         # Progress in current stage [0, 1]
    "stages_completed": (3,)       # Binary mask of completed stages
    "should_advance_hint": (1,)    # Hint: should RL advance? [0, 1]
}
```

### 2. **Point Cloud Collection** (`_update_garment_pcd_and_gam()`)

**Step-by-step:**

1. **Hide robot** (to avoid occlusions):
   ```python
   set_prim_visible_group(robot_paths, visible=False)
   ```

2. **Capture point cloud**:
   ```python
   pcd, _ = garment_camera.get_point_cloud_data_from_segment()
   ```

3. **Normalize to fixed size** (2048 points):
   ```python
   if len(pcd) > 2048:
       idx = np.random.choice(len(pcd), 2048, replace=False)
   else:
       idx = np.random.choice(len(pcd), 2048, replace=True)
   normalized_pcd = pcd[idx]
   ```

4. **Get GAM keypoints**:
   ```python
   keypoints, _, similarity = gam_model.get_manipulation_points(
       input_pcd=normalized_pcd,
       index_list=[957, 501, 1902, 448, 1196, 422]  # Stage-specific indices
   )
   ```

5. **NaN Protection**:
   ```python
   if np.isnan(normalized_pcd).any():
       normalized_pcd = np.nan_to_num(normalized_pcd, nan=0.0)
   ```

### 3. **Stage Information** (`_il_wrapper.get_stage_info()`)

```python
current_stage = FoldingStage.STAGE_1_LEFT_SLEEVE  # Enum
stage_step_count = 45  # Steps in current stage
stage_progress = stage_step_count / il_steps_per_stage  # [0, 1]

# One-hot encoding
stage_one_hot = [1, 0, 0, 0]  # Stage 1 active
stages_completed = [True, False, False]  # Stage 1 done
```

### 4. **Advance Hint** (`_get_should_advance_hint()`)

Helps RL learn when to advance:

```python
def _get_should_advance_hint():
    stage_steps = il_wrapper.stage_step_count
    stage_quality = _evaluate_current_stage_quality()
    
    step_ratio = stage_steps / il_steps_per_stage  # [0, 1]
    quality_ratio = stage_quality / 0.3            # [0, 1]
    
    combined = (step_ratio + quality_ratio) / 2.0
    
    # If proactive conditions met, hint = 1.0
    if _check_proactive_advance():
        combined = 1.0
    
    return np.clip(combined, 0.0, 1.0)
```

---

## Action Flow

### 1. **RL Policy Output**

PPO outputs a continuous action vector:

```python
action = rl_policy.predict(obs)  # Shape: (61,)
# action[0:60] = joint_residuals
# action[60] = stage_advance_signal
```

### 2. **Parse Action** (`step()`)

```python
residual_action = action[:60]      # Joint residuals
stage_advance_signal = action[60]  # Stage advance signal
```

### 3. **Get IL Action** (`_get_il_joint_action()`)

```python
# Get observation for IL
il_obs = {
    "agent_pos": joint_positions,
    "environment_point_cloud": env_pcd,
    "garment_point_cloud": garment_pcd,
    "points_affordance_feature": affordance,
}

# Get action from current stage's SADP_G model
il_action = il_wrapper.get_single_step_action(il_obs)  # Shape: (60,)
```

**Key Point:** Each stage has its own SADP_G model:
- `stage_1_model`: Folds left sleeve
- `stage_2_model`: Folds right sleeve
- `stage_3_model`: Folds bottom up

### 4. **Apply Residuals** (`_get_arm_only_residual()`)

**SIMPLIFIED VERSION:**
```python
def _get_arm_only_residual(residual_action):
    arm_residual = residual_action.copy()
    # Zero out hand joints - IL controls hands 100%
    arm_residual[6:30] = 0.0    # Left hand (24 joints)
    arm_residual[36:60] = 0.0   # Right hand (24 joints)
    return arm_residual

final_joint_action = il_action + arm_residual
```

**Joint Layout:**
```
[0:6]   = Left arm (6 DOF)
[6:30]  = Left hand (24 DOF) ← ZEROED (IL only)
[30:36] = Right arm (6 DOF)
[36:60] = Right hand (24 DOF) ← ZEROED (IL only)
```

### 5. **Execute Action** (`_execute_joint_action()`)

```python
# Convert to Isaac Sim format
action_left = ArticulationAction(joint_positions=final_joint_action[:30])
action_right = ArticulationAction(joint_positions=final_joint_action[30:])

# Apply to robot
robot.dexleft.apply_action(action_left)
robot.dexright.apply_action(action_right)

# Step physics (5 steps for smooth motion)
for _ in range(5):
    base_env.step()
```

---

## Reward Computation

### 1. **Reward Components** (`_compute_reward()`)

```python
reward = 0.0

# 1. Fold Progress (main reward)
initial_xy = initial_bbox_size[0] * initial_bbox_size[1]
current_xy = current_bbox_size[0] * current_bbox_size[1]
fold_progress = (initial_xy - current_xy) / (initial_xy + 1e-6)
reward += fold_progress * 2.0  # Weight: 2.0

# 2. Compactness
initial_vol = np.prod(initial_bbox_size)
current_vol = np.prod(current_bbox_size)
compactness = 1.0 - current_vol / (initial_vol + 1e-6)
reward += compactness * 1.0  # Weight: 1.0

# 3. Height Penalty (flattening)
height_var = np.var(garment_pcd[:, 2])  # Z-axis variance
reward -= height_var * 0.2  # Weight: 0.2

# 4. Residual Penalty (encourage small corrections)
residual_magnitude = np.sum(residual ** 2)
reward -= residual_magnitude * 0.01  # Weight: 0.01

# 5. Task Success Bonus (if all stages done)
if all(stages_completed) and _check_final_success():
    reward += 30.0  # Large bonus!
```

### 2. **Stage Advance Rewards**

**Proactive Advance:**
```python
if proactive_advance_triggered:
    reward += 5.0  # Stage completion bonus
```

**RL-Initiated Advance:**
```python
if rl_advance_triggered:
    if quality_ok and steps_ok:
        reward += 5.0  # Stage completion bonus
        reward += 2.0  # Bonus for RL learning
    else:
        reward -= 1.0  # Premature advance penalty
```

### 3. **Reward Flow Diagram**

```
┌─────────────────────────────────────────┐
│         Reward Computation               │
├─────────────────────────────────────────┤
│                                         │
│  Fold Progress:     +2.0 * progress     │
│  Compactness:       +1.0 * compactness  │
│  Height Penalty:    -0.2 * height_var  │
│  Residual Penalty:  -0.01 * ||res||²   │
│                                         │
│  ─────────────────────────────────────  │
│                                         │
│  Stage Advance:     +5.0 (if successful) │
│  RL Advance Bonus:  +2.0 (if RL learns)│
│  Premature Penalty: -1.0 (if too early) │
│                                         │
│  ─────────────────────────────────────  │
│                                         │
│  Task Success:      +30.0 (if complete) │
│                                         │
└─────────────────────────────────────────┘
```

---

## PPO Update Process

### 1. **Rollout Collection** (Stable-Baselines3)

PPO collects a batch of trajectories:

```python
# For n_steps (e.g., 2048 steps):
for step in range(n_steps):
    # Get action from policy
    action, value, log_prob = policy.predict(obs)
    
    # Execute in environment
    next_obs, reward, done, truncated, info = env.step(action)
    
    # Store in buffer
    rollout_buffer.add(
        obs=obs,
        action=action,
        reward=reward,
        value=value,
        log_prob=log_prob,
        done=done,
    )
    
    obs = next_obs
```

### 2. **Advantage Estimation** (GAE - Generalized Advantage Estimation)

```python
# Compute advantages using GAE
advantages = compute_gae(
    rewards=rollout_buffer.rewards,
    values=rollout_buffer.values,
    dones=rollout_buffer.dones,
    gamma=0.99,          # Discount factor
    gae_lambda=0.95,      # GAE lambda
)

# Compute returns (targets for value function)
returns = advantages + rollout_buffer.values
```

**GAE Formula:**
```
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
A_t = δ_t + (γ * λ) * δ_{t+1} + (γ * λ)² * δ_{t+2} + ...
```

### 3. **Policy Update** (PPO Clipped Objective)

```python
# For n_epochs (e.g., 10 epochs):
for epoch in range(n_epochs):
    # Sample mini-batches
    for batch in mini_batches:
        # Get new policy predictions
        new_log_probs, new_values, entropy = policy.evaluate(
            obs=batch.obs,
            action=batch.action,
        )
        
        # Compute probability ratio
        ratio = exp(new_log_probs - old_log_probs)
        
        # Clipped objective
        clipped_ratio = clip(ratio, 1 - ε, 1 + ε)  # ε = 0.2
        policy_loss = -min(ratio * advantages, clipped_ratio * advantages)
        
        # Value function loss
        value_loss = (new_values - returns)²
        
        # Entropy bonus (encourage exploration)
        entropy_bonus = entropy_coef * entropy
        
        # Total loss
        total_loss = policy_loss + vf_coef * value_loss - entropy_bonus
        
        # Update policy
        optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm(policy.parameters(), max_norm=0.5)
        optimizer.step()
```

### 4. **PPO Update Flow Diagram**

```
┌─────────────────────────────────────────┐
│      Collect Rollout (n_steps=2048)     │
│  obs → policy → action → env → reward   │
└──────────────────────┬──────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────┐
│      Compute Advantages (GAE)           │
│  rewards + values → advantages + returns │
└──────────────────────┬──────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────┐
│      Update Policy (n_epochs=10)        │
│                                         │
│  For each mini-batch:                   │
│    1. Compute new log_probs, values     │
│    2. Compute clipped policy loss       │
│    3. Compute value loss                 │
│    4. Backprop + update                  │
└─────────────────────────────────────────┘
```

---

## Stage Transition Logic

### 1. **Proactive Advance** (Automatic)

```python
def _check_proactive_advance():
    stage_steps = il_wrapper.stage_step_count
    stage_quality = _evaluate_current_stage_quality()
    
    # Condition 1: IL has run most of its duration
    steps_threshold = il_steps_per_stage * 0.85  # 85% of 200 = 170 steps
    if stage_steps >= steps_threshold and stage_quality >= 0.1:
        return True
    
    # Condition 2: Full duration reached
    if stage_steps >= il_steps_per_stage:
        return True
    
    return False
```

**Why Proactive?**
- IL models were trained for ~200 steps per stage
- After 85% duration, IL has likely done its job
- Prevents RL from getting stuck in a stage

### 2. **RL-Initiated Advance**

```python
def _handle_stage_advance():
    # Check minimum steps
    if stage_steps < min_steps_before_advance:  # 10 steps
        return False, -1.0  # Penalty
    
    # Check quality threshold
    stage_quality = _evaluate_current_stage_quality()
    if stage_quality < 0.15:
        return False, -1.0  # Penalty
    
    # Safe transition: open hands first!
    _safe_release_before_transition()
    
    # Advance stage
    success = il_wrapper.advance_stage()
    
    if success:
        return True, 5.0  # Bonus
    return False, 0.0
```

### 3. **Safe Transition** (`_safe_release_before_transition()`)

**CRITICAL:** Prevents dragging cloth during transition!

```python
# 1. Open both hands
robot.set_both_hand_state("open", "open")
for _ in range(30):
    base_env.step()

# 2. Let cloth settle (increased gravity)
garment.particle_material.set_gravity_scale(10.0)
for _ in range(50):
    base_env.step()
garment.particle_material.set_gravity_scale(1.0)

# 3. Move arms away
# (Move to safe position to avoid collision)
```

### 4. **Stage Quality Evaluation**

```python
def _evaluate_current_stage_quality():
    # Get bounding box at stage start
    initial_bbox = _stage_initial_bbox
    current_bbox = _compute_bbox(current_garment_pcd)
    
    # Compute area reduction
    initial_area = (initial_bbox[3] - initial_bbox[0]) * \
                   (initial_bbox[4] - initial_bbox[1])
    current_area = (current_bbox[3] - current_bbox[0]) * \
                   (current_bbox[4] - current_bbox[1])
    
    area_reduction = (initial_area - current_area) / initial_area
    return np.clip(area_reduction, 0.0, 1.0)
```

---

## Complete Step-by-Step Flow

### **Single Step Execution:**

```
1. ENVIRONMENT RECEIVES ACTION
   ├─ Parse: residual_action[60D], stage_advance_signal[1D]
   │
2. GET IL ACTION
   ├─ Get observation for IL (pcd, affordance, etc.)
   ├─ Query current stage's SADP_G model
   └─ Get il_action[60D] (joint-space)
   │
3. APPLY RESIDUALS
   ├─ Extract arm residuals (zero hand joints)
   ├─ final_action = il_action + arm_residuals
   └─ Execute: robot.apply_action(final_action)
   │
4. PHYSICS STEP
   ├─ Isaac Sim updates robot joints
   ├─ Garment deforms
   └─ Step physics 5 times
   │
5. UPDATE STATE
   ├─ Capture new point cloud
   ├─ Get GAM keypoints
   ├─ Update IL observation history
   └─ Track stage quality
   │
6. CHECK STAGE TRANSITIONS
   ├─ Priority 1: Proactive advance?
   │  └─ If yes: advance, reward += 5.0
   ├─ Priority 2: RL advance signal > 0.5?
   │  ├─ If quality/steps OK: advance, reward += 5.0 + 2.0
   │  └─ If not OK: reward -= 1.0 (penalty)
   └─ If advanced: reset stage quality, pre-position robot
   │
7. COMPUTE REWARD
   ├─ Fold progress: +2.0 * progress
   ├─ Compactness: +1.0 * compactness
   ├─ Height penalty: -0.2 * height_var
   ├─ Residual penalty: -0.01 * ||residual||²
   ├─ Stage advance bonus: +5.0 (if advanced)
   └─ Task success: +30.0 (if all done)
   │
8. CHECK TERMINATION
   ├─ terminated = all_stages_done AND final_success
   └─ truncated = current_step >= max_episode_steps
   │
9. GET NEXT OBSERVATION
   ├─ Collect all observation components
   ├─ Apply NaN protection
   └─ Return obs dict
   │
10. RETURN
    └─ (obs, reward, terminated, truncated, info)
```

### **PPO Training Loop:**

```
1. INITIALIZE
   ├─ Create environment
   ├─ Create PPO policy
   └─ Initialize rollout buffer
   │
2. COLLECT ROLLOUT (n_steps=2048)
   ├─ For each step:
   │  ├─ policy.predict(obs) → action, value, log_prob
   │  ├─ env.step(action) → next_obs, reward, done
   │  └─ buffer.add(obs, action, reward, value, log_prob, done)
   │  └─ obs = next_obs
   └─ End when buffer full or episode done
   │
3. COMPUTE ADVANTAGES (GAE)
   ├─ For each step in buffer:
   │  ├─ δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
   │  └─ A_t = δ_t + (γλ)*δ_{t+1} + (γλ)²*δ_{t+2} + ...
   └─ returns = advantages + values
   │
4. UPDATE POLICY (n_epochs=10)
   ├─ For each epoch:
   │  ├─ Shuffle buffer into mini-batches
   │  ├─ For each batch:
   │  │  ├─ Evaluate: new_log_probs, new_values, entropy
   │  │  ├─ ratio = exp(new_log_probs - old_log_probs)
   │  │  ├─ policy_loss = -min(ratio*A, clip(ratio, 1-ε, 1+ε)*A)
   │  │  ├─ value_loss = (new_values - returns)²
   │  │  ├─ total_loss = policy_loss + vf_coef*value_loss - ent_coef*entropy
   │  │  └─ optimizer.step()
   │  └─ Update old_log_probs = new_log_probs
   └─ End after n_epochs
   │
5. REPEAT
   └─ Go to step 2 until total_timesteps reached
```

---

## Key Design Decisions

### 1. **Why Joint-Space Residuals?**

- **SADP_G outputs joint-space actions** (60D)
- Residuals must be in same space to be meaningful
- Allows direct addition: `final = il_action + residual`

### 2. **Why Arm-Only Residuals?**

- **Hands are complex** (24 DOF each, delicate manipulation)
- IL has expert hand control from demonstrations
- RL should focus on **high-level positioning** (where to grab/release)
- Hands = IL, Arms = IL + RL

### 3. **Why Proactive Advance?**

- IL models trained for fixed duration (~200 steps)
- After 85% duration, IL has likely completed its task
- Prevents RL from getting stuck waiting
- RL can still learn to advance earlier if beneficial

### 4. **Why Stage Quality Evaluation?**

- Measures actual progress (area reduction)
- Prevents premature advances (quality < 0.15)
- Rewards proper stage completion
- Helps RL learn optimal timing

### 5. **Why Safe Transitions?**

- **Critical bug fix:** Hands closed during transition drag cloth
- Opens hands, settles garment, moves arms away
- Prevents cloth from being dragged to new manipulation point
- Ensures clean stage transitions

---

## Summary

**The algorithm flow:**

1. **Observation** → RL sees garment state, robot state, IL guidance, stage info
2. **Action** → RL outputs residuals + stage signal
3. **Execution** → IL action + RL residuals → robot moves
4. **Reward** → Progress, compactness, penalties, bonuses
5. **Update** → PPO uses GAE to compute advantages, updates policy

**Key mechanisms:**

- **Residual learning:** RL learns small corrections to IL
- **Stage transitions:** Proactive (automatic) + RL (learned)
- **Quality evaluation:** Measures actual fold progress
- **Safe transitions:** Prevents cloth dragging

**Simplified version removes:**

- Phase-aware control (too complex)
- Early termination (too aggressive)
- Complex gating (unnecessary)

**Keeps:**

- Multi-stage IL integration
- Proactive advance (helps IL)
- Simple residuals (arm-only)
- Core reward shaping

This simplified version should be easier to train and debug while maintaining the core learning objective!

