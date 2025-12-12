# VLA Fine-Tuning and Training Strategy for Garment Folding

## The Action Space Mismatch Problem

### Your Current Action Space:
```python
action = [joint_residuals(60D), stage_advance_signal(1D)]
# 60D = 30 joints per arm (6 arm + 24 hand) × 2 arms
# 1D = stage advance signal (continuous, 0-1)
```

### What VLA Models Are Typically Trained For:

**1. End-Effector Space (Most Common)**
- RT-2, RT-X: 7D (x, y, z, quaternion) per arm = 14D total
- OpenVLA: Often 6-7D end-effector poses
- **Gap**: You need 60D joint space, they output 14D end-effector

**2. Joint Space (Less Common)**
- Some VLA models: Variable joint counts (often 7-14 DOF)
- **Gap**: Different robot configurations, different joint counts

**3. Discrete Actions (Some Models)**
- Some VLA: Discrete action primitives
- **Gap**: You need continuous joint residuals

**4. Task-Specific Actions**
- GROOT: Often task-specific action spaces
- **Gap**: May not match your residual learning setup

---

## Training/Fine-Tuning Strategies

### **Strategy 1: No Fine-Tuning (Feature Extractor Only)** ⭐ RECOMMENDED

**Approach:**
- Use VLA as **feature extractor only**
- VLA outputs visual features, NOT actions
- Your RL policy learns to map VLA features → your action space

**Training Required:**
- ✅ **None!** Use pretrained VLA as-is
- ✅ Only train your RL policy (which you're already doing)

**Architecture:**
```
RGB Image → VLA Vision Encoder → Visual Features (256D)
                                    ↓
Point Cloud + VLA Features → RL Policy → [joint_residuals(60D), stage_advance(1D)]
```

**Pros:**
- ✅ No VLA fine-tuning needed
- ✅ No action space mismatch issues
- ✅ Can use any pretrained VLA model
- ✅ Fastest to implement
- ✅ Leverages VLA's visual understanding

**Cons:**
- ❌ VLA doesn't directly output actions
- ❌ RL needs to learn VLA → action mapping

**Data Requirements:**
- ✅ None (use pretrained VLA)
- ✅ Just your existing RL training data

**When to Use:**
- **Start here!** Easiest, lowest risk
- Good for proof-of-concept
- Can always add fine-tuning later

---

### **Strategy 2: Supervised Fine-Tuning (Action Prediction)**

**Approach:**
- Fine-tune VLA to predict your action space directly
- Train VLA: `(RGB, language) → [joint_residuals(60D), stage_advance(1D)]`

**Training Required:**
- ✅ **Supervised learning** on demonstration data
- ✅ Need: (RGB image, language instruction, action) pairs

**Architecture:**
```
RGB Image + Language → Fine-tuned VLA → [joint_residuals(60D), stage_advance(1D)]
```

**Data Requirements:**

**Option A: Use Your Existing IL Demonstrations**
```python
# Collect from your SADP_G demonstrations
dataset = []
for episode in il_demonstrations:
    for step in episode:
        rgb_image = env_camera.get_rgb()  # (H, W, 3)
        language = f"Stage {current_stage}: Fold garment"
        action = il_action  # From SADP_G (60D joint space)
        dataset.append((rgb_image, language, action))
```

**Option B: Collect New Demonstrations**
- Record expert demonstrations
- Capture RGB + language + actions
- Need: ~1000-10000 demonstrations

**Training Process:**
```python
# Fine-tune VLA action head
for rgb, lang, target_action in dataset:
    # Forward pass
    predicted_action = vla_model(rgb_image=rgb, instruction=lang)
    
    # Loss: MSE between predicted and target
    loss = mse_loss(predicted_action, target_action)
    
    # Backward
    loss.backward()
    optimizer.step()
```

**Pros:**
- ✅ VLA directly outputs your action space
- ✅ Can leverage VLA's visual understanding
- ✅ End-to-end learning

**Cons:**
- ❌ Requires demonstration data
- ❌ Need to modify VLA's action head
- ❌ May need significant fine-tuning
- ❌ Action space mismatch (VLA trained for different actions)

**Challenges:**
1. **Action Head Modification**: Need to replace VLA's action head
   - Original: 7D end-effector
   - New: 61D (60 joint residuals + 1 stage signal)
   
2. **Distribution Shift**: VLA trained on different tasks
   - May need extensive fine-tuning
   - May not transfer well

3. **Data Collection**: Need paired (RGB, language, action) data
   - Can use your IL demonstrations
   - Or collect new expert demonstrations

---

### **Strategy 3: Hierarchical Fine-Tuning (Two-Stage)**

**Approach:**
- VLA outputs high-level plan/guidance
- Separate network maps VLA output → your action space
- Fine-tune both stages

**Architecture:**
```
RGB + Language → VLA → High-level Plan (e.g., "adjust left arm position")
                            ↓
                    Action Mapper → [joint_residuals(60D), stage_advance(1D)]
```

**Training Required:**
- ✅ Fine-tune VLA for high-level planning
- ✅ Train action mapper network

**Stage 1: Fine-tune VLA for Planning**
```python
# VLA outputs high-level commands
vla_output = vla_model(rgb_image, instruction)
# Output: {"left_arm_adjust": 0.3, "right_arm_adjust": -0.1, "stage_advance": 0.7}
```

**Stage 2: Train Action Mapper**
```python
# Map VLA output to joint space
action_mapper = ActionMapper(input_dim=128, output_dim=61)
joint_action = action_mapper(vla_output)
```

**Data Requirements:**
- Option A: Use IL demonstrations (map to high-level commands)
- Option B: Annotate demonstrations with high-level descriptions

**Pros:**
- ✅ Separates visual understanding from action execution
- ✅ More interpretable
- ✅ Can fine-tune stages independently

**Cons:**
- ❌ More complex (two stages to train)
- ❌ Need to define high-level action space
- ❌ Still need demonstration data

---

### **Strategy 4: RL Fine-Tuning (VLA + RL Joint Training)**

**Approach:**
- Start with pretrained VLA
- Fine-tune VLA's action head using RL
- Joint training: VLA + RL policy

**Architecture:**
```
RGB + Language → VLA (frozen vision/language, trainable action head)
                            ↓
                    [joint_residuals(60D), stage_advance(1D)]
                            ↓
                    RL Environment → Reward
                            ↓
                    PPO Update (fine-tune VLA action head)
```

**Training Process:**
```python
# Freeze VLA vision/language encoders
for param in vla_model.vision_encoder.parameters():
    param.requires_grad = False
for param in vla_model.language_encoder.parameters():
    param.requires_grad = False

# Train action head with RL
for rollout in ppo_rollouts:
    # VLA predicts action
    action = vla_model(rgb_image, instruction)
    
    # Execute in environment
    reward = env.step(action)
    
    # PPO update (only action head gradients)
    ppo_loss.backward()  # Only updates action head
```

**Data Requirements:**
- ✅ No demonstration data needed!
- ✅ Use RL environment directly
- ✅ Learn from trial and error

**Pros:**
- ✅ No demonstration data needed
- ✅ Learns directly in your environment
- ✅ Can adapt to your specific task

**Cons:**
- ❌ Slower training (RL is sample-inefficient)
- ❌ May need many episodes
- ❌ VLA action head needs to be trainable
- ❌ May forget pretrained knowledge

**Challenges:**
1. **Sample Efficiency**: RL needs many samples
   - May need 10x more episodes than supervised learning
   
2. **Catastrophic Forgetting**: Fine-tuning may hurt VLA's visual understanding
   - Solution: Use low learning rate, freeze most layers
   
3. **Action Head Design**: Need to design trainable action head
   - Input: VLA features (e.g., 256D)
   - Output: Your action space (61D)

---

### **Strategy 5: Hybrid: VLA Features + RL Policy (Best of Both)**

**Approach:**
- Use VLA as feature extractor (no fine-tuning)
- Train RL policy to use VLA features
- RL learns: `(point_cloud, vla_features, ...) → actions`

**Architecture:**
```
RGB → VLA Vision Encoder (frozen) → Visual Features (256D)
                                            ↓
Point Cloud + VLA Features + ... → RL Policy → [joint_residuals(60D), stage_advance(1D)]
```

**Training Required:**
- ✅ **Only RL policy training** (which you're already doing!)
- ✅ No VLA fine-tuning

**Implementation:**
```python
# In your environment
def _get_observation(self):
    # Existing
    pcd = self._get_garment_pcd()
    gam_keypoints = self._get_gam_keypoints()
    
    # NEW: VLA features (from frozen pretrained model)
    rgb_image = self._env_camera.get_rgb()
    vla_features = self._vla_vision_encoder(rgb_image)  # (256,)
    
    obs = {
        "garment_pcd": pcd,
        "gam_keypoints": gam_keypoints,
        "vla_visual_features": vla_features,  # NEW
        # ... other observations
    }
    return obs

# RL policy learns to use VLA features
# No changes to training loop!
```

**Pros:**
- ✅ **No VLA fine-tuning needed**
- ✅ **No action space mismatch**
- ✅ **Leverages VLA's visual understanding**
- ✅ **Easiest to implement**
- ✅ **Can use any pretrained VLA**

**Cons:**
- ❌ RL needs to learn VLA → action mapping
- ❌ May need more RL training to learn to use VLA features

**Data Requirements:**
- ✅ None (use pretrained VLA)
- ✅ Just your existing RL training

**When to Use:**
- **Recommended starting point!**
- Best risk/reward ratio
- Can always add fine-tuning later if needed

---

## Detailed Training Requirements by Strategy

### **Strategy 1: Feature Extractor (No Fine-Tuning)**

**Training Required:**
- ✅ **None for VLA** (use pretrained)
- ✅ **RL policy training** (already doing this)

**Data Required:**
- ✅ None (pretrained VLA)
- ✅ Your existing RL training data

**Time Estimate:**
- VLA setup: 1-2 days
- RL training: Same as current (no change)

**Compute Requirements:**
- VLA inference: ~50-100ms per step (GPU)
- RL training: Same as current

---

### **Strategy 2: Supervised Fine-Tuning**

**Training Required:**
- ✅ **VLA fine-tuning**: 1-5 days (depending on data)
- ✅ **RL policy training**: Same as current

**Data Required:**
- ✅ **Demonstration dataset**: 1000-10000 (RGB, language, action) pairs
- ✅ Can use your IL demonstrations (SADP_G outputs)

**Data Collection:**
```python
# Option A: Use existing IL demonstrations
dataset = []
for episode in range(1000):
    env.reset()
    for step in range(400):
        rgb = env.get_rgb()
        lang = f"Stage {current_stage}: Fold garment"
        action = il_policy.predict(obs)  # 60D joint action
        dataset.append((rgb, lang, action))

# Option B: Collect expert demonstrations
# Record human expert performing task
# Capture: RGB + language + actions
```

**Training Process:**
```python
# 1. Load pretrained VLA
vla_model = load_pretrained_vla()

# 2. Replace action head
vla_model.action_head = nn.Linear(256, 61)  # 61D output

# 3. Fine-tune
for epoch in range(10):
    for rgb, lang, target_action in dataloader:
        pred_action = vla_model(rgb_image=rgb, instruction=lang)
        loss = mse_loss(pred_action, target_action)
        loss.backward()
        optimizer.step()
```

**Time Estimate:**
- Data collection: 1-3 days (if using IL) or 1-2 weeks (if collecting new)
- Fine-tuning: 1-5 days (depending on dataset size)
- RL training: Same as current

**Compute Requirements:**
- Fine-tuning: 1-2 GPUs, 1-5 days
- Inference: ~50-100ms per step

---

### **Strategy 3: Hierarchical Fine-Tuning**

**Training Required:**
- ✅ **VLA fine-tuning**: 1-3 days (for high-level planning)
- ✅ **Action mapper training**: 1-2 days
- ✅ **RL policy training**: Same as current

**Data Required:**
- ✅ **High-level annotations**: 1000-5000 examples
- ✅ Map demonstrations to high-level commands

**High-Level Action Space Definition:**
```python
# Define high-level actions
high_level_actions = {
    "left_arm_x": float,      # Adjust left arm X position
    "left_arm_y": float,      # Adjust left arm Y position
    "left_arm_z": float,      # Adjust left arm Z position
    "right_arm_x": float,
    "right_arm_y": float,
    "right_arm_z": float,
    "stage_advance": float,   # Stage advance signal
}
```

**Training Process:**
```python
# Stage 1: Fine-tune VLA for high-level planning
for rgb, lang, high_level_action in high_level_dataset:
    pred = vla_model(rgb_image=rgb, instruction=lang)
    loss = mse_loss(pred, high_level_action)
    # ...

# Stage 2: Train action mapper
action_mapper = ActionMapper(input_dim=7, output_dim=61)
for high_level, joint_action in mapper_dataset:
    pred = action_mapper(high_level)
    loss = mse_loss(pred, joint_action)
    # ...
```

**Time Estimate:**
- Data annotation: 2-5 days
- VLA fine-tuning: 1-3 days
- Action mapper training: 1-2 days
- RL training: Same as current

---

### **Strategy 4: RL Fine-Tuning**

**Training Required:**
- ✅ **VLA action head training**: Via RL (many episodes)
- ✅ **RL policy training**: Integrated with VLA training

**Data Required:**
- ✅ **No demonstration data!**
- ✅ Use RL environment directly

**Training Process:**
```python
# Freeze VLA encoders, train action head
vla_model.vision_encoder.requires_grad = False
vla_model.language_encoder.requires_grad = False
vla_model.action_head.requires_grad = True

# RL training loop
for episode in range(10000):  # May need many episodes
    obs = env.reset()
    for step in range(400):
        # VLA predicts action
        action = vla_model(rgb_image=obs['rgb'], instruction=stage_description)
        
        # Execute
        next_obs, reward, done, _ = env.step(action)
        
        # Store in buffer
        buffer.add(obs, action, reward, ...)
    
    # PPO update (only updates action head)
    if len(buffer) >= batch_size:
        advantages = compute_gae(buffer)
        for epoch in range(ppo_epochs):
            loss = ppo_loss(buffer, advantages)
            loss.backward()  # Only action head gradients
            optimizer.step()
```

**Time Estimate:**
- RL training: **10-50x longer** than supervised (sample inefficient)
- May need: 10,000-50,000 episodes (vs 1,000-5,000 for supervised)

**Compute Requirements:**
- Training: 1-2 GPUs, **weeks to months**
- Much slower than supervised fine-tuning

---

### **Strategy 5: Hybrid (VLA Features + RL)**

**Training Required:**
- ✅ **None for VLA** (use pretrained)
- ✅ **RL policy training** (already doing this, just with more features)

**Data Required:**
- ✅ None (pretrained VLA)
- ✅ Your existing RL training data

**Time Estimate:**
- VLA setup: 1-2 days
- RL training: **Same as current** (no change, just more features)

**Compute Requirements:**
- VLA inference: ~50-100ms per step
- RL training: Same as current

---

## Recommendation: Start with Strategy 5 (Hybrid)

### **Why Strategy 5 is Best:**

1. **No Fine-Tuning Needed**
   - Use pretrained VLA as-is
   - No action space mismatch
   - Fastest to implement

2. **Leverages VLA's Strengths**
   - Visual understanding (wrinkles, texture, alignment)
   - Semantic reasoning
   - No need to retrain

3. **Minimal Risk**
   - Doesn't break existing system
   - Can add fine-tuning later if needed
   - Easy to A/B test (with/without VLA)

4. **Your RL Already Handles Complex Observations**
   - You have MultiInputPolicy
   - Just add VLA features as another input
   - RL learns to use them

### **Implementation Plan:**

**Phase 1: Add VLA Features (1-2 days)**
```python
# 1. Load pretrained VLA vision encoder
vla_vision = load_pretrained_vla_vision_encoder()

# 2. Add to observation space
obs["vla_visual_features"] = vla_vision(rgb_image)  # (256,)

# 3. Update policy network to accept VLA features
# (Add VLA encoder to your feature extractor)

# 4. Train RL policy (same as before, just more features)
```

**Phase 2: Evaluate (1 week)**
- Compare: With VLA vs Without VLA
- Measure: Success rate, fold quality, learning speed

**Phase 3: Fine-Tune if Needed (Optional)**
- If VLA features help but not enough
- Consider Strategy 2 (supervised fine-tuning)
- Use your IL demonstrations as training data

---

## Action Space Conversion (If Fine-Tuning)

If you do fine-tune VLA to output actions, you'll need to handle the action space:

### **Option A: Direct Joint Space Output**
```python
# VLA outputs 61D directly
vla_model.action_head = nn.Linear(256, 61)  # 60D residuals + 1D stage
action = vla_model(rgb_image, instruction)
```

### **Option B: End-Effector → Joint Space Conversion**
```python
# VLA outputs end-effector poses (14D)
vla_output = vla_model(rgb_image, instruction)  # 14D

# Convert to joint space using IK
joint_action = inverse_kinematics(vla_output)  # 60D
```

### **Option C: Residual on End-Effector**
```python
# VLA outputs end-effector residual (14D)
vla_residual = vla_model(rgb_image, instruction)  # 14D

# Apply to IL's end-effector target
il_ee_target = il_policy.get_ee_target(obs)
final_ee_target = il_ee_target + vla_residual

# Convert to joint space
joint_action = inverse_kinematics(final_ee_target)  # 60D
```

---

## Summary

**For Your Use Case, I Recommend:**

1. **Start: Strategy 5 (Hybrid - Feature Extractor)**
   - No fine-tuning needed
   - Fastest to implement
   - Lowest risk
   - Can always add fine-tuning later

2. **If Needed: Strategy 2 (Supervised Fine-Tuning)**
   - Use your IL demonstrations as training data
   - Fine-tune VLA's action head
   - More powerful but requires data collection

3. **Avoid (Initially): Strategy 4 (RL Fine-Tuning)**
   - Too sample-inefficient
   - Takes too long
   - Only use if other strategies don't work

**Key Insight:**
You don't need VLA to output actions directly! Using VLA as a feature extractor lets you leverage its visual understanding while keeping your existing RL architecture. Your RL policy can learn to map VLA features to your action space, which is exactly what RL is good at!

