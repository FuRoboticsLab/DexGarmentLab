# Garment-Related Parameters Summary Doc

## Particle Material

Reference Doc Link : [Particle Material Doc](https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html#particle-material)

Related Parameters:

- **adhesion(粘附力)**

    Range : [0, inf)

    值越大，表示粒子和物体(刚体或可变形物体)之间的粘附力越强，容易产生粘连效果

- **adhesion_offset_scale(粘附力偏移量)**

    Range : [0, inf)

    值越大，表示粘附力的作用范围越大

- **cohesion(内聚力，更多作用在流体上)**

    Range : [0, inf)

    凝聚力可以理解为粒子之间的吸引力，使得它们倾向于保持整体的形状和结构，在模拟液体流动会用到（值越大表示吸引力越强，过大容易导致流体产生类似于果冻的质感）

- **particle_adhesion_scale**

    Range : [0, inf)

    表征固体粒子间的粘附力

- **particle_friction_scale**

    Range : [0, inf)

    表征固体粒子之间的摩擦力

- **drag**

    Range : [0, inf)

    控制物体受到的空气或流体阻力（值越大，物体运动速度衰减越快）

- **lift**

    Range : [0, inf)

    控制物体在流体或空气中产生的升力（值越大，升力越大）

- **friction**

    Range : [0, inf)

    控制粒子与刚体（或可变形物体）的摩擦（值越大，摩擦越大）

- **damping**

    Range : [0, inf)

    用于控制粒子的速度衰减

    较高的damping->快速稳定，减少震动，适用于高摩擦或粘稠的物体

    较低的damping->延长运动时间，保持动态效果，适用于轻盈物体或流体

- **gravity_scale**

    Range: (-inf , inf)

    重力加速度比例因子。它可以用来模拟比空气轻的充气玩具(例如气球)。设置为-1.0表示翻转重力。

- **viscosity(粘性)**

    Range : [0, inf)

    控制流体内部的摩擦力（表征为流体的高粘度以及低粘度）

- **vorticity_confinement(涡旋限制)**

    Range : [0, inf)

    模拟流体涡旋

    较高值->快速旋转的液体，比如风暴中的气流或者水流

    较低值->适用于需平滑流动的流体模拟

- **surface_tension(表面张力)**

    Range: [0, inf)

    较大值->流体颗粒更紧密地结合在一起，形成更加平静的表面，表面破裂或波纹现象较少

    较小值->表面易出现不稳定现象，波纹严重，容易发生表面破裂


## Particle System

Reference Doc Link : [Particle System Doc](https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html#particle-system)

Related Parameters:

- **particle_system_enabled(bool)**

    Whether to enable or disable the particle system.

- **simulation_owner(str)**

    Single PhysicsScene that simulates this particle system.
    
- **enable_ccd(bool)**

    Enable continuous collision detection for particles to help avoid tunneling effects.

- **solver_position_iteration_count(int)**

    Number of solver iterations for position.

- **max_depenetration_velocity(float)**

    The maximum velocity permitted to be introduced by the solver to depenetrate intersecting particles.

    该参数用于控制粒子之间在碰撞时的渗透速度，当两个物体发生碰撞并且出现相互渗透（即物体之间穿透了对方的边界），max_depenetration_velocity 将限制这种渗透的速度，当物体之间的相对速度超过该值时，系统将采取措施将它们分开，确保它们不会继续渗透。

- **global_self_collision_enabled(bool)**

    If True, self collisions follow particle-object-specific settings.

- **non_particle_collision_enabled(bool)**

    Enable or disable particle collision with non-particle objects for all particles in the system.

- **contact_offset(float)** 【可控制粒子的大小】

    Contact offset used for collisions with non-particle objects such as rigid or deformable bodies.

- **rest_offset(float)** 【可控制粒子的大小】

    Rest offset used for collisions with non-particle objects such as rigid or deformable bodies.

- **particle_contact_offset(float)** 【可控制粒子的大小】

    Contact offset used for interactions between particles. Must be larger than solid and fluid rest offsets.

- **solid_rest_offset(float)**

    Rest offset used for solid-solid or solid-fluid particle interactions. Must be smaller than particle contact offset.

- **fluid_rest_offset(float)**

    Rest offset used for fluid-fluid particle interactions. Must be smaller than particle contact offset.

- **wind(float)**

    The wind applied to the current particle system.

- **max_neighborhood(int)**

    The particle neighborhood size.

- **max_velocity(int)**

    Maximum particle velocity.

## Cloth Prim

Reference Doc Link : [Cloth Prim Doc](https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html#cloth-prim)

Related Parameters:

- **particle_mass(float)**

    range: [0, inf)

    the mass of one single particle.

- **pressure(float)**

    range: [0, inf)

    if > 0, a particle cloth has an additional pressure constraint that provides inflatable (i.e. balloon-like) dynamics. Pressure only works well for closed or approximately closed meshes, 

    pressure 参数模拟了布料内部的气体或液体压力，允许布料在碰撞或受力时产生膨胀效应。在模拟如旗帜或帐篷等受风力影响的布料时，可以通过调整 pressure 来模拟风的作用。在水下或气泡环境中，pressure 参数可以帮助实现布料的浮力效果，使其看起来更自然。

- **self_collision(bool)**

    enable self collision of the particles or of the particle object.

- **particle_group(int)**

    range: [0, 2^20)

    group Id of the particles

    通过将粒子分组，用户可以为每个组设置不同的物理特性，例如弹性、摩擦、密度等。这使得布料的不同部分可以表现出不同的物理特性。主要用于模拟复杂布料。

- **self_collision_filter(bool)**

    当设置为 True 时，布料的粒子之间的碰撞将会根据它们的静止位置（rest position）进行过滤。这意味着只有当粒子之间的距离小于某个阈值时，才会被视为有效碰撞，从而进行物理处理。这可以避免不必要的碰撞检测，提高计算效率。
    
    当设置为 False 时，布料粒子之间的自碰撞检测将不会考虑静止位置的距离限制，所有接触的粒子都会进行碰撞处理。

- **stretch_stiffness(float)**

    range: [0, inf)

    控制布料在拉伸时的抵抗力。值越大，布料在受到拉伸力时变形的程度越小，布料会更坚硬和不易拉伸。

    ps:在模拟如紧身衣物或运动装备时，增加拉伸刚度可以使布料在运动中保持更好的形状和支撑。

- **bend_stiffness(float)**

    range: [0, inf)

    控制布料在弯曲时的抵抗力。较高的弯曲刚度意味着布料在弯曲时不容易变形，保持较好的结构。
    
    ps:在模拟如帽子、裙子等需要保持形状的布料时，可以增加弯曲刚度以防止过度下垂或扭曲。

- **shear_stiffness(float)**

    range: [0, inf)

    控制布料在剪切（即横向变形）时的抵抗力。值越大，布料在受到横向力时变形的程度越小。

- **spring_damping(float)**

    range: [0, inf)

    控制布料粒子间弹簧的阻尼效果。较高的阻尼值可以减缓布料在受到力后振荡的速度，从而使其运动更加平滑和稳定。

## Deformable Material

Reference Doc Link : [Deformable Material Doc](https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html#deformable-material)

Related Parameters:

- **damping_scale(阻尼大小)**

    Range: [0, inf)

    与弹性形变无关，影响的是物体移动过程中受到的速度衰减（阻尼越大，速度衰减越快）

- **dynamic_friction(动态摩擦力)**

    Range: [0, inf)

    动摩擦在粒子和刚体或变形体之间的所有相互作用中都起作用

- **elasticity_damping(弹性恢复的阻尼效应)**

    Range: [0, inf)

    较高值->变形后恢复原状的速度慢很多（慢性恢复）

    较低值->模拟快速恢复的弹性材料（快速恢复）

- **poissons_ratio(泊松比)**

    Range: (0 , 0.5)

    描述在某个方向上发生形变时，其他方向上形变的程度

    0表示在某个方向上发生形变时，其他方向上完全不发生形变

    0.5表示在某个方向上发生形变时，其他方向上完全补偿形变

- **youngs_modululs(杨氏弹性模量)**

    Range: [0, inf)

    衡量受力时抵抗形变的能力

    较高值表示不易变形，较低值表示易变性

## Deformable Prim

- **vertex_velocity_damping (float)**

    速度衰减阻尼

- **sleep_damping (float)**

    在物理模拟中，为了提高计算效率，物体如果运动速度足够慢，可能会被标记为“休眠”（Sleeping）。处于休眠状态的物体被认为不再显著运动，物理引擎会暂时停止计算这些物体的物理更新。

    sleep_threshold是指当物体的运动速度接近 sleep_threshold（休眠阈值）时，sleep_damping 控制对物体运动的额外阻尼。这个阻尼值使慢速运动的物体更容易减速并达到休眠状态。

- **sleep_threshold (float)**

    它定义了物体每秒最大允许的线性运动幅度（速度），以触发休眠。在每一帧中，如果物体的速度小于 sleep_threshold，并且持续一段时间，物体会被认为接近静止，从而进入休眠状态。

- **settling_threshold (float)**

    如果 FEM 软体物体的线性运动幅度（每秒移动的最大距离）低于 settling_threshold，该物体会被标记为“即将静止”。
    之后，如果物体持续满足休眠条件一段时间（受其他参数如 sleep_damping 和 sleep_threshold 的控制），它会最终进入休眠状态。

- **self_collision (bool)**

    基于rest-position距离的自碰撞（开关）

- **self_collision_filter_distance (float)**

    当启用自碰撞检测时（即一个物体的不同部分可以彼此碰撞），物理引擎会检查该物体内的顶点或其他部位是否相互“侵入”或接触。
    然而，为避免过于频繁或不必要的碰撞检测，可以设定一个“过滤距离”阈值。

    self_collision_filter_distance定义自碰撞检测的“穿透距离”阈值。只有当两个部分的相对位置距离小于此值（即发生了一定程度的穿透）时，才会生成自碰撞的接触点。

- **solver_position_iteration_count (int)**

    在物理模拟中，解算器用于求解约束（例如碰撞、关节限制等）并更新物体的位置和速度。
    值较大：解算的精度更高，物体的穿透和约束误差会更小。适合需要高精度物理模拟的场景，例如涉及软体或复杂关节的物体。

- **simulation_hexahedral_resolution (int)**

    simulation_hexahedral_resolution 控制软体物体模拟网格的六面体网格分辨率。较高的分辨率能提供更精确的物体形变模拟

- **kinematic_enabled (bool)**
    物理引擎中，物体通常分为两种：

    动力学物体（Dynamic Body）：受物理引擎控制，遵循力学定律（如重力、碰撞、摩擦等）。

    运动学物体（Kinematic Body）：由用户明确指定位置和速度，不受物理引擎力学控制，但可以影响其他动力学物体。

    当设置为 True 时：

        物体切换为运动学模式，其位置和速度由用户直接控制，而非由物理引擎计算。

        物体不会受力或碰撞影响，但仍可参与碰撞检测，并推动或阻挡动力学物体。

    当设置为 False 时：
        
        物体切换为动力学模式，完全受物理引擎控制。


- **simulation_rest_points (Sequence[float])**

    simulation_rest_points 定义了物体静止状态下的四面体网格顶点。用户可以通过提供高质量的自定义点列表来精确控制柔性物体的物理行为。如果未提供，则系统会基于simulation_hexahedral_resolution自动生成网格。它在复杂弹性物体和软体模拟中至关重要。

- **simulation_indices (Sequence[int])**

    simulation_rest_points 提供了网格的顶点列表，而 simulation_indices 则提供了这些顶点的连接关系（即哪些顶点组成一个四面体单元）。
    这两个参数共同定义了网格的结构。

- **collision_rest_points (Sequence[float])**
    
    collision_rest_points 是一个浮动值的序列，表示碰撞网格的顶点坐标。在物体静止时，这些顶点定义了物体的碰撞几何形状。

    当提供了 simulation_rest_points（模拟网格顶点）时，必须同时提供 collision_rest_points（碰撞网格顶点），以确保碰撞网格与模拟网格一致。

- **collision_indices (Sequence[int])**

    collision_indices 是与 collision_rest_points 配合使用的。当 collision_rest_points 指定了碰撞网格的顶点位置时，必须同时提供 collision_indices，用来描述这些顶点如何构成四面体网格。

- **collision_simplification (bool)**

    当 collision_simplification 设置为 True 时，在将网格用于软体物体创建之前，物理引擎会对碰撞网格进行简化。

    该简化过程会减少碰撞网格的顶点和面数，从而提高碰撞检测的计算效率。

    如果已经提供了模拟网格（simulation_rest_points），则该简化选项会被忽略，因为模拟网格已经定义了物体的形状和拓扑，碰撞网格将会直接根据模拟网格生成。

- **collision_simplification_remeshing (bool)**

    用于指示网格简化是否基于重新网格化（remeshing）。

    这意味着在简化过程中，物体的网格不仅仅是减少顶点数量，而是通过重新排列和优化顶点的分布，确保简化后的网格仍然具备良好的拓扑结构。

- **collision_simplification_remeshing_resolution (int)**

    当启用重新网格化时，该参数决定了重新网格化的精度。它会影响最终碰撞网格的密度。

    如果该值设置为 0，物理引擎会根据物体的形状和其他因素动态地选择一个适当的分辨率，通常采用试探法。


- **collision_simplification_target_triangle_count (int)**
    当启用网格简化时，该参数确定了简化过程中网格三角形的最终目标数量。物理引擎会根据这个目标值调整网格结构。

    如果该值为 0，物理引擎将根据 simulation_hexahedral_resolution 的值来启发式地确定目标三角形数量，通常根据网格的分辨率来动态选择一个合理的简化目标。


- **collision_simplification_force_conforming (bool)**

    当该参数为 True 时，生成碰撞网格时的四面体化过程将强制与输入的三角形网格保持一致。这意味着生成的四面体将尽可能紧密地符合原始三角形网格的形状，确保碰撞网格与原网格的几何形状相符。

    当该参数为 False 时，物理引擎将选择适合的四面体化方法，但不强制四面体网格与输入三角形网格的精确匹配。这样可能会产生一些不完全符合原网格的四面体形状，从而可能牺牲一些精度，但可能有助于简化计算过程。


- **embedding (Sequence[int])**

    每个碰撞点（可能是物体表面上的一个点）都有一个对应的索引，这个索引指向它所在的四面体网格的编号。通过这种方式，系统能够快速查找和处理这些碰撞点与相应四面体之间的关系。
