import robosuite as suite
import numpy as np

env = suite.make(
    env_name="TwoArmTransport",
    robots=["Panda", "Panda"],
    env_configuration="single-arm-opposed",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,
    control_freq=20,
)

env.reset()

# Print all body names
print("Bodies:")
for i in range(env.sim.model.nbody):
    print(i, env.sim.model.body_id2name(i))

# Print all geom names
print("\nGeoms:")
for i in range(env.sim.model.ngeom):
    print(i, env.sim.model.geom_id2name(i))

while True:
    action = np.zeros(env.action_dim)
    obs, reward, done, info = env.step(action)
    env.render()
