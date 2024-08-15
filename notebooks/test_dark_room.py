# %%
from toymeta.dark_room import DarkRoom
import matplotlib.pyplot as plt

env = DarkRoom(terminate_on_goal=True)
# %%
obs, info = env.reset()
done = False
while not done:
    obs, reward, done, _, info = env.step(env.action_space.sample())
    print(obs, reward, info)
    plt.imshow(env.render())
    plt.show()
# %%
env.observation_space
# %%
env.action_space
# %%
