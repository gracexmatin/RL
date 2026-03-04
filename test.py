from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
from environment import BatchJobEnv


def _patch_numpy_bitgenerator_unpickle():
    """
    Compat for older pickle payloads that store BitGenerator as a class
    instead of a string name (seen across numpy version changes).
    """
    try:
        import numpy.random._pickle as np_pickle
    except Exception:
        return

    ctor_name = "__bit_generator_ctor"
    if not hasattr(np_pickle, ctor_name):
        return

    original_ctor = getattr(np_pickle, ctor_name)

    def _compat_ctor(bit_generator_name="MT19937"):
        if isinstance(bit_generator_name, type):
            bit_generator_name = bit_generator_name.__name__
        return original_ctor(bit_generator_name)

    setattr(np_pickle, ctor_name, _compat_ctor)


_patch_numpy_bitgenerator_unpickle()

# 1. 加载环境
batch_task_path = "RL/batch.csv"
resource_path = "RL/resources.csv"
env = DummyVecEnv([lambda: BatchJobEnv(
    task_csv=batch_task_path,
    resource_csv=resource_path,
)])

# 2. 加载归一化统计量 (关键步骤)
# 这会让测试环境使用训练时统计出的 Mean 和 Var 来缩放输入
env = VecNormalize.load("models/PPO/test3.3/vec_normalize.pkl", env)

# 测试时不要更新归一化参数 (training=False)，也不要归一化 Reward (norm_reward=False)
# 因为我们要看真实的 Reward 是多少
env.training = False 
env.norm_reward = False

# 3. 加载模型
model = PPO.load("models/PPO/test3.3/batch_job_ppo.zip", env=env)

# 4. 运行可视化
obs = env.reset()
total_reward = 0

print(f"{'Step':<5} | {'Action (GPU Alloc)':<30} | {'Price':<6} | {'Reward':<8}")
print("-" * 60)

for i in range(288): # 模拟一天
    # deterministic=True 代表使用最优策略，不进行随机探索
    action, _states = model.predict(obs, deterministic=True)

    obs, rewards, dones, infos = env.step(action)
    total_reward += rewards[0]

    # 获取原始环境中的真实数据（VecEnv 会把 info 包装在 list 里）
    # 注意：你需要去 Env 代码里把 obs 的最后一维(电价)取出来方便打印
    # 或者我们直接看 action 的情况

    # 简单的格式化打印，只打印前3个任务的动作
    act_str = str(np.round(action[0][:3], 2)) 

    # 这里的 obs 是归一化过的，看不出真实电价，所以这里只打印动作和奖励
    print(f"{i:<5} | {act_str:<30} | {'?':<6} | {rewards[0]:.2f}")

    if dones[0]:
        print("Episode Finished")
        obs = env.reset()
        break

print(f"Total Daily Reward: {total_reward:.2f}")