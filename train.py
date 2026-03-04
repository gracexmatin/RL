from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import os
import random
import json
from datetime import datetime
import numpy as np
import torch
import wandb
from wandb.integration.sb3 import WandbCallback

from environment import BatchJobEnv

# 保留明文密钥（按当前需求）
os.environ["WANDB_API_KEY"] = "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"

# ==========================================
# 目录配置（常改项）
# ==========================================
MODELS_DIR = "models/PPO/test3.3"
LOG_DIR = "logs"

# ==========================================
# 0. 统一训练配置（W&B配置与实际训练完全一致）
# ==========================================
TRAIN_CONFIG = {
    "algo": "PPO",
    "seed": 122,
    "total_episodes": 500,
    # SB3 仍要求传 total_timesteps，这里是每回合步数下限。
    # 若环境自然回合长度更大，会自动提升到自然长度。
    "max_steps_per_episode": 400,
    "n_steps": 576,
    "batch_size": 144,
    "learning_rate": 1e-3,
    "ent_coef": 0.001,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "n_epochs": 5,
    "target_kl": 0.03,
    "print_every_steps": 200,
    "weight_log_every_steps": 500,
    "weight_hist_every_steps": 5000,
    # ===== 检查点与恢复接口 =====
    # 每隔多少个 episode 自动保存一次（用于暂停/中断后续训）
    "checkpoint_every_episodes": 50,
    # 检查点文件名前缀
    "checkpoint_prefix": "batch_job_ppo",
    # 从已有检查点恢复训练（不恢复则设为 None）
    # 示例:
    # "models/PPO/batch_job_ppo_ep00050_ts00012345.zip"
    "resume_model_path": "models/PPO/test3.3/batch_job_ppo_ep00500_ts0006320500.zip",
    # 对应的 VecNormalize 统计量路径（不恢复则设为 None）
    # 示例:
    # "models/PPO/batch_job_ppo_ep00050_ts00012345_vecnormalize.pkl"
    "resume_vecnormalize_path": "models/PPO/test3.3/batch_job_ppo_ep00500_ts0006320500_vecnormalize.pkl",
    "env_kwargs": {
        "constraint_penalty_coef": 10.0,
        "finish_bonus_base": 5.0,
        "finish_bonus_workload_coef": 0.15,
        "finish_bonus_slack_coef": 0.1,
    },
    "policy_net_arch": [128, 128],
}


class TensorboardCallback(BaseCallback):
    """记录奖励相关指标，供 TensorBoard/W&B 展示。"""

    def __init__(
        self,
        print_every_steps=200,
        max_episodes=None,
        weight_log_every_steps=500,
        weight_hist_every_steps=5000,
        checkpoint_every_episodes=50,
        checkpoint_prefix="batch_job_ppo",
        models_dir=MODELS_DIR,
        verbose=0,
    ):
        super().__init__(verbose)
        self.print_every_steps = int(print_every_steps)
        self.max_episodes = int(max_episodes) if max_episodes is not None else None
        self.weight_log_every_steps = int(weight_log_every_steps)
        self.weight_hist_every_steps = int(weight_hist_every_steps)
        self.checkpoint_every_episodes = int(checkpoint_every_episodes)
        self.checkpoint_prefix = str(checkpoint_prefix)
        self.models_dir = str(models_dir)
        self._step_count = 0
        self._episode_count = 0
        self._episode_returns_window = []
        self._episode_lengths_window = []
        self._param_name_map = {}

    def _save_episode_checkpoint(self):
        """
        保存中断恢复检查点。
        命名规则:
        {checkpoint_prefix}_ep{episode:05d}_ts{timesteps:010d}.zip
        {checkpoint_prefix}_ep{episode:05d}_ts{timesteps:010d}_vecnormalize.pkl
        """
        episode_tag = f"ep{self._episode_count:05d}"
        timestep_tag = f"ts{int(self.num_timesteps):010d}"
        stem = f"{self.checkpoint_prefix}_{episode_tag}_{timestep_tag}"

        model_ckpt_path = os.path.join(self.models_dir, f"{stem}.zip")
        vecnorm_ckpt_path = os.path.join(self.models_dir, f"{stem}_vecnormalize.pkl")

        self.model.save(model_ckpt_path)
        vec_env = self.model.get_vec_normalize_env()
        if vec_env is not None:
            vec_env.save(vecnorm_ckpt_path)

        print(f"[Checkpoint] model={model_ckpt_path}")
        if vec_env is not None:
            print(f"[Checkpoint] vecnorm={vecnorm_ckpt_path}")

    def _log_weight_stats(self):
        policy = self.model.policy
        for idx, (name, param) in enumerate(policy.named_parameters()):
            data = param.detach()
            short_tag = f"p{idx:02d}"
            self._param_name_map[short_tag] = name
            self.logger.record(f"weights/{short_tag}_mean", float(data.mean().item()))
            self.logger.record(f"weights/{short_tag}_std", float(data.std().item()))
            self.logger.record(f"weights/{short_tag}_absmax", float(data.abs().max().item()))

            if param.grad is not None:
                grad = param.grad.detach()
                self.logger.record(f"grads/{short_tag}_mean", float(grad.mean().item()))
                self.logger.record(f"grads/{short_tag}_std", float(grad.std().item()))
                self.logger.record(f"grads/{short_tag}_absmax", float(grad.abs().max().item()))

    def _log_weight_histogram(self):
        if wandb.run is None:
            return
        policy = self.model.policy
        payload = {}
        for idx, (name, param) in enumerate(policy.named_parameters()):
            short_tag = f"p{idx:02d}"
            self._param_name_map[short_tag] = name
            payload[f"weights_hist/{short_tag}"] = wandb.Histogram(
                param.detach().cpu().numpy()
            )
        payload["weights_hist/param_name_map"] = dict(self._param_name_map)
        payload["time/num_timesteps"] = self.num_timesteps
        wandb.log(payload)

    def _on_step(self) -> bool:
        self._step_count += 1
        # 当前训练步的即时奖励（VecEnv 场景下取均值）
        rewards = self.locals.get("rewards", None)
        if rewards is not None:
            self.logger.record("reward/step_reward_mean", float(np.mean(rewards)))

        # Monitor 在 episode 结束时会在 info 中写入 episode 字段
        infos = self.locals.get("infos", [])
        reward_keys = [
            "reward_work_done",
            "reward_finish_bonus",
            "penalty_energy_cost",
            "penalty_deadline",
            "penalty_constraint",
            "reward_total",
        ]
        reward_values = {k: [] for k in reward_keys}
        for info in infos:
            for key in reward_keys:
                if key in info:
                    reward_values[key].append(float(info[key]))

            episode_info = info.get("episode")
            if episode_info is None:
                continue
            self._episode_count += 1
            ep_return = float(episode_info["r"])
            ep_length = float(episode_info["l"])
            self.logger.record("reward/episode_return", float(episode_info["r"]))
            self.logger.record("reward/episode_length", float(episode_info["l"]))

            self._episode_returns_window.append(ep_return)
            self._episode_lengths_window.append(ep_length)
            if len(self._episode_returns_window) > 20:
                self._episode_returns_window.pop(0)
            if len(self._episode_lengths_window) > 20:
                self._episode_lengths_window.pop(0)

            # 额外写入 W&B：用 episode/index 作为横轴
            if wandb.run is not None:
                wandb.log(
                    {
                        "episode/index": self._episode_count,
                        "episode/return": ep_return,
                        "episode/length": ep_length,
                        "episode/return_ma20": float(np.mean(self._episode_returns_window)),
                        "episode/length_ma20": float(np.mean(self._episode_lengths_window)),
                    }
                )

            if (
                self.checkpoint_every_episodes > 0
                and self._episode_count % self.checkpoint_every_episodes == 0
            ):
                self._save_episode_checkpoint()

            if self.max_episodes is not None and self._episode_count >= self.max_episodes:
                print(
                    f"Reached target episodes: {self._episode_count}/{self.max_episodes}. Stop training."
                )
                return False

        for key, values in reward_values.items():
            if values:
                self.logger.record(f"reward_components/{key}", float(np.mean(values)))

        if reward_values["reward_total"] and self._step_count % self.print_every_steps == 0:
            print(
                " | ".join(
                    [
                        f"R={np.mean(reward_values['reward_total']):.4f}",
                        f"work={np.mean(reward_values['reward_work_done']):.4f}",
                        f"finish={np.mean(reward_values['reward_finish_bonus']):.4f}",
                        f"energy=-{np.mean(reward_values['penalty_energy_cost']):.4f}",
                        f"ddl=-{np.mean(reward_values['penalty_deadline']):.4f}",
                        f"cap=-{np.mean(reward_values['penalty_constraint']):.4f}",
                    ]
                )
            )

        if self._step_count % self.weight_log_every_steps == 0:
            self._log_weight_stats()
        if self._step_count % self.weight_hist_every_steps == 0:
            self._log_weight_histogram()
        return True


def make_env():
    env = BatchJobEnv(
        task_csv="RL/batch.csv",
        resource_csv="RL/resources.csv",
        **TRAIN_CONFIG["env_kwargs"],
    )
    return Monitor(env)

def get_natural_episode_steps():
    """
    读取环境自然回合长度（由 resources.csv 决定的 total_steps）。
    用于动态修正 total_timesteps 上限，避免按 episode 训练时过早触发 timesteps 兜底。
    """
    probe_env = BatchJobEnv(
        task_csv="RL/batch.csv",
        resource_csv="RL/resources.csv",
        **TRAIN_CONFIG["env_kwargs"],
    )
    natural_steps = int(probe_env.data_stream.total_steps)
    probe_env.close()
    return natural_steps


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_wandb_run(config):
    """
    尝试按 WANDB_MODE 初始化。
    - WANDB_MODE=online: 在线模式，失败后自动回退 offline
    - WANDB_MODE=offline: 直接离线
    """
    requested_mode = os.getenv("WANDB_MODE", "online").strip().lower()
    if requested_mode not in {"online", "offline", "disabled"}:
        requested_mode = "online"

    if requested_mode == "disabled":
        print("W&B disabled by WANDB_MODE=disabled")
        return None

    init_kwargs = dict(
        project="batch_job_rl",
        name="ppo_train",
        sync_tensorboard=True,
        monitor_gym=True,
        config=config,
        mode=requested_mode,
    )

    try:
        run = wandb.init(**init_kwargs)
        print(f"W&B initialized in {requested_mode} mode.")
        return run
    except Exception as e:
        if requested_mode == "online":
            print(f"W&B online init failed: {e}")
            print("Falling back to W&B offline mode.")
            init_kwargs["mode"] = "offline"
            run = wandb.init(**init_kwargs)
            print("W&B initialized in offline mode.")
            return run
        print(f"W&B init failed in {requested_mode} mode: {e}")
        return None


def save_training_info_txt(models_dir, train_config, runtime_info):
    """
    在模型保存目录写入训练参数文本，便于后续复现实验。
    """
    os.makedirs(models_dir, exist_ok=True)
    info_path = os.path.join(models_dir, "training_info.txt")

    lines = [
        "=== Training Info ===",
        f"saved_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "[Runtime]",
    ]
    for k, v in runtime_info.items():
        lines.append(f"{k}: {v}")

    lines.extend(["", "[TRAIN_CONFIG]", json.dumps(train_config, ensure_ascii=False, indent=2)])

    with open(info_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Training info saved to: {info_path}")


def main():
    models_dir = MODELS_DIR
    log_dir = LOG_DIR
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    set_global_seed(TRAIN_CONFIG["seed"])
    wandb_run = init_wandb_run(TRAIN_CONFIG)
    run_id = wandb_run.id if wandb_run is not None else "local_run"
    if wandb_run is not None:
        wandb.define_metric("episode/index")
        wandb.define_metric("episode/*", step_metric="episode/index")

    # 向量化环境与归一化
    env = DummyVecEnv([make_env])
    env.seed(TRAIN_CONFIG["seed"])
    if TRAIN_CONFIG["resume_vecnormalize_path"]:
        env = VecNormalize.load(TRAIN_CONFIG["resume_vecnormalize_path"], env)
        env.training = True
        env.norm_reward = True
        print(f"Loaded VecNormalize stats from: {TRAIN_CONFIG['resume_vecnormalize_path']}")
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # PPO 对 Observation/Reward 尺度敏感，VecNormalize 可提升稳定性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if TRAIN_CONFIG["resume_model_path"]:
        model = PPO.load(
            TRAIN_CONFIG["resume_model_path"],
            env=env,
            device=device,
        )
        print(f"Resumed PPO model from: {TRAIN_CONFIG['resume_model_path']}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            learning_rate=TRAIN_CONFIG["learning_rate"],
            n_steps=TRAIN_CONFIG["n_steps"],
            batch_size=TRAIN_CONFIG["batch_size"],
            ent_coef=TRAIN_CONFIG["ent_coef"],
            gamma=TRAIN_CONFIG["gamma"],
            gae_lambda=TRAIN_CONFIG["gae_lambda"],
            clip_range=TRAIN_CONFIG["clip_range"],
            n_epochs=TRAIN_CONFIG["n_epochs"],
            target_kl=TRAIN_CONFIG["target_kl"],
            seed=TRAIN_CONFIG["seed"],
            policy_kwargs=dict(net_arch=TRAIN_CONFIG["policy_net_arch"]),
            tensorboard_log=f"runs/{run_id}",
        )

    print("Starting training...")
    callbacks = [
        TensorboardCallback(
            print_every_steps=TRAIN_CONFIG["print_every_steps"],
            max_episodes=TRAIN_CONFIG["total_episodes"],
            weight_log_every_steps=TRAIN_CONFIG["weight_log_every_steps"],
            weight_hist_every_steps=TRAIN_CONFIG["weight_hist_every_steps"],
            checkpoint_every_episodes=TRAIN_CONFIG["checkpoint_every_episodes"],
            checkpoint_prefix=TRAIN_CONFIG["checkpoint_prefix"],
            models_dir=models_dir,
        )
    ]
    if wandb_run is not None:
        callbacks.append(
            WandbCallback(
                gradient_save_freq=0,
                model_save_path=models_dir,
                verbose=0,
            )
        )

    natural_episode_steps = get_natural_episode_steps()
    effective_steps_per_episode_cap = max(
        int(TRAIN_CONFIG["max_steps_per_episode"]),
        int(natural_episode_steps),
    )
    if natural_episode_steps > int(TRAIN_CONFIG["max_steps_per_episode"]):
        print(
            "Auto-adjust timesteps cap per episode from "
            f"{TRAIN_CONFIG['max_steps_per_episode']} to {natural_episode_steps} "
            "to match environment natural episode length."
        )

    total_timesteps_cap = (
        int(TRAIN_CONFIG["total_episodes"]) * effective_steps_per_episode_cap
    )
    if wandb_run is not None:
        wandb.config.update(
            {
                "natural_episode_steps": natural_episode_steps,
                "effective_steps_per_episode_cap": effective_steps_per_episode_cap,
                "total_timesteps_cap": total_timesteps_cap,
            },
            allow_val_change=True,
        )

    model.learn(
        total_timesteps=total_timesteps_cap,
        callback=callbacks,
        log_interval=1,
        progress_bar=True,
    )

    model.save(f"{models_dir}/batch_job_ppo")
    env.save(f"{models_dir}/vec_normalize.pkl")
    save_training_info_txt(
        models_dir=models_dir,
        train_config=TRAIN_CONFIG,
        runtime_info={
            "run_id": run_id,
            "device": device,
            "natural_episode_steps": natural_episode_steps,
            "effective_steps_per_episode_cap": effective_steps_per_episode_cap,
            "total_timesteps_cap": total_timesteps_cap,
            "resume_model_path": TRAIN_CONFIG["resume_model_path"],
            "resume_vecnormalize_path": TRAIN_CONFIG["resume_vecnormalize_path"],
        },
    )

    print("Training finished and model saved.")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
