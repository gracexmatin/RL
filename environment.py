import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

# ==========================================
# 1. 数据流管理：提供 Ground Truth
# ==========================================
class DataStream:
    def __init__(
        self,
        resource_csv="RL/resources.csv",
        task_csv="RL/batch.csv",
        step_minutes=5.0,
    ):
        # 读取资源与电价数据 (Ground Truth)
        try:
            self.env_data = pd.read_csv(resource_csv, encoding="utf-8")
        except UnicodeDecodeError:
            self.env_data = pd.read_csv(resource_csv, encoding="gbk")
        self.step_minutes = float(step_minutes)
        time_col = None
        for candidate in ["时间", "timestamp", "time"]:
            if candidate in self.env_data.columns:
                time_col = candidate
                break
        if time_col is None and "step" not in self.env_data.columns:
            first_col = self.env_data.columns[0]
            self.env_data = self.env_data.rename(columns={first_col: "step"})

        if time_col is not None:
            time_series = pd.to_datetime(self.env_data[time_col])
            start_time = time_series.min()
            step_seconds = self.step_minutes * 60.0

            full_times = pd.date_range(
                start=start_time,
                end=time_series.max(),
                freq=f"{int(self.step_minutes)}T",
            )
            filled = self.env_data.copy()
            filled[time_col] = time_series
            filled = (
                filled.groupby(time_col, as_index=True)
                .last()
                .reindex(full_times)
                .fillna(0.0)
                .reset_index()
            )
            filled = filled.rename(columns={"index": time_col})
            filled["step"] = np.arange(len(filled), dtype=np.int64)
            self.env_data = filled
        else:
            step_series = pd.to_numeric(self.env_data["step"], errors="coerce")
            step_series = step_series.replace([np.inf, -np.inf], np.nan).dropna()
            if step_series.empty:
                raise ValueError("resources.csv 'step' column is empty or invalid.")
            min_step = int(step_series.min())
            max_step = int(step_series.max())
            full_steps = np.arange(min_step, max_step + 1, dtype=np.int64)
            filled = (
                self.env_data.copy()
                .set_index("step")
                .reindex(full_steps)
                .fillna(0.0)
                .reset_index()
            )
            filled["step"] = filled["step"] - min_step
            self.env_data = filled

        self.env_data = self.env_data.sort_values("step").set_index("step")
        self.total_steps = len(self.env_data)
        self.has_concurrency = "parallel_gpu" in self.env_data.columns

        # 读取任务队列（批量型任务；start_time/end_time 是 step 序号，不是时间戳）
        task_df = pd.read_csv(task_csv)
        if "start_time" not in task_df.columns or "end_time" not in task_df.columns:
            raise ValueError("batch.csv must include 'start_time' and 'end_time' step columns.")
        # Coerce non-numeric/blank values and drop invalid rows to avoid NaN->int errors.
        task_df["start_time"] = pd.to_numeric(task_df["start_time"], errors="coerce")
        task_df["end_time"] = pd.to_numeric(task_df["end_time"], errors="coerce")
        task_df = task_df.replace([np.inf, -np.inf], np.nan)
        task_df = task_df.dropna(subset=["start_time", "end_time"])
        if not np.issubdtype(task_df["start_time"].dtype, np.integer):
            task_df["start_time"] = task_df["start_time"].round().astype(int)
        if not np.issubdtype(task_df["end_time"].dtype, np.integer):
            task_df["end_time"] = task_df["end_time"].round().astype(int)
        self.task_queue = task_df.to_dict(orient="records")
        self.task_queue.sort(key=lambda x: x["start_time"])
        # 预索引任务到各个 step，避免每个环境步 O(N) 线性扫描任务表。
        self.tasks_by_start_step = {}
        for task in self.task_queue:
            start_step = int(task["start_time"])
            self.tasks_by_start_step.setdefault(start_step, []).append(task)

    def _row_by_step(self, step):
        if step < 0:
            return self.env_data.iloc[0]
        if step not in self.env_data.index:
            # 越界处理：返回最后一行
            return self.env_data.iloc[-1]
        return self.env_data.loc[step]

    def get_ground_truth(self, step):
        """获取某一时刻可用cap和电价，用于计算 Reward 和验证约束"""
        row = self._row_by_step(step)
        resource_cap = row["resource_cap"]
        price = row["price"]
        if self.has_concurrency:
            cap = resource_cap - row["parallel_gpu"]
        else:
            cap = resource_cap
        if cap < 0:
            cap = 0.0
        return cap, price

    def get_concurrency(self, step):
        if not self.has_concurrency:
            return None
        row = self._row_by_step(step)
        return row["parallel_gpu"]

    def get_new_tasks(self, step):
        return self.tasks_by_start_step.get(int(step), [])

    def reset(self):
        # DataStream is immutable after load; keep for API symmetry.
        return

# ==========================================
# 2. 环境核心逻辑
# ==========================================
class BatchJobEnv(gym.Env):
    def __init__(
        self,
        max_slots=50,
        resource_csv="RL/resources.csv",
        task_csv="RL/batch.csv",
        history_window=1,
        constraint_penalty_coef=10.0,
        finish_bonus_base=5.0,
        finish_bonus_workload_coef=0.15,
        finish_bonus_slack_coef=0.1,
    ):
        super().__init__()
        
        self.max_slots = max_slots
        self.history_window = int(history_window)
        
        # 核心时间参数
        self.step_minutes = 5.0
        self.step_duration = self.step_minutes / 60.0 
        self.default_startup_penalty = 5.0 / 60.0    
        self.constraint_penalty_coef = float(constraint_penalty_coef)
        self.finish_bonus_base = float(finish_bonus_base)
        self.finish_bonus_workload_coef = float(finish_bonus_workload_coef)
        self.finish_bonus_slack_coef = float(finish_bonus_slack_coef)
        self.data_stream = DataStream(
            resource_csv=resource_csv,
            task_csv=task_csv,
            step_minutes=self.step_minutes,
        )
        
        # 动作空间 (混合动作占位)
        self.action_space = spaces.Box(low=-1, high=1, shape=(max_slots * 2,), dtype=np.float32)
        
        # 状态空间
        self.slot_feat_dim = 7
        self.global_feat_dim = 2 # [current_price, current_cap_truth]
        self.history_feat_dim = self.history_window * 2 # 展平的历史
        
        total_obs_dim = max_slots * self.slot_feat_dim + self.global_feat_dim + self.history_feat_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32)

        self.current_step = 0
        self.slots = [None] * self.max_slots 
        self.prev_actions = np.zeros(self.max_slots) 
        self.has_started = [False] * self.max_slots
        
        # 这里的 history 存储真实的 (cap, price)，保留用于 Feature 观察
        self.history = [] 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        start_step = 0
        if options and "start_step" in options:
            start_step = int(options["start_step"])
        self.current_step = max(0, start_step)
        
        self.slots = [None] * self.max_slots
        self.prev_actions = np.zeros(self.max_slots)
        self.has_started = [False] * self.max_slots
        
        # 重置数据流
        self.data_stream.reset()
        
        # === 预热历史数据 (Warm-start History) ===
        # 历史特征是 Observation 的一部分。
        # 时序约定：s_t 的 history 仅包含 < t 的数据，不包含 t。
        # 在 step(t) 末尾写入 (cap_t, price_t)，因此它会出现在 s_{t+1} 中。
        self.history = []
        for s in range(self.current_step - self.history_window, self.current_step):
            if s < 0:
                self.history.append((0.0, 0.0)) # Padding
            else:
                cap, price = self.data_stream.get_ground_truth(s)
                self.history.append((cap, price))
                
        # 获取初始 Observation
        curr_cap, curr_price = self.data_stream.get_ground_truth(self.current_step)
        return self._get_obs(curr_cap, curr_price), {}

    def step(self, raw_action):
        # 1. 获取当前时刻真实的 Ground Truth (用于计算 Cost 和 物理约束)
        curr_cap, curr_price = self.data_stream.get_ground_truth(self.current_step)
        new_tasks = self.data_stream.get_new_tasks(self.current_step)
        
        # 填入新任务
        for task in new_tasks:
            empty_slots = [i for i in range(self.max_slots) if self.slots[i] is None]
            if not empty_slots:
                break
            i = int(self.np_random.choice(empty_slots))
            self.slots[i] = task.copy()
            self.slots[i]['remaining'] = task['workload']
            self.slots[i]['initial_workload'] = task['workload']
            self.slots[i]['startup_penalty_hr'] = task.get("startup_penalty_min", 5.0) / 60.0
            self.slots[i]['ddl_penalized'] = False
            self.has_started[i] = False

        # 解析动作 & 约束检查
        real_gpu_allocations = np.zeros(self.max_slots)
        for i in range(self.max_slots):
            if self.slots[i] is None: continue
            gate_val = raw_action[i]
            amount_val = raw_action[i + self.max_slots]
            task = self.slots[i]
            if (gate_val + 1.0) / 2.0 >= 0.5:
                normalized = (amount_val + 1.0) / 2.0
                real_gpu_allocations[i] = task['min_gpu'] + normalized * (task['max_gpu'] - task['min_gpu'])

        # 物理约束检查：先按策略请求量统计，再进行硬裁剪
        total_demanded = float(np.sum(real_gpu_allocations))
        overflow = max(0.0, total_demanded - float(curr_cap))
        overflow_ratio = overflow / max(float(curr_cap), 1e-6)
        overflow_ratio_capped = min(overflow_ratio, 3.0)
        # 硬约束：实际执行量不允许超过当前可用容量
        # 使用按比例缩放，避免出现“超容量也照常推进任务”的不可执行行为。
        if total_demanded > float(curr_cap) and total_demanded > 0.0:
            scale = float(curr_cap) / total_demanded
            real_gpu_allocations *= scale

        executed_total = float(np.sum(real_gpu_allocations))

        # === 奖励计算逻辑 ===
        reward_work_done = 0.0
        reward_finish_bonus = 0.0
        penalty_deadline = 0.0
        step_cost = 0.0
        for i in range(self.max_slots):
            if self.slots[i] is None: continue
            gpu = real_gpu_allocations[i]
            task = self.slots[i]
            effective_time = 0.0
            if gpu > 0:
                if self.prev_actions[i] == 0: # 冷启动
                    if not self.has_started[i]: effective_time = self.step_duration
                    else: effective_time = max(0.0, self.step_duration - task['startup_penalty_hr'])
                else: effective_time = self.step_duration
                self.has_started[i] = True
            
            step_cost += gpu * self.step_duration * curr_price
            work_done = gpu * effective_time
            task['remaining'] -= work_done
            reward_work_done += 1.0 * work_done
            
            if task['remaining'] <= 0:
                # 完成奖励：基础奖励 + 任务规模奖励 + DDL 裕度奖励（提前完成更多）
                slack_steps = max(0.0, float(task.get("end_time", self.current_step) - self.current_step))
                finish_bonus = (
                    self.finish_bonus_base
                    + self.finish_bonus_workload_coef * float(task.get("initial_workload", 0.0))
                    + self.finish_bonus_slack_coef * slack_steps
                )
                reward_finish_bonus += finish_bonus
                self.slots[i] = None
                real_gpu_allocations[i] = 0
                self.has_started[i] = False
            elif (
                task.get("end_time") is not None
                and (task["end_time"] - self.current_step) < 0
                and not task.get("ddl_penalized", False)
            ):
                penalty_deadline += 5.0
                task["ddl_penalized"] = True

        # 软惩罚：裁剪后仍惩罚“超额请求”本身，鼓励策略减少无效动作。
        penalty_constraint = self.constraint_penalty_coef * overflow_ratio_capped if overflow > 0.0 else 0.0
        reward = (
            reward_work_done
            + reward_finish_bonus
            - step_cost
            - penalty_deadline
            - penalty_constraint
        )

        # === 状态转移 ===
        # 1. 将当前时刻的真实数据加入历史
        self._update_history(curr_cap, curr_price)
        
        self.prev_actions = real_gpu_allocations.copy()
        self.current_step += 1
        
        terminated = False
        truncated = False
        if self.current_step >= self.data_stream.total_steps:
            truncated = True
            # 防止越界读取
            next_cap, next_price = curr_cap, curr_price
        else:
            # 获取下一时刻的 Truth 用于构建 State (Agent可以看到当前时刻的电价和Cap)
            next_cap, next_price = self.data_stream.get_ground_truth(self.current_step)

        # 2. 生成包含当前容量特征（由 Truth 读取）的 Next State
        obs = self._get_obs(next_cap, next_price)
            
        info = {
            "reward_work_done": float(reward_work_done),
            "reward_finish_bonus": float(reward_finish_bonus),
            "penalty_energy_cost": float(step_cost),
            "penalty_deadline": float(penalty_deadline),
            "penalty_constraint": float(penalty_constraint),
            "reward_total": float(reward),
            "total_demanded": float(total_demanded),
            "total_executed": float(executed_total),
            "curr_cap": float(curr_cap),
            "overflow": float(overflow),
            "overflow_ratio": float(overflow_ratio),
            "overflow_ratio_capped": float(overflow_ratio_capped),
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self, curr_cap, curr_price):
        """
        构建状态向量：
        State = [Slot_Features, History_Features, Global_Features]
        Global_Features = [Price_t, Cap_t_Truth]
        """
        obs = []
        # 1. Slot Features
        for i in range(self.max_slots):
            t = self.slots[i]
            prev_flag = 1.0 if self.prev_actions[i] > 0 else 0.0
            started_flag = 1.0 if self.has_started[i] else 0.0
            if t is None:
                obs.extend([0, 0, 0, 0, 0, 0, 0])
            else:
                obs.extend([
                    t['remaining'], 
                    t['end_time'] - self.current_step,
                    t['min_gpu'],
                    t['max_gpu'],
                    t.get("startup_penalty_hr", self.default_startup_penalty),
                    prev_flag,
                    started_flag
                ])
        
        # 2. History Features (Flattened)
        hist_feat = self._get_history_features()
        obs.extend(hist_feat)

        # 3. Global Features
        # 这里调用 _get_current_cap_feature 获取当前时刻容量（由真值读取）
        current_cap_feature = self._get_current_cap_feature()
        obs.extend([curr_price, current_cap_feature])
        
        return np.array(obs, dtype=np.float32)

    def _update_history(self, cap, price):
        """滚动更新历史窗口"""
        self.history.append((cap, price))
        if len(self.history) > self.history_window:
            self.history.pop(0)

    def _get_history_features(self):
        """将历史数据展平放入状态"""
        if self.history_window <= 0: return []
        current_len = len(self.history)
        history_list = self.history[:]
        if current_len < self.history_window:
            pad = [(0.0, 0.0)] * (self.history_window - current_len)
            history_list = pad + history_list
            
        flat = []
        for c, p in history_list:
            flat.extend([c, p])
        return flat

    def _get_current_cap_feature(self):
        """
        读取当前时刻的真实容量，用作状态中的全局特征。
        """
        target_step = self.current_step

        # 读取当前 step 的真实值
        current_cap, _ = self.data_stream.get_ground_truth(target_step)
        
        return current_cap

# 简单测试
if __name__ == "__main__":
    env = BatchJobEnv(resource_csv="RL/resources.csv", task_csv="RL/batch.csv")
    obs, _ = env.reset()
    print("Obs shape:", obs.shape)
    
    # 随机跑一步
    action = env.action_space.sample()
    obs, reward, term, trunc, _ = env.step(action)
    print("Step 1 Reward:", reward)
