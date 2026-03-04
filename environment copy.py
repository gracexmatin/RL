import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from predictor import ResourcePredictor

# ==========================================
# 1. 数据流管理：提供 Ground Truth
# ==========================================
class DataStream:
    def __init__(
        self,
        resource_csv="RL/resources.csv",
        task_csv="RL/batch.csv",
        step_minutes=5.0,
        step_start_time=None,
    ):
        # 读取资源与电价数据 (Ground Truth)
        try:
            self.env_data = pd.read_csv(resource_csv, encoding="utf-8")
        except UnicodeDecodeError:
            self.env_data = pd.read_csv(resource_csv, encoding="gbk")
        self.step_minutes = float(step_minutes)
        self.step_start_time = pd.Timestamp(step_start_time) if step_start_time else None
        self.start_time = None
        self.step_offset = 0
        time_col = None
        for candidate in ["时间", "timestamp", "time"]:
            if candidate in self.env_data.columns:
                time_col = candidate
                break
        if time_col is None and "step" not in self.env_data.columns:
            raise ValueError("resources.csv must include a 'step' column or a datetime column.")

        if time_col is not None:
            time_series = pd.to_datetime(self.env_data[time_col])
            self.start_time = time_series.min()
            if self.step_start_time is None:
                raise ValueError("step_start_time is required to align steps with resources.csv.")
            step_seconds = self.step_minutes * 60.0
            self.step_offset = int(
                round((self.start_time - self.step_start_time).total_seconds() / step_seconds)
            )

            full_times = pd.date_range(
                start=self.start_time,
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
            filled["step"] = (
                (pd.to_datetime(filled[time_col]) - self.step_start_time).dt.total_seconds()
                / step_seconds
            ).round().astype(int)
            self.env_data = filled
        else:
            step_series = self.env_data["step"].astype(int)
            min_step = int(step_series.min())
            max_step = int(step_series.max())
            full_steps = np.arange(min_step, max_step + 1, dtype=np.int64)
            filled = self.env_data.copy().set_index("step").reindex(full_steps).fillna(0.0).reset_index()
            self.env_data = filled

        self.env_data = self.env_data.sort_values("step").set_index("step")
        self.total_steps = int(self.env_data.index.max()) + 1
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

    def _row_by_step(self, step):
        if self.step_start_time is None or self.step_offset == 0:
            if step not in self.env_data.index:
                return self.env_data.iloc[-1]
            return self.env_data.loc[step]
        pos = int(step) + int(self.step_offset)
        if pos < 0:
            return self.env_data.iloc[0]
        if pos >= len(self.env_data):
            return self.env_data.iloc[-1]
        return self.env_data.iloc[pos]

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
        return [t for t in self.task_queue if t['start_time'] == step]

    def time_to_step(self, ts):
        if self.step_start_time is None:
            raise ValueError("step_start_time is required to map time to step.")
        delta = pd.to_datetime(ts) - self.step_start_time
        return int(round(delta.total_seconds() / (self.step_minutes * 60.0)))

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
        history_window=1,  # 确保这个窗口大小与 Predictor 训练时一致
        predictor_model_path=None,
        predictor_tasks_df=None,
        interactive_task_csv="RL/interactive.csv",
        predictor_parallel_gpu=None,
    ):
        super().__init__()
        
        self.max_slots = max_slots
        self.history_window = int(history_window)
        
        # 核心时间参数
        self.step_minutes = 5.0
        self.step_duration = self.step_minutes / 60.0 
        self.default_startup_penalty = 5.0 / 60.0    
        self.step_start_time = pd.Timestamp("2025-11-01 00:00:00")
        self.data_stream = DataStream(
            resource_csv=resource_csv,
            task_csv=task_csv,
            step_minutes=self.step_minutes,
            step_start_time=self.step_start_time,
        )
        
        # === 初始化 Predictor ===
        self.predictor_model_path = predictor_model_path
        if not self.predictor_model_path:
            raise ValueError("predictor_model_path is required.")
        self.predictor = ResourcePredictor(model_path=self.predictor_model_path)
        if predictor_tasks_df is not None:
            self.predictor_tasks_df = predictor_tasks_df
        else:
            try:
                self.predictor_tasks_df = pd.read_csv(interactive_task_csv, encoding="utf-8")
            except UnicodeDecodeError:
                self.predictor_tasks_df = pd.read_csv(interactive_task_csv, encoding="gbk")
        self.predictor_parallel_gpu = self._ensure_parallel_gpu_time(predictor_parallel_gpu)

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
        
        # 这里的 history 存储真实的 (cap, price)，用于喂给 predictor
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
        
        # 重置数据流（不重复读取 CSV）
        self.data_stream.reset()
        
        # === 预热历史数据 (Warm-start History) ===
        # Predictor 需要过去 history_window 步的数据才能预测 Step 0
        # 如果当前是 Step 0，我们需要填充 padding 或者读取 step < 0 的数据(如果存在)
        self.history = []
        # 为了简单，我们读取 start_step 之前的数据，如果没有则 Pad 0
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
        
        # ... [中间的任务填入、动作解析、Real GPU Allocation 逻辑保持不变] ...
        # (为了节省篇幅，此处省略中间未修改的代码，逻辑与原文件一致)
        
        # 填入新任务
        for task in new_tasks:
            empty_slots = [i for i in range(self.max_slots) if self.slots[i] is None]
            if not empty_slots:
                break
            i = int(np.random.choice(empty_slots))
            self.slots[i] = task.copy()
            self.slots[i]['remaining'] = task['workload']
            self.slots[i]['startup_penalty_hr'] = task.get("startup_penalty_min", 5.0) / 60.0
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

        # 物理约束检查：必须使用 curr_cap_truth
        total_demanded = np.sum(real_gpu_allocations)
        constraint_violation = False
        if total_demanded > curr_cap:
            constraint_violation = True

        # === 奖励计算逻辑 (保持不变) ===
        reward = 0
        step_cost = 0
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
            reward += 1.0 * work_done
            
            if task['remaining'] <= 0:
                reward += 20.0
                self.slots[i] = None
                real_gpu_allocations[i] = 0
                self.has_started[i] = False
            elif task.get("end_time") is not None and (task["end_time"] - self.current_step) < 0:
                reward -= 5.0

        reward -= step_cost
        if constraint_violation: reward -= 1000.0

        # === 状态转移的关键修改 ===
        # 1. 将当前时刻的真实数据加入历史，供下一步预测使用
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

        # 2. 生成包含预测值的 Next State
        obs = self._get_obs(next_cap, next_price)
            
        return obs, reward, terminated, truncated, {}

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
        cap_pred = self._predict_next_step()
        obs.extend([curr_price, cap_pred])
        
        return np.array(obs, dtype=np.float32)

    def _update_history(self, cap, price):
        """滚动更新历史窗口"""
        self.history.append((cap, price))
        if len(self.history) > self.history_window:
            self.history.pop(0)

    def _get_history_features(self):
        """将历史数据展平放入状态"""
        if self.history_window <= 0: return []
        # 如果历史不够，进行 padding (理论上在 reset 后不会发生，除非 window 很大)
        current_len = len(self.history)
        history_list = self.history[:]
        if current_len < self.history_window:
            pad = [(0.0, 0.0)] * (self.history_window - current_len)
            history_list = pad + history_list
            
        flat = []
        for c, p in history_list:
            flat.extend([c, p])
        return flat

    def _predict_next_step(self):
        """
        调用内部 Predictor 预测未来的资源情况。
        输入：self.history (序列数据)
        输出：scalar prediction
        """
        if self.history_window == 0:
            return 0.0
        if self.predictor_tasks_df is None:
            return self.history[-1][0] if self.history else 0.0
            
        try:
            pred = self.predictor.predict(
                horizon=1,
                tasks_df=self.predictor_tasks_df,
                parallel_gpu=self.predictor_parallel_gpu,
                current_step=self.current_step,
                step_start_time=self.step_start_time,
                step_minutes=self.step_minutes,
            )
            if isinstance(pred, (tuple, list)) and len(pred) == 2:
                mu, std = pred
                return float(mu) + 1.96 * float(std)
            pred_arr = np.asarray(pred)
            if pred_arr.size >= 2:
                mu, std = pred_arr.flatten()[:2]
                return float(mu) + 1.96 * float(std)
            return float(pred_arr.item())
        except Exception:
            return self.history[-1][0] if self.history else 0.0

    def _ensure_parallel_gpu_time(self, parallel_gpu):
        if parallel_gpu is None:
            return None
        if not isinstance(parallel_gpu, pd.DataFrame):
            raise ValueError("predictor_parallel_gpu must be a DataFrame.")
        if "parallel_gpu" not in parallel_gpu.columns:
            raise ValueError("predictor_parallel_gpu must include a 'parallel_gpu' column.")
        parallel_gpu = parallel_gpu.copy()
        if "step" in parallel_gpu.columns:
            steps = parallel_gpu["step"].to_numpy()
        elif parallel_gpu.index.name == "step" or np.issubdtype(parallel_gpu.index.dtype, np.integer):
            steps = parallel_gpu.index.to_numpy()
        else:
            steps = np.arange(len(parallel_gpu), dtype=np.int64)
        times = self.step_start_time + pd.to_timedelta(steps * self.step_minutes, unit="m")
        parallel_gpu["时间"] = times
        return parallel_gpu

# 简单测试
if __name__ == "__main__":
    # 需要有一个假的模型文件路径来跑测试
    # env = BatchJobEnv(predictor_model_path="RL/my_best_model.pth")
    # obs, _ = env.reset()
    # print("Obs shape:", obs.shape)
    pass
