import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. 必须保留的模型结构定义 (用于加载权重)
# ==========================================
class DeepResidualAttentionNet(nn.Module):
    def __init__(self, dense_dim=881, task_dim=2, hidden_dim=16, dense_proj_dim=64):
        super().__init__()
        # --- A. LSTM 部分 ---
        self.lstm = nn.LSTM(
            input_size=task_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        # --- B. 投影层 ---
        self.dense_projection = nn.Sequential(
            nn.Linear(dense_dim, dense_proj_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # --- C. 混合 MLP ---
        concat_dim = hidden_dim + dense_proj_dim
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.res_mu = nn.Linear(32, 1)
        self.res_var = nn.Linear(32, 1)

    def forward(self, seq, lens, dense_scaled):
        # lens: (B,)
        lens = lens.view(-1)
        # 1. LSTM & Attention
        lstm_out, _ = self.lstm(seq)  # (B, L, H)
        attn_scores = self.attention_net(lstm_out).squeeze(2)
        B, L, _ = lstm_out.shape
        
        # Masking padding
        arange_tensor = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, -1)
        mask = arange_tensor >= lens.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(mask, -1e9)
        
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(2)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (B, H)

        # 2. Dense Projection
        dense_proj = self.dense_projection(dense_scaled)     # (B, 64)

        # 3. Concat
        feat = torch.cat([context, dense_proj], dim=1)       # (B, 128)

        # 4. MLP & Head
        h = self.mlp(feat)
        mu = self.res_mu(h).squeeze(1)
        var = torch.nn.functional.softplus(self.res_var(h).squeeze(1)) + 1e-4
        return mu, var

# ==========================================
# 2. 新的 Predictor 类
# ==========================================
class ResourcePredictor:
    def __init__(self, model_path: Optional[str] = None, device=None):
        """
        :param env_data: 包含历史数据的 DataFrame，必须包含 ['开始时间', '结束时间', '时长_分钟']
        :param model_path: .pth 模型文件路径
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not model_path:
            raise ValueError("model_path is required to initialize ResourcePredictor.")

        # 1. 加载模型和 Scaler
        print(f"📂 Predictor: 加载模型 {model_path} ...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 恢复 Scalers
        self.x_scaler = StandardScaler()
        self.x_scaler.mean_ = checkpoint['x_scaler_mean']
        self.x_scaler.scale_ = checkpoint['x_scaler_scale']
        self.x_scaler.var_ = self.x_scaler.scale_ ** 2
        
        self.y_scaler = StandardScaler()
        self.y_scaler.mean_ = checkpoint['y_scaler_mean']
        self.y_scaler.scale_ = checkpoint['y_scaler_scale']
        self.y_scaler.var_ = self.y_scaler.scale_ ** 2

        self.df = None
        self.starts = None
        self.ends = None
        self.durations = None
        self.time_index = None
        self.lookup_series = None
        self.resource_index = None

        # 恢复 Ridge 参数
        self.w_coef = checkpoint['ridge_coef'].to(self.device)
        self.w_inter = checkpoint['ridge_inter'].to(self.device)

        # 恢复 Deep 模型
        # 注意：训练代码中 dense_dim 写死为 881，如果不一致需要参数化
        self.model = DeepResidualAttentionNet(dense_dim=881, dense_proj_dim=64).to(self.device)
        self.model.load_state_dict(checkpoint['deep_state_dict'])
        self.model.eval()

        # 序列特征的 Scaler (训练代码中是硬编码 fit 的)
        self.feat_scaler = StandardScaler()
        self.feat_scaler.fit(np.array([[0, 10], [7 * 24 * 3600, 1000]]))

        self.lookback_window = pd.Timedelta(days=7.0)
        self.max_seq_len = 512

    def _normalize_parallel_gpu(self, parallel_gpu: pd.DataFrame):
        if not isinstance(parallel_gpu, pd.DataFrame):
            raise ValueError("parallel_gpu must be a DataFrame with time and parallel_gpu columns.")
        if "parallel_gpu" not in parallel_gpu.columns:
            raise ValueError("parallel_gpu must include a 'parallel_gpu' column.")

        time_col = None
        for candidate in ["时间", "timestamp", "time", "开始时间"]:
            if candidate in parallel_gpu.columns:
                time_col = candidate
                break
        if time_col is not None:
            res_time = pd.to_datetime(parallel_gpu[time_col])
            res_series = parallel_gpu["parallel_gpu"].astype(np.float32).values
            return pd.Series(res_series, index=res_time).sort_index()
        if isinstance(parallel_gpu.index, pd.DatetimeIndex):
            res_series = parallel_gpu["parallel_gpu"].astype(np.float32).values
            return pd.Series(res_series, index=parallel_gpu.index).sort_index()
        raise ValueError("parallel_gpu must have a datetime column (e.g. 时间/timestamp) or a DatetimeIndex.")

    def _normalize_tasks_df(
        self,
        tasks_df: pd.DataFrame,
        current_time: Optional[pd.Timestamp],
        step_start_time: Optional[pd.Timestamp],
        step_minutes: float,
    ):
        if "开始时间" in tasks_df.columns and "结束时间" in tasks_df.columns:
            df = tasks_df.copy()
            df["开始时间"] = pd.to_datetime(df["开始时间"])
            df["结束时间"] = pd.to_datetime(df["结束时间"])
        elif "start_time" in tasks_df.columns and "end_time" in tasks_df.columns:
            df = tasks_df.copy()
            df["开始时间"] = pd.to_datetime(df["start_time"])
            df["结束时间"] = pd.to_datetime(df["end_time"])
        else:
            raise ValueError("tasks_df must include 开始时间/结束时间 or start_time/end_time columns.")

        if current_time is not None:
            df = df.sort_values("开始时间")
            times = df["开始时间"].values
            week_start = current_time - pd.Timedelta(days=7)
            left = np.searchsorted(times, np.datetime64(week_start), side="left")
            right = np.searchsorted(times, np.datetime64(current_time), side="right")
            df = df.iloc[left:right]
            df["结束时间"] = df["结束时间"].clip(upper=current_time)
            if len(df) > 512:
                df = df.iloc[-512:]
        return df

    def _truncate_parallel_gpu(self, series: pd.Series, current_time: Optional[pd.Timestamp]):
        if current_time is None:
            return series
        hist_start = current_time - pd.Timedelta(days=3)
        lw_t = current_time - pd.Timedelta(days=7)
        lw_start = lw_t - pd.Timedelta(minutes=30)
        lw_end = lw_t + pd.Timedelta(minutes=30)
        mask = ((series.index >= hist_start) & (series.index <= current_time)) | (
            (series.index >= lw_start) & (series.index <= lw_end)
        )
        return series.loc[mask]

    def _build_lookup_series_from_tasks(self):
        all_min = self.df['开始时间'].min()
        all_max = self.df['结束时间'].max()
        self.time_index = pd.date_range(start=all_min, end=all_max, freq='5T')

        counts = []
        for ts in (self.time_index.view(np.int64) // 10**9):
            idx = np.searchsorted(self.starts, ts, side='right')
            counts.append(np.sum(self.ends[:idx] > ts))

        scaled_counts = self.y_scaler.transform(np.array(counts).reshape(-1, 1)).flatten()
        self.lookup_series = pd.Series(scaled_counts, index=self.time_index)
        self.resource_index = self.lookup_series.index

    def set_env_data(self, tasks_df: pd.DataFrame, parallel_gpu: Optional[object] = None):
        # 数据预处理（构建 lookup_series，只需做一次）
        self.df = tasks_df.sort_values('开始时间').reset_index(drop=True)
        self.df['开始时间'] = pd.to_datetime(self.df['开始时间'])
        self.df['结束时间'] = pd.to_datetime(self.df['结束时间'])
        if '时长_分钟' not in self.df.columns:
            self.df['时长_分钟'] = (self.df['结束时间'] - self.df['开始时间']).dt.total_seconds() / 60.0

        # 预计算用于序列特征的数组
        self.starts = (self.df['开始时间'].astype(np.int64) // 10**9).values
        self.ends = (self.df['结束时间'].astype(np.int64) // 10**9).values
        self.durations = self.df['时长_分钟'].values

        # 优先使用传入的 parallel_gpu 序列，否则按训练格式从任务数据重建
        if parallel_gpu is not None:
            raw_series = self._normalize_parallel_gpu(parallel_gpu)
            self.resource_index = raw_series.index
            scaled_counts = self.y_scaler.transform(raw_series.values.reshape(-1, 1)).flatten()
            self.lookup_series = pd.Series(scaled_counts, index=self.resource_index)
        else:
            self._build_lookup_series_from_tasks()

    def _extract_time_features(self, dt_index):
        # 批量提取时间特征
        # dt_index: pd.DatetimeIndex
        h = dt_index.hour.values
        d = dt_index.dayofweek.values
        
        f1 = np.sin(2 * np.pi * h / 24)
        f2 = np.cos(2 * np.pi * h / 24)
        f3 = np.sin(2 * np.pi * d / 7)
        f4 = np.cos(2 * np.pi * d / 7)
        f5 = (d >= 5).astype(np.float32)
        
        return np.stack([f1, f2, f3, f4, f5], axis=1).astype(np.float32)

    def _prepare_batch(self, target_times):
        """
        为一组目标时间构建 Batch 输入
        """
        if self.lookup_series is None:
            raise ValueError("ResourcePredictor has no env_data. Call set_env_data(...) first.")
        batch_seq = []
        batch_lens = []
        batch_dense = []

        for t in target_times:
            # --- 1. Dense Features Construction ---
            # A. 历史 3 天 (864 points)
            hist_end = t - pd.Timedelta(minutes=5)
            hist_start = t - pd.Timedelta(days=3)
            # 使用 asof 或切片，注意处理边界
            h_slice = self.lookup_series.loc[hist_start:hist_end]
            h_values = h_slice.values[-864:] # 取最后 864 个
            
            # Padding if insufficient history
            if len(h_values) < 864:
                h_values = np.pad(h_values, (864 - len(h_values), 0), 'constant')
            
            # B. 同比上周 (12 points)
            lw_t = t - pd.Timedelta(days=7)
            # 注意：这里需要精确切片，如果索引不对齐可能为空，需容错
            lw_slice = self.lookup_series.loc[lw_t - pd.Timedelta(minutes=30) : lw_t + pd.Timedelta(minutes=30)]
            lw_values = lw_slice.values[:12]
            if len(lw_values) < 12:
                lw_values = np.pad(lw_values, (0, 12 - len(lw_values)), 'constant')

            # C. 时间特征
            t_feat = self._extract_time_features(pd.DatetimeIndex([t]))[0]

            # Concatenate Dense
            dense_vec = np.concatenate([h_values, lw_values, t_feat]).astype(np.float32)
            batch_dense.append(dense_vec)

            # --- 2. Sequence Features Construction ---
            target_ts = t.timestamp()
            window_start = target_ts - self.lookback_window.total_seconds()
            
            # 快速查找范围内的任务
            s_idx = np.searchsorted(self.starts, window_start, side='left')
            e_idx = np.searchsorted(self.starts, target_ts, side='right')
            
            rel_starts = self.starts[s_idx:e_idx]
            raw_durs = self.durations[s_idx:e_idx]
            
            if len(rel_starts) > 0:
                elapsed_mins = (target_ts - rel_starts) / 60.0
                rel_durs = np.minimum(raw_durs, elapsed_mins)
                rel_durs = np.maximum(rel_durs, 0.0)
                
                seq_len = min(len(rel_starts), self.max_seq_len)
                
                # 构建 (relative_start_sec, duration_min)
                raw_feats = np.hstack([
                    (target_ts - rel_starts[-seq_len:]).reshape(-1, 1),
                    rel_durs[-seq_len:].reshape(-1, 1)
                ])
                # Scaling
                seq_feats = self.feat_scaler.transform(raw_feats)
                
                # Padding to max_seq_len
                padded = np.zeros((self.max_seq_len, 2), dtype=np.float32)
                padded[:seq_len, :] = seq_feats
                
                batch_seq.append(padded)
                batch_lens.append(seq_len)
            else:
                batch_seq.append(np.zeros((self.max_seq_len, 2), dtype=np.float32))
                batch_lens.append(1) # 避免长度为0报错

        # Convert to Tensors
        batch_seq = torch.FloatTensor(np.array(batch_seq)).to(self.device)
        batch_lens = torch.LongTensor(batch_lens).to(self.device)
        batch_dense = torch.FloatTensor(np.array(batch_dense)).to(self.device) # Unscaled dense feats
        
        return batch_seq, batch_lens, batch_dense

    def predict(
        self,
        horizon=12,
        tasks_df: Optional[pd.DataFrame] = None,
        parallel_gpu: Optional[pd.DataFrame] = None,
        current_step: Optional[int] = None,
        step_start_time: Optional[pd.Timestamp] = None,
        step_minutes: float = 5.0,
    ):
        """
        :param horizon: 预测未来多少个步长 (假设每步 5 分钟)
        :param tasks_df: 可选，若提供则在预测前更新任务数据
        :param parallel_gpu: 可选，parallel_gpu DataFrame (含时间列或 DatetimeIndex + parallel_gpu 列)
        :return: np.array [horizon]
        """
        current_time = None
        if current_step is not None:
            if step_start_time is None:
                raise ValueError("step_start_time is required when current_step is provided.")
            current_time = pd.Timestamp(step_start_time) + pd.to_timedelta(
                int(current_step) * step_minutes, unit="m"
            )
        if (tasks_df is not None or parallel_gpu is not None) and current_time is None:
            raise ValueError("current_step and step_start_time are required for time-based truncation.")

        if tasks_df is not None:
            norm_tasks = self._normalize_tasks_df(tasks_df, current_time, step_start_time, step_minutes)
            if parallel_gpu is not None:
                series = self._normalize_parallel_gpu(parallel_gpu)
                series = self._truncate_parallel_gpu(series, current_time)
                parallel_gpu = pd.DataFrame({"时间": series.index, "parallel_gpu": series.values})
            self.set_env_data(norm_tasks, parallel_gpu)
        if self.df is None:
            raise ValueError("ResourcePredictor has no env_data. Call set_env_data(...) before predict().")
        # 1. 确定时间锚点
        # 使用最新的时间作为预测起点
        if current_time is not None:
            current_time = current_time
        elif self.resource_index is not None and len(self.resource_index) > 0:
            current_time = self.resource_index[-1]
        elif self.lookup_series is not None and len(self.lookup_series.index) > 0:
            current_time = self.lookup_series.index[-1]
        else:
            current_time = self.df['开始时间'].iloc[-1]

        # 2. 生成未来 horizon 个时间点
        # 注意：predict 通常是预测 current_time 之后的时间
        future_times = [current_time + pd.Timedelta(minutes=5 * (i+1)) for i in range(horizon)]
        
        # 3. 准备 Batch 数据
        seq, lens, dense = self._prepare_batch(future_times)

        # 4. 推理
        with torch.no_grad():
            # Standardize Dense Features (Using loaded mean/scale)
            # 手动实现 StandardScaler 的 transform (x - u) / s
            dense_mean = torch.tensor(self.x_scaler.mean_, dtype=torch.float32, device=self.device)
            dense_scale = torch.tensor(self.x_scaler.scale_, dtype=torch.float32, device=self.device)
            dense_scaled = (dense - dense_mean) / dense_scale

            # A. Wide Prediction (Ridge)
            wide_pred = dense_scaled.matmul(self.w_coef) + self.w_inter

            # B. Deep Prediction (Residual)
            r_mu, _ = self.model(seq, lens, dense_scaled)

            # C. Combine & Inverse Scale
            final_pred_scaled = wide_pred + r_mu
            final_pred_scaled_np = final_pred_scaled.cpu().numpy().reshape(-1, 1)
            
            final_pred = self.y_scaler.inverse_transform(final_pred_scaled_np).flatten()

        return final_pred.astype(np.float32)

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 假设这是你的环境数据
    task_path = "RL/interactive.csv"
    resource_path = "RL/resources.csv"
    model_path = "RL/my_best_model.pth"

    tasks_df = pd.read_csv(task_path)
    resources_df = pd.read_csv(resource_path)

    # 初始化 Predictor (只做一次)
    predictor = ResourcePredictor(model_path=model_path)

    # 预测未来 12 个步长 (比如未来1小时)
    preds = predictor.predict(horizon=12, tasks_df=tasks_df, parallel_gpu=resources_df)

    print(f"预测未来值: {preds}")
