# RL 环境说明（`environment.py`）

本文档用于记录 `RL/environment.py` 的当前行为，确保后续检查时口径一致。

## 1. 范围

- 文件：`RL/environment.py`
- 环境类：`BatchJobEnv`
- 数据流类：`DataStream`
- 时间步长：`5` 分钟（`step_duration = 5/60` 小时）

## 2. 数据与时间线

### 2.1 资源流（`resources.csv`）

- 每个时间步 `t` 使用：
  - `resource_cap`
  - `price`
  - 可选 `parallel_gpu`
- 有效可用容量：
  - 若存在 `parallel_gpu`：`cap_t = max(resource_cap - parallel_gpu, 0)`
  - 否则：`cap_t = resource_cap`

### 2.2 任务流（`batch.csv`）

- 必需列：`start_time`、`end_time`
- 任务按批处理作业处理，包含 workload 与 GPU 上下界（`min_gpu`、`max_gpu`）。
- 当 `start_time == current_step` 时，新任务会被放入空槽位。

### 2.3 时间推进

- `reset()` 在 `current_step` 构建初始状态。
- `step(action)` 完成从 `t` 到 `t+1` 的状态转移。
- 历史特征约定：
  - `s_t` 的 history 只包含 `< t` 的数据（不包含 `t`）
  - `(cap_t, price_t)` 在 `step(t)` 末尾写入，因此会出现在 `s_{t+1}`

## 3. State 设计

观测是一个一维向量：

`State = [Slot_Features, History_Features, Global_Features]`

### 3.1 槽位特征（Slot Features）

- 每个 slot 维度：`7`
- 顺序：
  1. `remaining`
  2. `end_time - current_step`（截止时间剩余步数）
  3. `min_gpu`
  4. `max_gpu`
  5. `startup_penalty_hr`
  6. `prev_flag`（上一步分配 GPU > 0 则为 `1`，否则 `0`）
  7. `started_flag`（该任务是否曾经启动过，是 `1` 否 `0`）
- 空 slot 使用 7 个 0 填充。

### 3.2 历史特征（History Features）

- 维度：`history_window * 2`
- 由最近若干步 `(cap, price)` 展平得到
- 历史不足时，在左侧补 0

### 3.3 全局特征（Global Features）

- 维度：`2`
- 顺序：
  1. `curr_price`
  2. `current_cap_feature`（当前步真实容量）

### 3.4 观测总维度

`max_slots * 7 + history_window * 2 + 2`

## 4. Action 设计

动作是连续空间：`Box(low=-1, high=1, shape=(max_slots * 2,))`。

- 前半段（`0 : max_slots`）：gate
  - 映射为 `(gate + 1)/2`
  - 映射值 `>= 0.5` 时认为该任务开启分配
- 后半段（`max_slots : 2*max_slots`）：amount
  - 从 `[-1, 1]` 映射到 `[0, 1]`
  - 再按任务自身区间缩放到 `[min_gpu, max_gpu]`
- 空槽位对应的动作会被忽略。

即使任务完成，action 的维度仍固定；变化的是“哪些位置当前有效”。

## 5. Reward 设计

当前每步奖励由以下部分组成：

1. 工作收益：
   - `+ work_done`
   - `work_done = gpu * effective_time`
2. 电费成本：
   - `- gpu * step_duration * curr_price`
3. 完成奖励（任务完成时）：
   - `+ finish_bonus_base`
   - `+ finish_bonus_workload_coef * initial_workload`
   - `+ finish_bonus_slack_coef * max(0, end_time - current_step)`
4. 截止惩罚：
   - 任务首次被检测为超期时扣 `-5`（每个任务只扣一次）
5. 容量违规惩罚：
   - `overflow = max(0, total_demanded - curr_cap)`
   - `overflow_ratio = overflow / max(curr_cap, 1e-6)`
   - 惩罚为 `constraint_penalty_coef * overflow_ratio`

`BatchJobEnv.__init__` 中默认参数：

- `constraint_penalty_coef = 1000.0`
- `finish_bonus_base = 5.0`
- `finish_bonus_workload_coef = 0.15`
- `finish_bonus_slack_coef = 0.1`

## 6. 启动与执行逻辑

- 若 `gpu == 0`：本步无工作量（`effective_time = 0`）。
- 若 `gpu > 0`：
  - 若上一步该 slot 分配为 `0`：
    - 该任务首次启动：本步 `effective_time = step_duration`
    - 重启场景：`effective_time = step_duration - startup_penalty_hr`（下限为 `0`）
  - 若上一步该 slot 分配 `>0`：`effective_time = step_duration`

## 7. Episode 结束条件

- `terminated` 当前始终为 `False`（尚未设置“任务全部完成即结束”）。
- 当 `current_step >= total_steps` 时，`truncated = True`。

## 8. 已确认的改动基线

以下行为已确认并作为当前基线：

1. State 使用当前步容量特征（不再使用下一步 Oracle 预测）。
2. 环境中 predictor 相关接口/注释已移除。
3. DDL 超期惩罚改为每任务只触发一次（`ddl_penalized` 标记）。
4. 容量违规惩罚由固定硬惩罚改为按超额比例惩罚。
5. 完成奖励由固定常数改为组合奖励。

## 9. 快速检查清单

- State 维度是否为 `max_slots*7 + history_window*2 + 2`
- Global 第二维是否为“当前步容量”
- Action 维度是否始终为 `2*max_slots`
- 超期惩罚是否每任务只触发一次
- 容量惩罚是否按违规比例生效
