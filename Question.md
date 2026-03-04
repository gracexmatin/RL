2. set_env_data 的性能瓶颈
问题：目前的 set_env_data 会根据 env_data 的全量时间范围（all_min 到 all_max），按 5 分钟粒度重新计算一遍并发量。

场景：如果你的 env_data 有几万条历史数据，且你在 RL 训练的每一步（step）都调用 predict(env_data=df)，这会触发全量重算。
结合 predictor.py 的实现来看，set_env_data 会：
- 重新排序和时间解析（开始/结束时间 -> 时间戳）；
- 重新生成全量 5 分钟时间索引；
- 在 for 循环里遍历每个时间点并用 searchsorted + 统计 ends 计算并发量；
- 重新生成 lookup_series 并做一次 scaler transform。
因此在训练步里反复调用 predict(env_data=...) 会把这些重计算都做一遍，导致明显的性能瓶颈。

后果：训练或推理速度会变得非常慢（可能会卡死在 set_env_data 的那个循环里）。

建议：
如果只在每个 Episode 开始前加载一次数据，那没问题。
如果是实时流式预测，建议不要每次都重建整个 ResourcePredictor 或者重置整个 data，而是只 append 新数据（但这需要改写 set_env_data 的逻辑，目前先凑合用，只要你不觉得慢）。


记录TODO
将计算资源这一操作提前进行，而使得predictor和environment都可以查表
