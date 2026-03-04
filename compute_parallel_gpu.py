import pandas as pd
import numpy as np


def compute_parallel_gpu(
    input_csv="RL/interactive.csv",
    output_csv="RL/result.csv",
    start_time="2025-10-15 00:00:00",
    freq="5min",
    duration_threshold_seconds=150,
):
    try:
        df = pd.read_csv(input_csv, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(input_csv, encoding="gbk")
    if "start_time" not in df.columns or "end_time" not in df.columns:
        raise ValueError("interactive.csv must include 'start_time' and 'end_time' timestamp columns.")

    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
    df = df.dropna(subset=["start_time", "end_time"])
    df["duration_seconds"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
    df = df[df["duration_seconds"] >= duration_threshold_seconds]

    start_ts = pd.Timestamp(start_time)
    end_ts = df["end_time"].max()
    if pd.isna(end_ts) or end_ts < start_ts:
        raise ValueError("end_time in interactive.csv must be >= start_time.")

    time_index = pd.date_range(start=start_ts, end=end_ts, freq=freq)

    if df.empty:
        out = pd.DataFrame({"时间": time_index, "parallel_gpu": np.zeros(len(time_index), dtype=int)})
        out.to_csv(output_csv, index=False)
        return out

    starts_sec = (df["start_time"].astype("int64") // 10**9).to_numpy()
    ends_sec = (df["end_time"].astype("int64") // 10**9).to_numpy()
    starts_sec.sort()
    ends_sec.sort()

    counts = []
    for ts in (time_index.view(np.int64) // 10**9):
        started = np.searchsorted(starts_sec, ts, side="right")
        ended = np.searchsorted(ends_sec, ts, side="right")
        counts.append(int(started - ended))

    out = pd.DataFrame({"时间": time_index, "parallel_gpu": counts})
    out.to_csv(output_csv, index=False)
    return out


if __name__ == "__main__":
    compute_parallel_gpu()
