from pathlib import Path
import csv
import numpy as np
import torch

from train_chat import Agent, Args
from act.make_env import make_eval_envs

SEED = 2024
OUT = Path("results/act_08_seed2024_detail")

if OUT.exists():
    raise SystemExit(f"保存先が既に存在します: {OUT}")

OUT.mkdir(parents=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

args = Args()
args.control_mode = "pd_ee_delta_pos"
args.num_queries = 30
args.max_episode_steps = 150
args.temporal_agg = False

envs = make_eval_envs(
    "PickCube-v1",
    num_envs=1,
    sim_backend="physx_cpu",
    env_kwargs=dict(
        control_mode="pd_ee_delta_pos",
        reward_mode="sparse",
        obs_mode="state",
        render_mode="rgb_array",
        max_episode_steps=150,
    ),
    other_kwargs=None,
    video_dir=str(OUT / "videos"),
)

def flag(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])

def xyz(value):
    return (
        value.detach()
        .cpu()
        .numpy()
        .reshape(-1, 3)[0]
        .copy()
    )

rows = []

try:
    agent = Agent(envs, args).to(device)

    ckpt = torch.load(
        "runs/act_pickcube_train_08_phase/"
        "checkpoints/5000.pt",
        map_location=device,
        weights_only=True,
    )

    agent.load_state_dict(ckpt["ema_agent"])
    agent.eval()

    obs, _ = envs.reset(seed=SEED)
    base = envs.envs[0].unwrapped
    initial_z = xyz(base.cube.pose.p)[2]

    with torch.no_grad():
        for t in range(150):
            if t % 30 == 0:
                obs_tensor = torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                    device=device,
                )
                actions = agent.get_action(obs_tensor)

            action = actions[:, t % 30].cpu().numpy()

            obs, reward, terminated, truncated, info = (
                envs.step(action)
            )

            cube = xyz(base.cube.pose.p)
            goal = xyz(base.goal_site.pose.p)

            rows.append({
                "step": t + 1,
                "distance_mm": round(
                    float(np.linalg.norm(cube - goal) * 1000),
                    2,
                ),
                "lift_mm": round(
                    float((cube[2] - initial_z) * 1000),
                    2,
                ),
                "grasped": flag(info["is_grasped"]),
                "placed": flag(info["is_obj_placed"]),
                "static": flag(info["is_robot_static"]),
                "success": flag(info["success"]),
            })

            if np.any(truncated):
                break

finally:
    envs.close()

csv_path = OUT / "trace.csv"

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(rows)

success_rows = [r for r in rows if r["success"]]

print("\n=== seed=2024 詳細結果 ===")
print("実行ステップ数:", len(rows))
print("最大持ち上げ:", max(r["lift_mm"] for r in rows), "mm")
print("最短目標距離:", min(r["distance_mm"] for r in rows), "mm")

if success_rows:
    print("初回成功:", success_rows[0]["step"])
    print("最終成功:", success_rows[-1]["step"])
else:
    print("正式成功: なし")

print("終了時成功:", rows[-1]["success"])

print("\n=== 成功判定の切り替わり ===")

previous = False

for r in rows:
    if r["success"] != previous:
        print(
            f"Step {r['step']:3d}: "
            f"成功={r['success']}, "
            f"距離={r['distance_mm']:.1f}mm, "
            f"静止={r['static']}, "
            f"把持={r['grasped']}"
        )
    previous = r["success"]

print("\n=== 最後の15ステップ ===")

for r in rows[-15:]:
    print(
        f"Step {r['step']:3d}: "
        f"距離={r['distance_mm']:6.1f}mm "
        f"目標内={r['placed']} "
        f"静止={r['static']} "
        f"成功={r['success']}"
    )

print("\nCSV:", csv_path)
print("動画:", OUT / "videos")
