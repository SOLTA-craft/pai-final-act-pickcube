from pathlib import Path
from datetime import datetime
import csv
import gymnasium as gym
import numpy as np
import torch

from train_chat import Agent, Args
from act.make_env import make_eval_envs
from evaluate_chat import evaluate


class MotionTrace(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.rows = []
        self.episode_id = 0
        self.step_id = 0
        self.initial_z = None

    def reset(self, **kwargs):
        self.step_id = 0
        self.initial_z = None
        return super().reset(**kwargs)

    def step(self, action):
        base = self.unwrapped

        if self.initial_z is None:
            self.initial_z = float(
                base.cube.pose.p[0, 2].item()
            )

        obs, reward, terminated, truncated, info = (
            super().step(action)
        )

        self.step_id += 1

        cube = base.cube.pose.p[0].detach().cpu().numpy()
        goal = base.goal_site.pose.p[0].detach().cpu().numpy()
        tcp = base.agent.tcp_pose.p[0].detach().cpu().numpy()

        def flag(key):
            return bool(np.asarray(info[key]).reshape(-1)[0])

        self.rows.append({
            "episode": self.episode_id,
            "step": self.step_id,
            "cube_x_mm": float(cube[0] * 1000),
            "cube_y_mm": float(cube[1] * 1000),
            "cube_z_mm": float(cube[2] * 1000),
            "goal_x_mm": float(goal[0] * 1000),
            "goal_y_mm": float(goal[1] * 1000),
            "goal_z_mm": float(goal[2] * 1000),
            "tcp_x_mm": float(tcp[0] * 1000),
            "tcp_y_mm": float(tcp[1] * 1000),
            "tcp_z_mm": float(tcp[2] * 1000),
            "lift_mm": float((cube[2] - self.initial_z) * 1000),
            "goal_distance_mm": float(
                np.linalg.norm(cube - goal) * 1000
            ),
            "grasped": flag("is_grasped"),
            "placed": flag("is_obj_placed"),
            "static": flag("is_robot_static"),
            "success": flag("success"),
        })

        if np.asarray(truncated).any():
            self.episode_id += 1

        return obs, reward, terminated, truncated, info


out = Path("results") / (
    "act_07_aligned_" + datetime.now().strftime("%Y%m%d_%H%M%S")
)
out.mkdir(parents=True, exist_ok=False)

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
    video_dir=str(out / "videos"),
    wrappers=[MotionTrace],
)

try:
    agent = Agent(envs, args).to(device)

    ckpt = torch.load(
        "runs/act_pickcube_train_07_xyz/checkpoints/"
        "best_eval_success_at_end.pt",
        map_location=device,
        weights_only=True,
    )
    agent.load_state_dict(ckpt["ema_agent"])

    metrics = evaluate(
        10,
        agent,
        envs,
        dict(
            stats=ckpt["norm_stats"],
            num_queries=30,
            temporal_agg=False,
            max_timesteps=150,
            device=device,
            sim_backend="physx_cpu",
        ),
    )

    wrapper = envs.envs[0]
    while not isinstance(wrapper, MotionTrace):
        wrapper = wrapper.env

    rows = list(wrapper.rows)

finally:
    envs.close()


with (out / "trace.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


print("\n=== エピソード別診断 ===")

for ep in range(10):
    data = [r for r in rows if r["episode"] == ep]
    if not data:
        continue

    max_lift = max(r["lift_mm"] for r in data)
    min_dist = min(r["goal_distance_mm"] for r in data)
    final_dist = data[-1]["goal_distance_mm"]

    grasp_steps = sum(r["grasped"] for r in data)

    longest_grasp = 0
    current = 0
    for r in data:
        current = current + 1 if r["grasped"] else 0
        longest_grasp = max(longest_grasp, current)

    success = any(r["success"] for r in data)

    print(
        f"Episode {ep}: "
        f"持上={max_lift:.1f}mm, "
        f"最短目標距離={min_dist:.1f}mm, "
        f"終了距離={final_dist:.1f}mm, "
        f"把持={grasp_steps}step, "
        f"連続把持={longest_grasp}step, "
        f"成功={success}"
    )

print("\n保存先:", out)
print("CSV:", out / "trace.csv")
print("動画:", out / "videos")
