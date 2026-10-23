from pathlib import Path
import csv
import numpy as np
import torch

from train_chat import Agent, Args
from act.make_env import make_eval_envs

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

out = Path("results/act_07_200steps_seed30.csv")

if out.exists():
    raise SystemExit(
        f"既存ファイルを保護するため停止: {out}"
    )

models = {
    "ACT-07_chunk30": (
        "runs/act_pickcube_train_07_xyz/"
        "checkpoints/5000.pt"
    ),
    "ACT-07_chunk10": (
        "runs/act_pickcube_train_07_xyz/"
        "checkpoints/5000.pt"
    ),
}

args = Args()
args.control_mode = "pd_ee_delta_pos"
args.num_queries = 30
args.max_episode_steps = 200
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
        max_episode_steps=200,
    ),
    other_kwargs=None,
    video_dir=None,
)

agents = {}

try:
    for name, path in models.items():
        agent = Agent(envs, args).to(device)

        ckpt = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )

        agent.load_state_dict(ckpt["ema_agent"])
        agent.eval()

        agents[name] = agent
        print("モデル読み込み:", name)

    def xyz(tensor):
        return (
            tensor.detach()
            .cpu()
            .numpy()
            .reshape(-1, 3)[0]
            .copy()
        )

    def flag(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return bool(np.asarray(value).reshape(-1)[0])

    def evaluate_one(name, seed):
        agent = agents[name]
        interval = 10 if name == 'ACT-07_chunk10' else 30

        obs, _ = envs.reset(seed=seed)
        initial_obs = np.asarray(obs).copy()

        base = envs.envs[0].unwrapped
        initial_z = xyz(base.cube.pose.p)[2]

        grasped_once = False
        max_lift = 0.0
        min_distance = float("inf")

        with torch.no_grad():
            for t in range(200):

                if t % interval == 0:
                    obs_tensor = torch.as_tensor(
                        obs,
                        dtype=torch.float32,
                        device=device,
                    )
                    action_seq = agent.get_action(
                        obs_tensor
                    )

                action = (
                    action_seq[:, t % interval]
                    .cpu()
                    .numpy()
                )

                obs, reward, terminated, truncated, info = (
                    envs.step(action)
                )

                cube = xyz(base.cube.pose.p)
                goal = xyz(base.goal_site.pose.p)

                distance = (
                    np.linalg.norm(cube - goal) * 1000
                )

                lift = (cube[2] - initial_z) * 1000

                min_distance = min(
                    min_distance, distance
                )

                max_lift = max(max_lift, lift)

                grasped_once |= flag(info["is_grasped"])

                if np.any(truncated):
                    break

        if "episode" not in info:
            raise RuntimeError(
                "エピソード終了情報がありません"
            )

        episode_info = info["episode"]

        result = {
            "seed": seed,
            "steps_run": t + 1,
            "model": name,
            "success_once": int(
                flag(episode_info["success_once"])
            ),
            "success_at_end": int(
                flag(episode_info["success_at_end"])
            ),
            "grasped_once": int(grasped_once),
            "max_lift_mm": round(max_lift, 2),
            "min_goal_distance_mm": round(
                min_distance, 2
            ),
        }

        return result, initial_obs

    results = []
    seeds = range(1000, 1030)

    print("\n=== 同一初期条件で比較開始 ===")

    for index, seed in enumerate(seeds, start=1):
        paired = {}
        initial_states = {}

        for name in models:
            result, initial_obs = evaluate_one(
                name, seed
            )

            paired[name] = result
            initial_states[name] = initial_obs

        names = list(models)

        if not np.allclose(
            initial_states[names[0]],
            initial_states[names[1]],
            rtol=1e-5,
            atol=1e-5,
        ):
            raise RuntimeError(
                f"seed={seed} の初期状態が一致しません"
            )

        results.extend(paired.values())

        a = paired[names[0]]
        b = paired[names[1]]

        print(
            f"{index:02d}/30 seed={seed}: "
            f"chunk30={a['success_at_end']} "
            f"chunk10={b['success_at_end']}"
        )

finally:
    envs.close()

out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys()),
    )
    writer.writeheader()
    writer.writerows(results)

print("\n=== 比較結果 ===")

for name in models:
    data = [
        r for r in results
        if r["model"] == name
    ]

    success = sum(
        r["success_at_end"] for r in data
    )

    grasp = sum(
        r["grasped_once"] for r in data
    )

    avg_distance = np.mean([
        r["min_goal_distance_mm"]
        for r in data
    ])

    print(f"\n{name}")
    print(f"成功: {success}/{len(data)}")
    print(f"把持: {grasp}/{len(data)}")
    print(f"平均最短目標距離: {avg_distance:.1f}mm")

print("\nCSV保存:", out)
print("比較終了")
