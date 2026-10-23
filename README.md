# PickCube模倣学習 — ACT（Physical AI 最終課題）

ManiSkill **3.0.1** の `PickCube-v1`（Franka Panda）で、Motion Planningから取得したデモを用い、ACT（Action Chunking Transformer）で状態観測から行動チャンクを予測する模倣学習を行いました。入力は **`state`**、制御は **`pd_ee_delta_pos`** であり、画像入力・実機検証は対象外です。

> **提出用公開資料 / 2026-10-23。** このリポジトリには学習・評価コードと保存済みCSVがありますが、デモ本体（HDF5/JSON）と学習済みモデル（`.pt`）は**含まれません**。それらの入手先がない状態で、第三者が同一チェックポイントを推論・学習再現できるとは主張しません。以下では、**実行済みの検証**と**未検証の手順例**を区別します。

## 実験環境（実機上で確認した値）

| 項目 | 確認値 |
|---|---|
| OS / GPU | Ubuntu 24.04 / NVIDIA GeForce RTX 5060 Laptop GPU（8GB） |
| Python | 3.11.16 |
| uv | 0.12.9 |
| PyTorch / CUDA | 2.7.1+cu128 / PyTorchビルドCUDA 12.8 |
| torchvision | 0.22.1+cu128 |
| ManiSkill / source commit | 3.0.1 / `a4a4f9272ad64b1564035874b605ceb687b63ed8` |
| NumPy / OpenCV | 1.26.4 / opencv-python 4.11.0.86 |
| diffusers / TensorBoard | 0.33.1 / 2.21.0 |
| その他 | 元の環境では `uv pip freeze` で **111件**を記録。全件ロックファイルは公開候補ZIPに未収録。 |
| 観測・制御 | `state` / `pd_ee_delta_pos` |
| 評価条件 | `physx_cpu`, 150ステップ, 30ステップごとにチャンク再生成、temporal aggregationなし |

`ManiSkill-src/examples/baselines/act/` はローカルのManiSkillソースから取得したACT基盤実装です。上記commitは2026-10-23に `git rev-parse HEAD` で確認し、同時に `git status --short` が空であることを確認しました。詳細は [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) を参照してください。

## 手法と主な結果

- **ACT-06 / 5,000反復**：標準のL1 + KL。歴史的学習コマンドが欠けており、`train_chat.py` と `train_chat_v2.py` のどちらを用いたか未確定。
- **ACT-07 / 5,000反復**：各行動チャネル X/Y/Z/Grip のL1損失を `(5,5,2,1)` で重み付けし平均する。KL項は別途加算。**L1とKLの相対尺度も変わる**ため、成分重みだけの効果は識別できない。
- **ACT-08 / 4,000・5,000反復**：ACT-07の損失を維持し、各デモで `0.35 <= t/(L-1) < 0.70` に該当する**行動チャンク開始点**へ、通常側の **2.5倍の抽出重み**を設定。`WeightedRandomSampler(replacement=True)` による復元抽出。**重点区間全体の抽出確率が単純に2.5倍**という意味ではない。

| モデル | 共通30条件（1000–1029）終了時 | 別の50条件（2000–2049）終了時 |
|---|---:|---:|
| ACT-06 5k | 0/30 (0%) | 未評価 |
| ACT-07 5k | 3/30 (10.0%) | 3/50 (6.0%) |
| ACT-08 4k | 13/30 (43.3%) | 17/50 (34.0%) |
| **ACT-08 5k** | **13/30 (43.3%)** | **18/50 (36.0%)** |

**主指標**は150ステップ終了時の `success_at_end`、**補助指標**は一度でも成功した `success_once` です。ACT-08 5kの50条件では、一度でも成功 **38/50 (76%)**、終了時成功 **18/50 (36%)**。一度成功した38条件のうち**20条件が終了時には成功状態を失いました**。これは到達と維持を区別すべきことを示しますが、この集計だけで原因は特定できません。50条件は4k/5kのモデル選択にも使用しており、未使用の独立最終テストではありません。学習乱数を変えた反復訓練は未実施です。

## ファイルの所在

```text
train_chat.py / train_chat_v2.py  # ACT-06候補（確定前）
train_chat_xyz.py                 # ACT-07 学習コード
train_chat_act08.py               # ACT-08 学習コード
evaluate_chat.py                  # ACT関連の評価ユーティリティ
compare_act_*.py / inspect_*.py  # seed別評価・詳細トレース
ManiSkill-src/examples/baselines/act/  # 派生元ACTの基盤コード
reference_results/              # 比較CSV、成功トレース、学習ログ
tools/summarize_results.py       # 保存済みCSVを再集計（標準ライブラリのみ）
docs/                            # 再現テストの実績・実行例・公開前チェック
requirements-core.txt            # 最小依存の構築例（完全lockfileではない）
```

### A. 【実行確認済み】結果CSVの再集計

GitHub公開候補ZIPを元の実験フォルダとは別フォルダに展開し、2026-10-23に以下を実行して成功しています。

```bash
cd <REPOSITORY_ROOT>
python3 tools/summarize_results.py
```

この処理は**保存済みCSVの再集計**であり、新規シミュレーションは実行しません。

### B. 【実行確認済み】既存環境＋既存モデルによるseed=2024推論

`inspect_act08_seed2024.py` のモデル参照先は以下です。公開ZIPにモデルはなく、元の実験PCに保存された学習済みモデルを**シンボリックリンクで参照**して実行しました。

```text
runs/act_pickcube_train_08_phase/checkpoints/5000.pt
```

Python/依存パッケージを導入済みで、**適合するACT-08 5kチェックポイントがこの相対パスに配置されている場合のみ**、次のコマンドが対象です。実行すると `results/act_08_seed2024_detail/` に新規CSVと動画を書き込みます。既に同名フォルダがあればスクリプトは安全のため停止します。

```bash
cd <REPOSITORY_ROOT>
export PYTHONPATH="$PWD/ManiSkill-src/examples/baselines/act:$PWD"
python inspect_act08_seed2024.py
```

2026-10-23のローカル検証では、**最大持ち上げ188.14mm、最短目標距離5.81mm、Step 68で初回成功、Step 150終了時成功**が元の記録と表示桁まで一致しました。トレースの成功フラグはStep 68–150の**83記録点（両端含む）で連続して真**でした。`results/` は新規生成物であり、同梱済みの比較CSVは `reference_results/` に保存しています。

この検証は**既存の仮想環境と既存モデルを使った同一PCでの再実行**です。新しいPCでの依存構築や、公開ZIP単体での推論成功までは確認していません。詳しい実施記録は [`docs/REPRO_VERIFICATION_20261023.md`](docs/REPRO_VERIFICATION_20261023.md) を参照してください。

### C. 【未実行の参考手順】新規環境構築

このコマンドは**動作検証していない構築例**です。PyTorchとCUDA、OS、GPUの組合せを適合させてください。`pip` ではなく `uv` を用いた環境でも問題ありません。**現在の学習用仮想環境を上書きしないこと。**

```bash
# Python 3.11が利用可能なシステムで、新規cloneフォルダ内から
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements-core.txt
export PYTHONPATH="$PWD/ManiSkill-src/examples/baselines/act:$PWD"
```

完全な依存ロックファイルがないため、今後のライブラリ更新によって追加調整が必要になる可能性があります。

### D. 【未実行の参考手順】教師データ生成・再学習

学習ログで**100軌道の読み込み**を確認していますが、今回の資料レビューではHDF5/JSON本体の全軌道successを照合していません。したがって本文では「Motion Planningデモ100軌道（学習ログで読込確認）」と表記します。

- **生成**：ManiSkill 3.0.1のMotion Planningサンプルを使い、PickCube-v1の軌道を収録する。サンプルのCLI使用例は [`docs/RUN_COMMANDS_AND_GAPS.md`](docs/RUN_COMMANDS_AND_GAPS.md) に記載（当時の完全な履歴コマンドではありません）。
- **変換**：ManiSkill 3.0.1のtrajectory replayで `obs_mode=state`、`target_control_mode=pd_ee_delta_pos`、`sim_backend=physx_cpu` に変換する。**元の100軌道生成・変換コマンドは未照合**です。
- **学習**：変換済み `.h5` と同名の `.json` を `data/PickCube-v1/motionplanning/` に配置する。スクリプトのtyro引数を確認した上で、`train_chat_xyz.py` または `train_chat_act08.py` を使用する。コマンド例は [`docs/RUN_COMMANDS_AND_GAPS.md`](docs/RUN_COMMANDS_AND_GAPS.md) に記載。**元の学習を実行した正確な引数と乱数seedは未確定**です。

### E. 【チェックポイントが必要】50条件評価

ACT-07 5k、ACT-08 4k/5kのモデルを配置した上で、次の評価スクリプトを使用します。4kの評価には `best_eval_success_at_end.pt` を使用しています。**学習ログと学習コードの照合では、4,000反復時の評価でこのbestが更新され、その後の5,000反復時には更新されていません。** さらに2026-10-23にユーザーのUbuntuで両ファイルを読み取り専用で照合し、`ema_agent`の構成・全テンソルが完全一致、`norm_stats`はいずれも`None`と確認しました。**チェックポイント全体のバイト一致を意味するものではありません。**

```bash
export PYTHONPATH="$PWD/ManiSkill-src/examples/baselines/act:$PWD"
python compare_act_07_08_new50seeds.py
```

`runs/` のモデルファイルは別途準備が必要です。CSVは `results/` に新規生成され、既存の同名ファイルがあれば停止する設計です。**この50条件比較スクリプトを公開候補ZIPから再実行したことはまだありません**。CSV集計とseed=2024の個別推論を実測済みとするにとどめます。


### F. 【実行確認済み】ACT-08の4,000反復チェックポイントとの照合

2026-10-23にユーザーのUbuntuで、同じ実験の `4000.pt` と
`best_eval_success_at_end.pt` を読み取り専用でロードして比較しました。
比較結果は **EMAパラメータ構成一致：True／EMA全重み完全一致：True／
`norm_stats`：両方 `None`** でした（総合判定PASS）。
これに加え、学習ログの4,000反復時のbest更新記録と、学習コードの保存処理は整合しています。
**両チェックポイントの全保存内容・バイト列が一致するとは主張しません。**
また、実行ログはユーザー提供のもので、チェックポイント本体やSHA-256値はこの公開候補ZIPに含まれません。

再照合する場合は、**信頼できる自分の学習済みファイルのみ**を対象に次を実行します（ZIPにはチェックポイント本体はありません）。

```bash
export PYTHONPATH="$PWD/ManiSkill-src/examples/baselines/act:$PWD"
python tools/verify_act08_4k_checkpoint.py \
  --checkpoints ~/pai_final/chat_branch/runs/act_pickcube_train_08_phase/checkpoints
```

上記の照合コマンド自体は公開候補ZIPからの再実行は未確認であり、ユーザーが実行した同等の比較スクリプトによる結果を記録しています。

## 制約と今後の検証

1. 複数学習seedの反復比較は未実施。性能差の因果効果は断定しません。
2. 50条件はモデル選択にも使ったため、完全に独立なホールドアウトではありません。
3. 成功維持が課題。成功一次判定後の脱落20条件について失敗形態は未分類です。
4. 画像入力・実機・異なるシミュレータへの転移は未検証です。
5. 100デモ本体の全件成功とハッシュ、チェックポイントのSHA-256、元の学習起動コマンド、別PCでの環境構築は未照合です。4k/bestのEMA重みと`norm_stats`の比較は同一PCで完了しています。
6. ManiSkill v3.0.1内部の成功判定式の細部は未レビュー。`success_at_end` から物体の解放まで保証されるとは解釈しません。

## 公開動画・レポート・出典

- 提出動画：**[PAI最終課題_142463](https://youtu.be/kdLdKnrBnG4)**。最終採用は `PAI_ACT_final_video_v4_teacher_wording_20261023.mp4`（ユーザーが内容・表示を確認済み）。v3は提出対象外です。Motion Planningの例、ACT-06/07/08比較、ACT-08代表成功例を収録します。動画はユーザーがアップロードしました。Workでは公開動画の再生・尺・公開設定を未確認です。
- PDF：omnicampusへ別途提出（GitHubでの公開は必須条件でない）。
- ManiSkill: https://github.com/haosulab/ManiSkill
- ACT baseline: https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/act
- ACT原論文: Zhao et al., RSS 2023, https://doi.org/10.15607/RSS.2023.XIX.016
- ManiSkill3: Tao et al., https://arxiv.org/abs/2410.00425
- 派生元実装の著作権・ライセンス情報：[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)

公開前に [`docs/PUBLISH_CHECKLIST.md`](docs/PUBLISH_CHECKLIST.md) を完了してください。会社のコード・CAD・設定は使用・公開しない方針です。
