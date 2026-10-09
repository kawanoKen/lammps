# ampb向け実行プロンプト：RL対公式balance方式

以下をampb上のエージェントへそのまま渡す。実験コードを新規実装させず、
Git管理済みスクリプトだけでamp2と同じ比較を再現するための指示である。

```text
既にclone済みのLAMMPS repositoryで、branch `rl-hpc-research` を最新化し、
Shock/NEMDのRL対公式balance方式をampb上で再現してください。

新しい実験ロジックを実装してはいけません。Git管理済みの以下をそのまま使ってください。
- rl_hpc/characterization/shock/build_shock_cpu.sh
- prepare_heavy.py / prepare_seed_restarts.py
- run_balance_hybrid_study.py
- evaluate_balance_hybrid.py
- evaluate_official_balance.py
- analyze_rl_vs_official.py

条件を変更してはいけません。
- workload: Shock/NEMD, nx=480, ny=nz=16, 491,520 atoms
- MPI 32 ranks, OpenMP 1 thread/rank, oversubscribe禁止
- skin=0.5, 500 MD steps/decision, 120 decisions/episode
- action: skip、または balance weight neigh factor in [0.5,1.5]
- 学習: idle 48 episodeとCPU contention 48 episodeをampb上で別々に実施
- 各学習は20 exploration episode + 28 online-FQI episode
- CPU contentionはcores 0-7、jitter duration 2--5 decisions
- 評価中は方策を凍結し、更新・探索をしない
- 評価seedは学習seedから完全に分離する
- 評価は物理seed 47287/57287 × 2反復 × idle/contention
- schedule seed=20261008
- 比較方策はRL-idle、RL-contention、official atoms、neighbor 0.8、
  neighbor 1.0、neighbor 1.5、time 1.0
- すべて逐次実行し、同時に複数のLAMMPS測定を走らせない

まずOS、CPU、NUMA、メモリ、compiler、CMake、MPI、Git commitを記録してください。
32物理coreを利用できなければ停止してください。既存buildを壊さず、
`BUILD_JOBS=16 bash rl_hpc/characterization/shock/build_shock_cpu.sh` で専用buildを確認してください。

`rl_hpc/characterization/data/shock_online_seed_restarts_v1/` が完全でなければ、
seed 87287の491,520-atom初期restartをprepare_heavy.pyで作り、それを入力に
prepare_seed_restarts.pyを既定seed列・train-count=4で実行してください。
train/test restart listのseed分離、atom数、SHA-256、return codeを確認してください。

次に `python3 -u rl_hpc/characterization/shock/run_balance_hybrid_study.py`
を実行し、idle/CPU contention各48 episode、各5,760 transitions、dangerous build 0、
nonzero exit 0を確認してください。中断時は既存のresume機能だけを使ってください。

学習後、次の3評価を同じtest restart listとseed=20261008で実行してください。
1. evaluate_balance_hybrid.py:
   output=`rl_hpc/characterization/data/shock_balance_rl_eval_matched_official_v2/run`
   policies=`idle_trained jitter_trained`
2. evaluate_official_balance.py:
   output=`rl_hpc/characterization/data/shock_official_balance_eval_main_v2`
   policies=`official_atoms official_neigh_10 official_neigh_15 official_time_10`
3. evaluate_official_balance.py:
   output=`rl_hpc/characterization/data/shock_official_neigh_08_eval_v1`
   policies=`official_neigh_08`

3評価はいずれも decisions=120、repeats=2、regimes=`idle jitter`、
expected-atoms=491520、cpu-cores=0-7、seed=20261008としてください。
完了後、次を実行してください。
`python3 rl_hpc/characterization/shock/analyze_rl_vs_official.py --output-dir rl_hpc/docs/experiments/2026-10-09_ampb`

途中でpackage不足、unsafe transition、dangerous build、NaN/Inf、atom数不一致、
MPI error、残留processが発生したら停止し、設定やコードを独断で変更しないでください。
生データ、restart、binary、log、checkpointはcommitしないでください。

最終報告には、環境/build、学習96 episodeの完全性、評価56 episodeの完全性、
各regimeの7方策mean/std/CV/95% CI、paired差、順位、RL対最良公式方式の短縮率、
RLのbalance率/factor分布、安全性、amp2との差を記載してください。
報告先は `rl_hpc/docs/experiments/YYYY-MM-DD_ampb/RL_VS_OFFICIAL_HEURISTICS.md`
とし、PNGとcompact CSVのみGit管理対象にしてください。
```
