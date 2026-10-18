# 2026-10-18 amp2：CPU contention × balance factor反実仮想実験

## 目的

同一のShock/NEMD内部状態からidle/L4とbalance actionへ分岐し、CPU contentionによって
最適actionが変わるかを直接測る。同時に、action選択前に取得可能なCPU/application telemetryが
contentionを識別し、action選択に有用な状態量となるかを検証する。

## 条件

| 項目 | 設定 |
|:---|:---|
| workload | Shock/NEMD、491,520 atoms |
| MPI / OpenMP | 32 ranks / 1 thread |
| skin | 0.5固定 |
| checkpoint | 既存の代表8状態 |
| contention | idle、CPU 0–7上のL4（8 workers、100% duty） |
| action | skip、factor 0.50–1.50（0.125刻み） |
| 反復 | 各cell 5回、反復block内をrandomize |
| reward区間 | balance適用時間 + 500 MD steps |

## 1試行の流れ

```text
restart読込
→ factor 1.0で共通partition再構成（timer外）
→ idleまたはL4を開始
→ factor 1.0の共通500-step probe
→ action前state/telemetryを保存
→ skipまたはfactor action
→ 500 stepsを計測
→ next state、reward、診断telemetryを保存
→ process/co-runnerを完全終了
```

probeは全actionに共通で、そこで得た情報だけをaction前stateとして扱う。contention labelと
co-runner CPU時間は解析専用であり、将来のpolicy入力候補には含めない。

## 保存するaction前観測

- LAMMPS：温度、圧力、atom/neighbor/ghost imbalance、Pair/Neigh/Comm/Modify時間、
  neighbor build回数
- CPU：対象物理coreのutilization mean/max/std、LAMMPS CPU entitlement
- MPI rank：CPU efficiencyとvoluntary/involuntary context-switch rateのmean/min/max/std
- 診断：対象物理coreのbusy時間、対応LAMMPS rank CPU時間、その差による外部busy推定

## 規模と安全性

smokeは2状態 × 4 actions × 2 regimes = 16試行。本収集は
8 × 10 × 2 × 5 = 800試行。smokeでdangerous build、step不一致、MPI配置、co-runner稼働、
telemetry分離を検査し、不合格なら本収集を開始しない。開始前にclean-baseline guardを通し、
実行中はhost process monitorを保存する。

生成データはgit管理外の`rl_hpc/characterization/data/`以下へ保存する。
