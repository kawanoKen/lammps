# GPU Shock/NEMD + CUDA MPS preflight（amp2）

実施日: 2026-10-12。既存の `build_rl/`、`build_shock_char/`、
`build_kokkos_cuda/` は変更していない。

## 構成

- GPU: NVIDIA GeForce RTX 3090 x 4
- CUDA toolkit: 13.4.59
- NVIDIA driver: 615.71.09
- MPI: Open MPI 5.0.10、CUDA-aware MPIなし
- 新規build: `build_shock_kokkos_cuda/`
- packages: KOKKOS、SHOCK、EXTRA-PAIR、EXTRA-COMPUTE
- Kokkos backends/architectures: CUDA + SERIAL、AMPERE86 + SKX
- 実行設定: MPI 4 ranks、GPU 4枚、OpenMP 1 thread/rank、
  `-k on g 4 -sf kk -pk kokkos gpu/aware off`

再現buildは `rl_hpc/characterization/shock/build_shock_kokkos_cuda.sh`、
短時間入力は `rl_hpc/characterization/shock/in.gpu_smoke` を使用する。

## CUDA MPS

MPS control/serverは導入済みだった。ユーザー専用directoryを使って起動した。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  rl_hpc/characterization/shock/mps_control.sh start
```

状態確認と停止は、それぞれ `status` と `stop` を指定する。MPSはGPUを高速化する
一般的な設定ではなく、複数processが同一GPUを共有するcontention実験で利用する。
4 ranks / 4 GPUsの通常実行では各rankが別GPUを使うため、MPSは必須ではない。

## 実測結果

491,520原子の既存step-1200 checkpointから、CPUとGPUを同じ入力で100 steps進めた。

| backend | ranks / GPUs | LAMMPS loop time | dangerous builds |
|---|---:|---:|---:|
| CPU | 4 / 0 | 5.0225 s | 0 |
| KOKKOS/CUDA | 4 / 4 | 0.796775 s | 0 |

step 1300のCPU/GPU比較では、原子数と密度は一致した。温度は
166.6562322対166.6558036、全エネルギーは249.3509528対249.3503037、圧力は
127.8430164対127.8426883だった。相対差はいずれも約0.0003%以下であり、この短い
smokeでは大きな数値的不整合はない。ただしbitwise一致や長時間の物理同値性を証明する
ものではない。

追加の2,000-step GPU実行は15.9813 s、dangerous builds 0で完了した。実行中の4 GPUは
すべて利用され、GPU utilizationは33--41%、powerは約118--152 Wだった。MPSの`ps`でも
4個のLAMMPS clientが互いに異なるGPUへ割り当てられていることを確認した。

## 注意点

- binary restartはMPI 32で作成され、smokeはMPI 4で読んだため、rank数変更の警告は想定内。
- KOKKOSはrestart時のNewton設定差について警告する。短時間比較は正常終了したが、
  本学習前にはGPU上でbaseline trajectory/restartを作り直す方が比較設計として明確。
- `gpu/aware off`を固定する。現環境で`on`をactionに含めない。
- GPU contention学習ではtargetとco-runnerを同一GPUへ載せる場合にだけMPSを使い、
  client数、GPU mapping、power、clock、temperatureを必ず記録する。

## Shock action reward pilot

同日、GPU-nativeな8 checkpoint × 6 balance actions × 3 regimes × 5反復の
720測定を完了した。MPI32、4 GPU、8 ranks/GPU、MPS共有で、各GPUに1つずつ
medium CUDA contender（配列ごと128 MiB、repeats=32、sleep=0 us）を置いた。
全720測定がsafeで、unsafeは0だった。

| Regime | 500-step action runtime mean | n |
|---|---:|---:|
| Idle | 2.0234 s | 240 |
| CPU contention | 2.0340 s | 240 |
| Direct GPU contention | 3.6601 s | 240 |

GPU contention下ではidleに対して平均約80.9%遅くなり、報酬
`r=-action_wall_seconds`は明瞭に低下した。CPU contention平均はidleとほぼ同じ
だった。これは全state/actionを平均した記述値であり、GPU機種をまたぐ強度の
移植やaction ranking reversalを単独で証明する結果ではない。生ログは
ignored `characterization/data/shock_gpu_state_action_main_v1/` に保持する。

ampbでは同じ固定medium強度を仮定せず、低強度から5反復pilotを行い、報酬の
paired差とGPU telemetryから強度を決める手順を
[GPU contention再現指示](../2026-10-12_ampb/GPU_CONTENTION_REPRODUCTION.md)
に記載した。
