# ampb GPU contention実験：中断時点の進捗

中断日：2026-10-14（Asia/Tokyo）。ユーザーの明示指示によりamp2対応状態でのmedium pilotを停止した。今後の測定は自動継続しない。

## 確定した条件と成果

- ampb: RTX PRO 6000 Blackwell Max-Q ×4（compute capability 12.0）、MPI32、OMP1、491,520 atoms、skin0.5。KOKKOS/CUDA、MPS、8 ranks/GPU。CUDA13.1.115、Open MPI5.0.10。GPU binary SHA-256 `f6ba2128095c017c5b9d9e9b35a3004084193b462d0c053b988c66d72a00c24f`。
- 初期restart seed47287はGPU-nativeで生成し、returncode0、安全性検査を通過。manifestのgit commitは`dfd37932149c4383263d00eb84fde852d08a8532`。その後、amp2報告の追加commitを取得し、軌跡・checkpoint・pilotのmanifestは`e3d3a52dd043c917b117664224760aa45a727d3d`。実行バイナリのSHAは固定。
- `collect_gpu_unbalanced_states.py`で動的balanceなしの120 decisions × 3軌跡＝360状態を採取。3 episodeともstep61300に到達。
- amp2の高不均衡状態と同じcheckpoint step `1300,2300,7300,12300,24300,31300,41800,51300` をampb上で8個保存。SHA-256全件照合、dangerous build0。保存したmeshの31領域境界を、forkの`--preserve-source-partition`で再構成する。最初のS1 probe後はatom不均衡1.21185、neighbor不均衡1.86360で、amp2報告の1.212/1.868と近い。
- amp2対応状態のlight pilot（64 MiB、8 repeats、sleep 1000 µs）は60/60試行がsafe、returncode0、dangerous0。idleとの対応slowdown中央値+6.77%で、手順書の目標25–100%未達。

## 中断時に走っていた試行

amp2対応状態のmedium pilot（128 MiB、32 repeats、sleep 0 µs、seed20261012）を意図的に停止。`amp2-state-pilot-medium/measurements.jsonl`には**完了済み6試行**（CPU contention4、GPU contention2）が記録され、6件すべてsafe、returncode0、dangerous0。`runs/`には7件目の中断directoryもある。medium pilotの60件は未完了であり、校正判定には使用しない。終了コード143はユーザー指示による停止で、unsafeとして扱わない。保存済みpartial dataとログは削除していない。

当初開始した別の8 checkpoint（step5300以降）のmainは7試行完了時に停止し、別状態の部分データとして保全。amp2対応結果には混ぜない。

MPS control daemonは停止済み。停止確認時にMPI/LAMMPS/MPS残留processなし。生データ、restart、binary、checkpoint、巨大logはGitへ追加していない。Git作業ツリーには互換性修正2ファイルとこの進捗文書が未commitで残る。

## 再開時の手順

1. 指定GPU、GPU compute process、32+4物理core、MPI/MPS、build binary SHA-256、8 checkpoint SHA-256を再確認する。
2. medium pilotは、部分反復を保持してresumeせず、**新しいoutput directoryから60試行を最初から**実行する。`rl_hpc/characterization/data/ampb_gpu_contention_v1/imbalance_medium_commands.sh`の既存スクリプト呼び出しと`--preserve-source-partition`を参照し、出力先だけ新規にする。完了済み6件は監査用に保持する。
3. 完全な60試行の安全性、4 GPUの活動、対応slowdown中央値とpaired 95%区間、power/thermal cap、memoryを評価し、強度をmain前に固定する。lightよりmediumが高すぎる場合は手順書に従い1パラメータずつ調整する。
4. 校正済み強度でamp2対応8 states × 6 actions × idle/CPU/GPU contention × 5 repetitions＝720試行を、保存した分割を復元して逐次測定する。unsafeが1件でもあれば停止する。
5. 完全な反復だけで解析し、日本語レポート・図・compact summaryを生成する。amp2比較は相対変化とaction順位を中心とする。

詳細な経緯は[GPU_CONTENTION_STATUS.md](GPU_CONTENTION_STATUS.md)。元の実行条件は[GPU_CONTENTION_REPRODUCTION.md](GPU_CONTENTION_REPRODUCTION.md)。
