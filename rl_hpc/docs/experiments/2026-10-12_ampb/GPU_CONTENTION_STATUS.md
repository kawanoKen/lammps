# GPU contention実行状況（2026-10-12）

手順書を全文読み、research branchをfetch/ff-only更新。実行commitはdfd37932149c4383263d00eb84fde852d08a8532。独立GPU build成功、GPU-native初期生成の安全性チェックで停止。pilot/mainは未実施。実験スクリプトは未修正。

## 環境

ampb、4枚のRTX PRO6000 Blackwell Max-Q、compute capability12.0、開始時各97,340MiB free、GPU compute processなし。専用ホスト利用の既存許可を使用し、scheduler-assigned CUDA maskなし。CUDA13.1.115、OpenMPI5.0.10（/opt/openmpi）、GCC15.2、Kokkos BLACKWELL120、contender sm_120。MPI32 physical cores0–31、contender CPUs32/33/96/97。GPU0/1はNUMA0、GPU2/3はNUMA3。GPU0/1のcontender CPUはMPIのNUMA0 coreを避けた同socketの隣接NUMA1、GPU2/3はNUMA3。

既存build scriptをBUILD_JOBS16で実行し、build_shock_kokkos_cuda/lmpのSHOCK、lj/cubic/kk、wall/piston/kk、momentum確認成功。CPU buildは変更していない。生のpreflight・build log・binary SHAはGit ignoreされたcharacterization/data/ampb_gpu_contention_v1に保存。

MPSは既存スクリプトが環境変数を上書きするため、未使用だったユーザー専用/tmp/kawano-lammps-mps-pipeと-logに統一。今回起動したdaemonだけを停止済み。停止確認時、MPI/LAMMPS残留0。

## 停止理由

prepare_gpu_initial.pyは全logにNaN/Inf正規表現を適用する。ログにechoされた合法なregionコマンドのINF境界を誤検出し、manifest safe=false、スクリプトexit1になる。マッチは以下の3行だけ。

```text
region bulk block ${bulklo} ${bulkhi} INF INF INF INF units box
region bulk block 5 ${bulkhi} INF INF INF INF units box
region bulk block 5 825.262769414726 INF INF INF INF units box
```

LAMMPS自体はreturncode0、stderr空、ERROR/Lost atomsなし、checkpoint step1200、有限temp166.8022728492/press127.2438365871/density0.7071067812/xshock12.0616482506。restart SHA-256は02ac71ea8e4169b3a83116e80712cc8e37d3de78a5041c9789c032a4925e4206。初期生成は既存protocolのcheck=noなのでdangerous builds not checkedであり、0と解釈しない。

安全性判定を無視して後続を走らせず停止。修正案は入力echoの合法なregion境界INFと、数値出力のNaN/Infを区別すること。ユーザー承認前に実験スクリプトを変更しない。初期生成の生結果を保全し、8checkpoint・校正pilot60試行・main720試行は未実施。Gitへの生データ/restart/binary/log追加なし。


## 承認後の再開

ユーザー承認によりprepare_gpu_initial.pyのNaN/Inf検査のみ修正。echoされたregion blockの6境界引数にある合法なINFだけを除外し、数値NaN/Inf・ERRORは引き続き検出する。元logと人工的なNaN/Inf例で検査を確認。旧初期結果をinitial-seed47287-before-inf-fixに保存し、指定条件で再生成。初期生成safe=true、returncode0。PID3971722で初期生成→restart spec→8 checkpoint→light pilot60試行を逐次バックグラウンド実行。失敗時は後続を停止し、今回のMPSだけを終了する。現在checkpoint生成中。main720試行は校正後に強度を選ぶため未開始。


## 2026-10-13確認：pilot開始前のbuild停止

8 checkpointの生成とSHA-256照合は完了。light pilotはCUDA contenderのcompileで停止し、計測は0/60試行。CUDA13.1と現行system headersのrsqrt/rsqrtf exception specificationが競合しgpu_factorial.pyの既存build_gpu_contenderがRuntimeErrorを出した。pilot pipeline exit1。実験コードは修正していない。今回のMPSはtrapで停止済み。残留MPI/LAMMPS/MPSなし。mainは未開始。

## コンパイル互換性修正後のpilot再開

ユーザー承認によりgpu_factorial.pyのCUDA contender compile commandに`-Xcompiler -U_GNU_SOURCE`だけを追加。glibcのGNU数学関数宣言とCUDA13.1のrsqrt/rsqrtf宣言の競合を回避。CUDA kernelとintensity条件は未変更。既存builderのsm_120ビルドと短時間GPU実行がexit0で通過した。

計測0件だった旧pilot directoryはpilot-light-before-compile-fixに保全し、同じ8 checkpointsからlight pilot（64MiB,8 repeats,1000us sleep、seed20261012、5反復、60試行）をPID3989547で逐次バックグラウンド再開。最初の試行はsafe=true。失敗時は後続を停止し、今回のMPSのみ終了する。mainは校正解析後に実施予定で未開始。

## 2026-10-13 light pilot完了、medium校正開始

light pilotは60/60 safe、全returncode0、dangerous build0、errorなし。4つの(phase,action)セルでのGPU対idle平均slowdownはS1 skip +9.044%、S1 factor1.5 +4.982%、S8 skip +6.827%、S8 factor1.5 +4.447%。対応20組のslowdown中央値は+6.805%。各セルの対応差95%区間のうちfactor1.5の2セルは0を跨ぐ。手順書の25–100%目標未達なので、この強度をmainには採用しない。GPU0–3すべてで利用率の非ゼロ最大値を記録し、最大温度45°C、最大power148.1W、最大使用memory6466MiB。pilot終了後MPS停止、残留MPI/LAMMPSなし。

次段階として手順書指定のmedium強度（128MiB、32 repeats、sleep0）で同じ60試行pilotをバックグラウンド起動。PID4165533。mainは校正後まで保留。

## medium校正とmain開始

medium pilotも60/60 safe、returncode0、dangerous0、原子数/step異常なし。GPU/idle対応slowdown中央値+91.883%。S1 skip +97.72%、S1 factor1.5 +78.90%、S8 skip +97.55%、S8 factor1.5 +88.96%。全4セルのpaired 95%差区間は0を上回る。各GPUで非ゼロ利用率を記録。温度max52°C、memory max6588MiB、power max296.36W（limit300W）。pilot前後のSW Power Capping累積カウンタは全GPUで不変、thermal slowdown未検出。

手順書の校正基準に従い128MiB・32repeats・sleep0をmain前に固定し、calibration_selection.jsonに記録。8checkpoint SHA-256再照合済み。main720試行をPID42695で逐次バックグラウンド起動。新しいaction winning結果を使って強度選択していない。完了後に安全性・統計を解析する。

## amp2高不均衡状態への切り替え

ユーザーの指摘で`2026-10-13_amp2/RESULTS.md`を確認。先に開始したampb mainの8 checkpoint stepは5300,11800,20300,30800,35800,45300,47800,58300で、amp2の高不均衡状態と異なる。7試行が安全に完了した時点でrunnerを停止し、旧rawを別状態の部分データとして保全。MPI/LAMMPS/MPS残留0。旧mainの終了コード143は意図的停止を示し、unsafe試行として混ぜない。

amp2のcheckpoint step1300,2300,7300,12300,24300,31300,41800,51300（decision0,2,12,22,46,60,81,100）を固定選択した。既存`collect_gpu_unbalanced_states.py`で動的balanceなしの120 decision×3軌跡を採取し、選択スクリプトのampb候補も比較用に生成。その後、既存`materialize_imbalance_checkpoints_gpu.py`でamp2と同じstepの8 checkpointを再生成する。forkには`--preserve-source-partition`を使い、保存したmeshの31境界を復元する。旧mainはこの保存分割を使っておらず、同一状態比較には不適切。

PID46186で上記を順次バックグラウンド実行中。新しい高不均衡状態ではpilotから負荷強度を再確認してからmainを始める。先の校正mediumは別状態上の結果なのでそのまま流用しない。

## 2026-10-14：amp2対応状態の準備完了

動的balanceなし120decisions×3軌跡は全360 observations完了、3episodeともstep61300へ到達。amp2と同じcheckpoint steps 1300/2300/7300/12300/24300/31300/41800/51300を保存し、8 SHA-256検証、dangerous0、policy=no dynamic balance。保存したinitial.meshの分割をforkで`--preserve-source-partition`により再構成する。ampb独自の不均衡選択候補も別に作ったが、amp2との対応比較には混ぜない。

amp2対応stateのlight pilot60試行をPID567709で逐次バックグラウンド実行中。最初のS1試行はsafe=true、probe後step1800、atom不均衡1.21185、neighbor不均衡1.86360。amp2報告のS1 action開始step1800、atom1.212、neighbor1.868と近い。新状態群でGPU負荷強度を再校正してから720試行mainを開始する。

## 2026-10-14：amp2対応状態のlight校正結果

amp2対応8checkpointのうちpilotはS1/S8、skip/factor1.5で実施。light pilot60/60 safe、rc0、dangerous0。GPU/idle対応slowdownはS1 skip+7.24%、S1 factor1.5+9.16%、S8 skip+5.06%、S8 factor1.5+7.04%、中央値+6.77%。全セルのpaired 95%差区間は0超だが手順書目標25–100%未達。max power141.77W、max温度44°C。medium（128MiB、32repeats、sleep0）pilot60試行をPID285453で逐次起動。保存元MPI分割を明示復元する`--preserve-source-partition`を使用。mainは校正後まで未開始。
