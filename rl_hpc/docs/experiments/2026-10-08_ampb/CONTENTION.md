[監督付き再測定v3の完了結果](CONTENTION_V3.md)（各5回、main480試行）。以下は先行実行の記録。

## 再開済み（2026-10-09）

ユーザーの再開指示に従い、独立バックグラウンド監督PID2324958でmain_v3を再開。安全な283件を保持し、元のseed20261008の順序で残り197件を実行する。起動確認時点で287/480件がsafeで完了。VSCode extensionHostの一時的活動は記録し、5 check（約10秒）累積2.5 CPU秒超で停止する。その他の外部CPU負荷・GPU計算の監視とunsafe時の即停止、240秒の進行停止guardは維持する。

旧manifest/status/launchと中断試行を保存し、runner SHAと監視方針の変更だけをresume_auditへ記録した。原子数、checkpoint、MPI/OMP、skin、action、500 steps、計測入力とランダム順は変更していない。日をまたいだ時間変動を含むことは最終報告へ記載する。最新の進捗は[status.json](../../../characterization/data/portable_balance_contention_background_v3/status.json)を参照。完了後に自動集計する。以下は先行状況の記録。

## バックグラウンドv3の確認結果（2026-10-09）

smoke12/12は安全に完了。mainは283/480（58.96%）が完了し、完了した283件は全てsafe。反復0/1は各96件、反復2は91件。2026-10-08 21:13:46 JSTに外部負荷監視で停止した。

停止対象は別ユーザーのVSCode extensionHostプロセス（PID2046135、name MainThread）で、約2秒間のCPU時間増分0.510秒が監視閾値0.5秒を超えた。前回のCPUベンチマークを検出したのではない。この短い活動の実際の測定影響は未評価で、simulationの物理的unsafeが検出された停止ではない。中断試行はphase-08-factor-0.5-r2-idle。各条件5反復は未完了、自動集計には到達していない。

確認時点でMPI/LAMMPSと実験runnerの残留はない。元データと監視証拠は保持している。

## バックグラウンド再実行v3（起動済み）

ユーザーの指示により、独立sessionの監督プロセスPID2136627で再実行を起動した。stdout/stderrはファイルへ保存し、terminal/tool sessionの接続から切り離した。smoke12試行が安全ならmain480試行（8 states × 6 actions × idle/contention × 5 repetitions）を同じseed20261008のランダム順で順次実行し、その後自動集計する。unsafe、外部負荷検出、240秒の進行停止で後続実行を止め、自分のプロセス階層を終了する。

[進捗status.json](../../../characterization/data/portable_balance_contention_background_v3/status.json)、[監督ログ](../../../characterization/data/portable_balance_contention_background_v3/supervisor.log)。mainデータ保存先は `rl_hpc/characterization/data/portable_balance_contention_main_v3/`。完了時に `CONTENTION_V3.md` と全5回matrixを生成する。前回の不完全なデータとは混ぜない。以下は先行実行の記録。

# 同一checkpointからのCPU contention比較

2026-10-08、ユーザーの追加承認に基づく実験。元のportable protocolのno-contention条件からの明示的な拡張であり、元の再現実験とは区別する。

## 実験条件

既存8 checkpoint × 6 actions（skip、weight neigh 0.50 / 0.75 / 1.00 / 1.25 / 1.50）× idle/contention × 5反復。各反復内の96条件をseed 20261008でshuffleし、480個の独立MPI実行を順次実行する。smokeは序盤・終盤 × skip / 0.50 / 1.50 × 2負荷条件の12試行で、本集計に含めない。

MPI32、OMP1、491,520原子、skin0.5、500 steps。各試行は同じ保存checkpointを読み、計測外でweight neigh1.0の共通partitionを再構成する。CPU負荷は再構成時にはinactiveである。contention条件では計測開始前にactivateし、既存RL側と同じ0.15秒の待機後に計測を始める。計測中はcontinuous active。idle条件ではworkerを生成しない。

CPU負荷には既存 `rl_hpc/characterization/shock/cpu_jitter.py` の `CpuJitter` を変更せずに使う。CPU0–7それぞれに1個の既存整数演算workerをpinする。MPI ranksは32個の異なる物理coreへbindし、mpirunのnooversubscribeを維持する。CPU0–7でco-runnerとMPIの物理core共有を意図的に発生させる。MPIのSMT siblingへの移動を含む既存core-bindingを変更しない。workerのCPU時間、MPI配置、worker終了を記録する。

これはRL側の2–5 decisionごとのON/OFFスケジュールを評価する実験ではなく、同一checkpoint/actionにおける負荷条件付きの次状態・報酬比較である。新しい連続軌跡上での訪問頻度や、学習方策の性能を測るものではない。

## 保存先と実行

- runner: `rl_hpc/characterization/shock/fork_balance_contention.py`（既存fork入力、MPI Simulation、RL CpuJitterを再利用する追加runner）
- smoke: `rl_hpc/characterization/data/portable_balance_contention_smoke_v1/`
- main: `rl_hpc/characterization/data/portable_balance_contention_main_v1/`

```bash
python3 rl_hpc/characterization/shock/fork_balance_contention.py \
  --checkpoints rl_hpc/characterization/data/portable_representative_checkpoints_v1 \
  --output rl_hpc/characterization/data/portable_balance_contention_main_v1 \
  --phase main --repetitions 5 --seed 20261008 --cpu-cores 0-7
```

manifestに順序、seed、checkpoint manifest・binary・runner・既存fork・co-runnerのSHA-256を保存する。return code、原子数、step+500、dangerous build、NaN/Inf、simulation error、MPI配置を確認し、unsafeが1件でも出たら停止する。生成データはGitへ追加しない。

## 比較する量

状態/actionごとに両条件の5測定を比較する。次状態のatom/neigh/ghost rank max/meanによる不均衡を、Pair/Neigh/Comm時間を含む実行状態と分けて集計する。報酬はaction準備・balance・500 stepsの合計wall secondsの負値。mean、標本std、CV、個々の測定値を示す。

同じ条件を均等に混合した集計に加え、各checkpoint/actionでの比較を主結果とする。5測定では分布の裾や小さな差の有無を精密に評価できない。反復番号をブロックとしてruntime差を計算するが、二条件のプロセスは独立であり、同じ瞬間のhost noiseを共有する厳密な同時ペアではない。

## 再測定v2の進捗（停止）

監視付きsmokeは12/12安全に完了。新しいmain_v2は180/480試行（37.5%）が完了し、全180件はsafe・return code0。第1反復96条件と第2反復84条件までが保存されている。各条件5反復は未完了で、v2の最終集計はまだ実施していない。

次の `phase-02-factor-0.5-r1-contention` がrestart/common partition再構成段階で進まなくなった。logはstep10300、原子491,520のrun0で止まり、計測用500-step segment、result.json、main_complete.jsonはない。runnerとMPIが長時間残留し、実行tool sessionも利用できなくなっていた。原因は未特定で、tool接続失効が原因であるとは断定しない。監視ログ最終時刻は2026-10-08T18:53:33.707690+09:00、その時点までの613チェックで外部負荷検出はゼロ。以後の期間を監視済みとは扱わない。

自分のrunnerとその子孫に限定して終了させ、残留MPI/LAMMPS/負荷workerなしを確認した。`STALLED.json`、`stalled_process_snapshot.json`、準備段階logを保存した。以下に掲載する数値結果は**前回v1の完全な4反復の暫定集計**であり、v2の5反復結果ではない。

## 実行結果（v1・4反復の暫定結果）

**計画した5反復は未完了。** smoke 12/12、main 413/413の完了試行はすべてsimulation上safeでreturn code 0。mainの414番目は計測前に手動停止し、runner return codeは130、中断したMPIのreturn codeは1（500-step計測未実行）。最後の不完全な反復を集計から全て除き、最初の4反復＝384試行（各条件4回、idle/contention各192）を暫定集計した。

別ユーザーのCPUベンチマーク開始時刻は 2026-10-08T17:05:00.100000+09:00。第4反復までの最終完了時刻は 2026-10-08T17:03:22.521817+09:00。この外部プロセスの存続期間と重なる完了試行は17件で、全て第5反復にある。CPU affinityは128–159と320–351で、MPIのCPU0–31/192–223および意図したworkerのCPU0–7とは直接重ならない。専用ホストであることを確認できなくなったため停止した。runtimeの変化をこの外部プロセスの因果効果とは断定しない。別socketでも共有資源等への影響はこのデータだけでは切り分けられない。

既知の外部プロセスは第4反復終了後に起動している。ただしホスト全体の負荷を全測定期間連続監視したわけではなく、過去の他負荷が完全にゼロだったことを事後に証明するものではない。相手のプロセスには手を触れていない。停止後に残留MPI/LAMMPSなし、8 checkpointのSHA-256不変を確認した。

[完全な暫定matrix（各4測定値、mean/std/CV、次状態・報酬）](CONTENTION_MATRIX_FIRST4.md)。元データとmanifestは `rl_hpc/characterization/data/portable_balance_contention_main_v1/` に保持。第5反復の29完了試行は削除せず保存した。

### 次状態と報酬の違い

同じcheckpoint/actionに属する8測定（idle4＋contention4）について、atom/neigh/ghost rank max/meanの範囲は全48組で0。したがって、今回測定した不均衡のみの条件付き次状態の経験分布はログ精度で一致した。一方、時間を含む状態と報酬の経験分布には差がある。これは同一checkpoint/actionからの500-step遷移に関する結果であり、連続軌跡の訪問状態分布を評価したものではない。

|量|idle mean|contention mean|変化|
|---|---:|---:|---:|
|runtime_seconds|5.150978215|5.531517151|+7.388%|
|reward|-5.150978215|-5.531517151|-7.388%|
|pair_per_step|0.004708486|0.005036378|+6.964%|
|neigh_per_step|0.002295375|0.002445028|+6.520%|
|comm_per_step|0.000563531|0.000687658|+22.027%|

runtime平均は48条件の均等混合。混合分布のstd/CVには状態・action間の違いも入るため、測定ノイズとして解釈しない。次表のCVは状態ごとの反復内CVを8状態で算術平均したもの。

|Action|idle runtime mean (s)|contention mean (s)|増加率|idle 状態内CVの平均 %|contention 状態内CVの平均 %|
|---|---:|---:|---:|---:|---:|
|skip|4.956251|5.263158|+6.192%|0.198|0.845|
|0.50|4.511942|5.596180|+24.030%|0.814|0.470|
|0.75|4.539968|5.154655|+13.539%|0.240|0.655|
|1.00|5.058525|5.306872|+4.909%|0.255|0.692|
|1.25|5.663775|5.689420|+0.453%|0.162|0.174|
|1.50|6.175408|6.178819|+0.055%|0.091|0.219|


### Action順位（同じデータ上の記述値）

|State|idle mean-best|runner-up gap (s)|contention mean-best|runner-up gap (s)|
|---|---|---:|---|---:|
|S1|skip|0.074043|1.25|0.026207|
|S2|skip|0.250934|skip|0.026257|
|S3|0.75|0.116324|skip|0.058046|
|S4|0.50|0.124922|skip|0.051269|
|S5|0.50|0.379582|1.00|0.069654|
|S6|0.75|0.079434|0.75|0.107824|
|S7|0.50|0.463391|0.75|0.256926|
|S8|0.50|0.634051|0.75|0.092290|

平均上のbest actionは8状態中6状態で変わった。ただしS1/S3/S4/S5等の負荷ありbestとrunner-upの小さい差を、確定した最適actionとは扱わない。標本が4回と少なく、同じデータでactionを選択している。matrixの95% CIは反復ブロック差に対するdf3のt区間で、48比較の多重性や選択を補正した保証ではない。

|Regime|Best Fixed|8状態meanの合計 (s)|state-conditioned oracle (s)|差 (s)|記述的削減率|
|---|---|---:|---:|---:|---:|
|idle|0.50|36.095534|33.907795|2.187739|6.061%|
|contention|0.75|41.237240|39.339184|1.898057|4.603%|

oracleは同じデータから選択・評価した記述値。学習方策の性能や不偏な汎化性能ではない。以前のno-contention 5反復ではBest Fixed 0.50、oracle削減率6.076%であり、今回のidle4反復でも0.50、6.061%と近い。一方、元amp2との純粋な速度比較はcheckpoint選択とホストが違うため行わない。

### Action reversalと観測されたばらつき

以下は固定した2 action間のruntime差を4測定の各負荷条件で比較したもの。差は後者−前者で、正なら前者が速い。CIは無補正・df3の反復ブロックt区間。

|State|前者 / 後者|idle difference mean [95% CI] (s)|contention difference mean [95% CI] (s)|
|---|---|---|---|
|S1|skip / 1.25|+0.236150 [+0.222818, +0.249482]|-0.426669 [-0.456133, -0.397206]|
|S3|0.75 / skip|+0.116324 [+0.101229, +0.131419]|-0.178689 [-0.261428, -0.095951]|
|S4|0.50 / skip|+0.265421 [+0.244818, +0.286025]|-0.501652 [-0.613440, -0.389864]|
|S5|0.50 / 1.00|+0.824984 [+0.745940, +0.904028]|-0.089975 [-0.134006, -0.045943]|
|S7|0.50 / 0.75|+0.463391 [+0.404481, +0.522301]|-0.256926 [-0.447860, -0.065992]|
|S8|0.50 / 0.75|+0.634051 [+0.610272, +0.657830]|-0.092290 [-0.114191, -0.070389]|

これらの対比では順位反転が反復内の揺れを超える例が観測される。ただし4反復の記述的検証であり、5反復完了・専用利用確認後の再検証を要する。負荷ありの細かいrunner-up順位まで安定と主張しない。

### 不均衡と時間を含む状態のPCA

不均衡3特徴PCAのPC1/PC2はidle分散の64.995%/32.833%（計97.828%）。不均衡＋Pair/Neigh/Comm per-stepの6特徴では58.639%/26.019%（計84.658%）。標準化とPCAはidle4反復にfitし、contentionも同じ軸へ投影する。PCAの符号は任意で、以前の図と符号が逆でも意味の違いではない。青の中空点がidle、橙がcontention。不均衡のみでは点が重なる。時間を含む場合はずれる組がある。



不均衡のみの状態入力では、この実験のCPU負荷条件を識別する情報が得られていない。一方、報酬とactionの優劣は負荷条件で変わる。時間を含む特徴には負荷に関する情報が現れる可能性があるが、識別器やRL方策の学習・評価は実施していない。

### Physical scalarsと安全性

同じcheckpoint/actionのidle4＋contention4で観測した、physical scalarの最大範囲は次の通り。物理状態のbitwise一致までは主張しない。

|量|48組内での最大範囲|
|---|---:|
|temperature|0|
|pressure|0|
|density|0|
|total_energy_per_atom|0|
|shock_position|0|

全完了試行について原子491,520、step+500、dangerous builds0、finite thermo/timers、simulation errorなしを検証。smokeとfirst4それぞれで計測前thermoおよびreconstructed.meshのSHAが状態ごとに一致。負荷workerは全完了試行でreapされ、計測中のCPU時間合計は約8 CPU秒/wall秒。

### 再集計と未完了事項

```bash
MPLCONFIGDIR=/tmp/ampb-contention-matplotlib python3 rl_hpc/characterization/shock/analyze_balance_contention.py \
  --input rl_hpc/characterization/data/portable_balance_contention_main_v1 \
  --output rl_hpc/characterization/data/portable_balance_contention_main_v1/analysis_first4 \
  --repetitions 4
```

各条件5回を完了するには専用利用を再確認した上で再測定する。現在の出力ディレクトリには中断された試行があるため、無条件にresumeしない。元データ、停止時のhost snapshot、外部負荷の開始時刻/affinity、STOPPED.jsonを保存した。生成データ・restart・log・binaryはGitへ追加していない。

binary SHA-256: `e5a6191be7c0ab164d794651950f5577a6433451b79274224171c841b21682b4`。測定時checkout: `bae96b699e96efbdbe48facaaac452f2ee0f071c`。build条件は[既存ampb報告](README.md)を参照。元のfork・RL CpuJitterを変更せず、承認された追加runnerを使用した。

main measurements SHA-256: `89a7ae8ce7e711ad95cebf23013e47e26db96f54e61c96c725ef9ee9848763aa`。解析対象はその中のrepetition0–3のみ。
