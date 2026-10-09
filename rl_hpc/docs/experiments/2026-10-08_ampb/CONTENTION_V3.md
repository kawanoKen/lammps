# CPU contention比較：監督付きバックグラウンド再測定v3

VSCodeの一時的CPU活動による監視停止後、ユーザーの再開指示に従って安全な283試行を保持し、残り197試行を元のランダム順で継続した。旧manifest・中断ログとresume_auditを保持。計測条件の変更はなく、監視とrunnerのSHAのみ履歴として記録する。VSCode extensionHostは約10秒の累積2.5 CPU秒超で停止、それ以外の外部ユーザープロセスは従来の0.5 CPU秒/checkで停止する。日をまたいだ測定の時間変動を含むため、細かい順位差は慎重に解釈する。

同じ8 checkpoint × 6 actions × idle/contention × 各5回。各反復の96条件をseed20261008でshuffleし、MPI32/OMP1、skin0.5、500 stepsを順次実行した。共通weight neigh1.0のpartition再構成は計測外、contentionは既存RL CpuJitterのCPU0–7の8 workersを計測中のみactiveにした。

smoke12試行とmain480試行が安全に完了。試行の安全性・外部負荷監視・checkpoint SHAはrunnerで確認し、停止時は後続段階を実行しない。新しい監督プロセスは240秒進行なしでも停止する。元のfork、RL負荷実装と実験条件は変更していない。

[指定形式のcontention図・idle比較図（6枚）](CONTENTION_PLOTS.md)。

[全5測定値・mean/std/CV・報酬・次状態の完全matrix](CONTENTION_MATRIX_V3.md)。CIは同じ反復ブロック差の無補正df4 t区間であり、48比較を同時に保証しない。Pooled分布は状態/action間の違いを含む。

|量|idle mean|contention mean|変化 %|
|---|---:|---:|---:|
|runtime_seconds|5.15232489|5.53660662|+7.458|
|reward|-5.15232489|-5.53660662|-7.458|
|atom_imbalance|2.03185628|2.03185628|+0.000|
|neighbor_imbalance|1.2821594|1.2821594|+0.000|
|ghost_imbalance|1.30868506|1.30868506|+0.000|
|pair_per_step|0.00470985167|0.00503859167|+6.980|
|neigh_per_step|0.002295555|0.00244700992|+6.598|
|comm_per_step|0.0005659255|0.000688717833|+21.698|

|State|idle mean-best|runner-up gap s|contention mean-best|runner-up gap s|
|---|---|---:|---|---:|
|S1|skip|0.072733|1.25|0.019748|
|S2|skip|0.257646|skip|0.023069|
|S3|0.75|0.110789|skip|0.051797|
|S4|0.50|0.130429|1.00|0.019744|
|S5|0.50|0.387129|skip|0.029648|
|S6|0.75|0.129720|0.75|0.077481|
|S7|0.50|0.470889|0.75|0.233567|
|S8|0.50|0.639284|0.75|0.117946|

|Regime|Best Fixed|8-state mean sum s|Oracle s|差 s|削減率 %|
|---|---|---:|---:|---:|---:|
|idle|0.50|36.104589|33.907643|2.196947|6.085|
|contention|0.75|41.418305|39.407018|2.011286|4.856|

oracleは同じデータから選択・評価した記述値。学習方策の性能や不偏な汎化性能ではない。各条件5回では分布の裾や小さい差を精密に評価できない。今回の比較は同一checkpoint/actionからの条件付き次状態であり、連続軌跡の訪問状態分布ではない。




環境・正確なbuild設定は[ampb報告](README.md)を参照。先行する中断データは混ぜずに保存。v3元データは `rl_hpc/characterization/data/portable_balance_contention_main_v3/`、監督statusは `rl_hpc/characterization/data/portable_balance_contention_background_v3/status.json`。生成データ等をGitへ追加していない。
