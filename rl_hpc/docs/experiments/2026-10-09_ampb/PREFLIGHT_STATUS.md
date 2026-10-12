# ampb 再現実験：実行前確認

指定コミット `26e9bbb5e227c41f8d161ed31764c373ce119a0d` を取得してdetached HEADでcheckoutし、AMPB_REPRODUCTION_PROMPT.mdを全文読了。実験コードと設定は変更していない。学習・評価・restart生成・buildはまだ実行していない。

環境：ampb.g2.gsic.titech.ac.jp、Ubuntu 26.04.1、Xeon 6972P×2、192物理core/384 logical CPUs、6 NUMA nodes、1.5 TiB RAM、GCC 15.2.0、CMake 4.2.3。確認プロセスのCPU affinityは0–383。確認時にMPI/LAMMPS測定の残留processは見つからなかった。これだけで32 coreの排他的予約が成立したとは判断しない。

## 停止理由

1. 指定buildスクリプトは `cmake --fresh` を使用し、OPENMPとSHOCKだけを有効化する。既存buildはさらにEXTRA-PAIRとEXTRA-COMPUTEを有効化しており、既存binaryのGit infoは3be966cff1。Shock入力には `pair_style lj/cubic` と `compute mymom all momentum` があり、それぞれsrc/EXTRA-PAIRとsrc/EXTRA-COMPUTEに実装されている。指定スクリプトをそのまま実行すると既存build設定を置き換え、必要packageが欠ける。既存buildを保護すること、およびpackage不足時は独断で変更しないという手順に従いbuild前に保留した。

2. sandbox外でも `/usr/bin/mpirun --version` はバージョンを表示せず、以下を出力した。終了コードは0なので、終了コードだけでは正常性を判断できない。MPI実験は開始していない。

```text
Sorry! You were supposed to get help about:
    version
But I couldn't open the help file:
    /usr/lib/x86_64-linux-gnu/pmix2/share/pmix/help-schizo-ompi.txt: No such file or directory
    /usr/lib/x86_64-linux-gnu/prrte3/share/prte/help-schizo-ompi.txt: No such file or directory.
```

## その他の確認

今回指定のseed restart directoryとidle/contention学習directoryはまだ存在しない。以前のstate×action実験を48 episode学習の代わりには使わない。既定prepare_seed_restarts.pyは元seed87287を先頭に加え、train-count=4により学習87287/17287/27287/37287、評価47287/57287に分割する。指定のseed分離と一致する。prepare_heavy.pyのnx既定値は240だが、今回指定nx480を明示する必要がある。

今回のCPU負荷は既存RLの2–5 decision ON/OFF schedule。以前の同一checkpoint action実験の「500 stepsの間ずっとactive」とは異なる。今回は指定されたRL scheduleを使用する計画であり、以前の負荷条件を流用しない。

学習実施数0/96 episode、評価実施数0/56 episode。unsafe transitionの有無は未実施なので評価していない。生データ・restart・binary・checkpoint・logはGitへ追加していない。

## 承認後の環境復旧・build（2026-10-09）

ユーザー承認により、既存buildを`build_shock_char_preserved_3be966cff1`に保存。新buildは`build_shock_rl_ampb`、既存スクリプトの固定参照`build_shock_char`は新buildへのsymlinkに切り替えた。実験スクリプトは変更していない。

MPIは正常にversionを表示する既存`/opt/openmpi`のOpen MPI 5.0.10をPATH/LD_LIBRARY_PATHで選択。MPI32 `/bin/true`はreturn code 0、socket0の32 distinct physical coresへbind。設定はRelease、MPI/OMP/OPENMP/SHOCK ON、`-march=native`、Unix Makefiles、build parallel16。承認された追加packageはEXTRA-PAIR/EXTRA-COMPUTE。MPI include/libraryは`/opt/openmpi-5.0.10.cuda.gcc`。binary SHA-256は`800599e36d1865f5ce5ff8ead25629e93e09ede25b7726b78f881159970647b0`。`lmp -h`確認済み。

上記2 packageとMPIインストール選択が手順記載からの明示的な環境差。MD条件・学習経験量・方策・評価scheduleは変更しない。prepare_heavyには既定nx240の代わりに指定nx480を明示。初期restart生成開始。

## 実行開始

6 restartの全SHA-256、return code 0、491,520 atoms、step1200、数値finite、simulation errorなしを確認。学習4 seedと評価2 seedの分離も確認。MPI/LAMMPS残留なしを確認して、既存スクリプトを逐次呼び出すshell command列をバックグラウンド起動した（PID2441645、parent1）。command列と全logはGit ignoreされた`rl_hpc/characterization/data/ampb_rl_official_run_v1/`に保存。shellは`set -euo pipefail`で失敗時に後続を実行しない。

実行順：既存run_balance_hybrid_study.py（idle48→cpu_jitter48）、RL評価16episode、公式4方式32episode、neighbor0.8評価8episode、既存analyze_rl_vs_official.py。評価はいずれもseed20261008、test restart list、120decisions、2repeats、idle/jitter、MPI32/OMP1。実験コードの追加・修正なし。

学習開始直後のidle episode000では最初の6 transitionsを確認し、すべてsafe、dangerous build0、491,520 atoms、step増分500だった。全96学習episode・56評価episodeはまだ完了していない。最終日本語レポートは全測定の検証後に作成する。既存解析スクリプトはPNG/compact CSVを出力するが、日本語Markdownは自動生成しない。


## 最終完了

学習96 episode、評価56 episode、全18,240 transitionsの検証完了。描画固定範囲による解析エラーはユーザー承認の範囲内で表示のみ修正し、再解析return code0。PNG・compact CSV・日本語レポートを保存。追加実験なし。元のpipeline_exit_code.txt=1は初回描画失敗の履歴として保持。
