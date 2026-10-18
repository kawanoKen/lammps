# LAMMPSにおけるMPI rank単位の制御可能性

## 結論

現在のShock/NEMD CPU実験は、MPI 32 ranksを `32 x 1 x 1` のprocessor gridとして
x方向に分割している。この条件では、標準LAMMPSの `balance` コマンドで31個の
x方向cut位置を数値指定することで、各rankの担当区間幅を直接制御できる。
LAMMPS本体の変更は不要である。

さらに、remote側の先行研究コード
`rl_hpc/characterization/shock/rank_target_balance.py` に、この標準機能を使った
32-rank目標ベクトル制御が既に実装されている。各rankへ正のcapacityを与え、atomの
x座標分布から目標累積atom数に対応する31 cutを求め、`balance 0.0 x ...` を発行する。
したがって、実装可能性だけでなく、ampbでは安全性と動作が実測済みである。

ただし、これは各rankが独立にローカル設定を変更する機能ではない。全rankが参加する
collectiveな領域再分割であり、31個の境界から32個の連続区間を定義する。粒子移送、
通信構造の再設定、Pair/KSpace gridのresetを伴うため、action costを含めて評価する
必要がある。

## 1. 担当領域を直接指定する既存実装

processor gridが `P_x x P_y x P_z` のとき、次の形式で各軸のcutを指定できる。

```lammps
balance 1.0 x c1 c2 ... c31
```

MPI32 Shockの `P_x=32, P_y=P_z=1` では、

```text
0 = c0 < c1 < ... < c31 < c32 = 1
```

として、rank `i` のx方向担当幅は

```text
w_i = c_(i+1) - c_i
```

となる。実座標での幅はbox長を `L_x` として `L_x w_i` である。したがって、
31 cutの変更は、この実験では実質的に32 ranksの領域幅制御になる。

実装経路は次の通りである。

- `Balance::command()` が `x/y/z` の数値cutを読み込む
  (`src/balance.cpp`, lines 115--221)。
- 指定値を `comm->xsplit`, `ysplit`, `zsplit` に格納する
  (`src/balance.cpp`, lines 284--320)。
- `Domain::set_local_box()` で各rankのsubdomainを更新する。
- `Irregular::migrate_atoms()` で粒子を新しい所有rankへ移す。
- `modify->reset_grid()` と `pair->reset_grid()` で分散gridを更新する
  (`src/balance.cpp`, lines 337--363)。

現在のpersistent-process型runnerから、decision interval間に通常のLAMMPS commandとして
発行できる。`sublo/subhi` のlibrary pointerを直接書き換える方法は、split情報、粒子移送、
neighbor/communication再構築が同期しないため使用すべきではない。

### 制約と安全性

- cutは0から1の間で厳密な昇順でなければならない。
- 各subdomainが狭すぎると、ghost通信や粒子移送が不正になる可能性がある。
- `fix balance` の動的経路には `subbox_too_small_check(neighbor->skin)` があるが、
  数値cutを直接与える `balance x ...` の経路には同じ明示的checkがない。
- controller側で最小幅を制約し、少なくともneighbor skinとの整合性を検証する必要がある。
- action適用は無料ではない。MPI barrier、LAMMPS再初期化、粒子移送、grid resetを含む。

安全なaction表現として、無制約なcut座標を直接出力するのではなく、正の区間幅
`w_i` を出力し、最小幅を加えた後で和が1になるよう正規化し、累積和からcutを作る方がよい。

## 2. rankごとの負荷を利用する既存実装

標準LAMMPSには、rankごとの観測値を内部で重みに変換する機能もある。ただし、controllerが
32個のrank weightを直接指定するinterfaceではなく、全rank共通の `factor` で重みの差を
拡大・縮小する方式である。

### `weight neigh factor`

rank `p` のowned atom数を `N_p`、neighbor総数を `G_p` とすると、基礎重みは概ね

```text
q_p = G_p / N_p
```

であり、そのrankが所有する全atomへ同じ値を与える。`factor` が1以外なら、rank間の
`q_p` の範囲を線形に拡大または縮小する。実装は
`ImbalanceNeigh::compute()` (`src/imbalance_neigh.cpp`, lines 45--97) にある。

これはrank別neighbor負荷を認識するが、actionは1個のglobal factorである。また、
KOKKOS GPU実行では明示的にskipされる。

### `weight time factor`

rank `p` の前回以降の時間を

```text
C_p = T_pair,p + T_neigh,p + T_bond,p + T_kspace,p
```

として、基礎重みを

```text
q_p = C_p / N_p
```

とする。実装は `ImbalanceTime::compute()`
(`src/imbalance_time.cpp`, lines 55--104) にある。このため、特定rankがCPU contentionで
遅くなれば、その差を自動的に領域分割へ反映できる可能性がある。

ただし、`Comm` や `Modify` の時間はこの重みに含まれず、actionもglobal factorだけである。
また、正しい計測windowを確保せずtimerをresetすると有効なtime weightにならない。

### `weight var`

`ImbalanceVar::compute()` はatom-style variableの値を各atomの重みとして利用できる
(`src/imbalance_var.cpp`)。空間領域やatom group由来の任意重みには使えるが、標準入力だけで
永続的な「rank IDごとの任意weight」を与える専用actionではない。粒子はbalance後にrank間を
移動するため、rankラベルをatom属性として固定する設計も目的と一致しない。

## 3. 何がrank単位で可能か

| 制御対象 | 標準実装 | 現在のShock/MPI32での意味 |
|:--|:--|:--|
| x方向の各rank担当幅 | 可能: `balance ... x <31 cuts>` | 32 ranksの領域幅を直接制御可能 |
| rank別負荷に応じた自動cut | 可能: `shift x` + `weight neigh/time` | rank別計測を使うがactionはglobal factor |
| 各rankへ任意の負荷weightを直接入力 | 専用interfaceなし | 必要なら小さなsource拡張候補 |
| `skin` / `neigh_modify every`をrank別設定 | 不可 | neighbor設定はsimulation全体で共通 |
| processor-grid内のrank配置 | 初期化時のみ可能: `processors ... map/grid custom` | episode-level設定。実行中actionではない |
| 任意形状のrank領域 | `comm_style tiled` + RCBで可能 | RCBが形状を決め、controllerの直接座標actionではない |

## 4. 現研究への適合性

rank単位の明示制御を試すなら、現在の `32 x 1 x 1` Shockは適している。shockはx方向へ
進行し、cutの順序とrankの順序が一対一だからである。候補stateはrankごとの
`Nlocal`, neighbor数, ghost数, Pair/Neigh/Comm時間、CPU contentionであり、actionは
x区間幅またはその低次元補正になる。

一方で、31次元のcut actionをそのまま学習させるのは探索空間が大きく、順序・最小幅制約も
強い。最初の検証では次の順序が妥当である。

1. 同じcheckpointから、少数の手設計cut perturbationを分岐して安全性とaction costを測る。
2. cut変更後の `Nlocal/neighbor/ghost` とPair/Comm時間が意図通り変わることを確認する。
3. built-in `weight time` と、明示cut制御を同じstate・horizonで比較する。
4. 明示cutに追加価値がある場合だけ、低次元actionまたはrank-weight入力拡張を検討する。

現時点の回答は「rankごとの担当領域制御は標準LAMMPSと既存研究コードで可能。ただし
rank別skin等は不可、rank別任意weightをLAMMPSへ直接入力する専用コマンドはない」である。

## 5. 既存研究実装の再利用範囲

`rank_target_balance.py` のactionは32要素のcapacity vector

```text
c = (c_0, ..., c_31),  c_i > 0
```

であり、rank `i` の目標atom数を

```text
N_i^target = N_total * c_i / sum_j(c_j)
```

としている。atomの経験的x方向CDFをこの累積目標で逆変換し、31 cutを作る。
同時に、隣接cut間の実距離がskin未満ならactionをrejectする安全checkがある。

この実装はamp2と同じ491,520 atoms、MPI32、`32 x 1 x 1`、skin 0.5という主要条件に
一致するため、制御核 `target_counts()` と `cuts_from_x()` は再利用できる。一方、次は
現状hard-codeされているため、amp2 runnerへそのまま接続する前に設定化が必要である。

- atom数、rank数、skin、500-step horizon
- checkpoint pathと初期化手順
- CPU contention対象core（現在はrank 0--7を想定）
- dumpによる全atom x座標収集

特に全atom座標dumpとPython側sortはcontroller overheadになり、既存ampb評価では約
0.26--0.30秒を要した。この費用もrewardへ含める必要がある。将来高速化するなら、
座標histogramまたはrank-local累積量だけをMPI集約するlibrary-side実装が候補になる。

## 調査根拠

- `src/balance.cpp`: `Balance::command()`, explicit XYZ cuts, migration/reset
- `src/fix_balance.cpp`: `FixBalance::rebalance()`, dynamic rebalance and subbox check
- `src/imbalance_neigh.cpp`: per-rank neighbor-derived weights
- `src/imbalance_time.cpp`: per-rank timing-derived weights
- `src/imbalance_var.cpp`: atom-variable-derived weights
- `src/procmap.cpp`, `doc/src/processors.rst`: processor grid and rank mapping
- `doc/src/balance.rst`: numeric cuts and weighted balancing specification
- Existing Shock logs: MPI32 runs report `32 by 1 by 1 MPI processor grid`
