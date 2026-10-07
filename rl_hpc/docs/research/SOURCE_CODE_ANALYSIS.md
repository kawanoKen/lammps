# LAMMPSソース実装に基づく内部状態適応の可能性

対象はこのworktreeの LAMMPS（`c8bd2ae`）である。これは実験結果の
推測ではなく、C++実装とbundled inputの静的調査である。特に
`neighbor skin` と `balance` / `fix balance` が、どの内部・計算状態で
有利になり得るかを整理する。

## 結論

- `skin` は **候補近傍数・ghost/通信範囲を増やすコスト** と、**再近傍化
  （atom migration、binning、neighbor-list構築）の頻度** の交換条件である。
  温度・速度・box変形は後者、密度・局所構造・粒子数は前者を変えるため、
  実装上は最適値が変化し得る。
- `balance` は rankごとの atom数またはユーザ定義の重みの最大/平均比を
  評価し、閾値を超えたときだけdomainを再分割する。空間密度の偏り、局所的に
  高価なpair/neigh/KSpace、移動するcluster/interfaceは価値を上げる。
- 単に温度や総粒子数が一様に変わるだけなら、全rankに同程度に作用するため、
  `balance`の利益は必ずしも増えない。重要なのはrank間の**不均一性**である。
- GPU/KOKKOSでは `balance ... weight neigh` は実装上スキップされる。GPUで
  neighbor数を直接重みにしたbalance実験を設計してはいけない
  （`src/imbalance_neigh.cpp:45-58`）。

## 1. `neighbor skin` の実装経路

### cutoff、neighbor-listサイズ、Pair/Neighコスト

`Neighbor::init()` は全type pairについて実効neighbor cutoffを
`pair cutoff + skin` として `cutneighsq`、`cutneighmax`、
`cutneighmaxsq` に格納する（`src/neighbor.cpp:337-388`）。したがって
skinを大きくすると、一定密度では概ね `(r_cut + skin)^d` に比例して候補
neighbor数が増える（dは次元）。実装上の直接の帰結は次の通り。

- `Neighbor::build()` はbin/stencilと各neighbor listを構築する
  （`src/neighbor.cpp:2525-2585`）。大きいskinはbin/stencil探索とlist memoryを
  増やす。
- Pair styleはlist中の候補に対し実pair cutoffを判定するため、skin内だけの
  候補もdistance check等の負担になる。ゆえにPair時間だけでなくNeigh時間も
  増え得る。
- communication cutoffもneighbor cutoffから導かれ、ghost領域とComm量を増やす
  （`src/neighbor.cpp:1818-1840`）。小domain・多数MPI rankほどこの影響は強い。

### rebuild頻度、温度・速度・box変形

`Neighbor::decide()` は`delay`後、`every`の倍数stepだけ判定し、`check yes`
なら`check_distance()`に委ねる（`src/neighbor.cpp:2439-2455`）。
`check_distance()`は、どれかのatomが最後のbuildから**skinの半分**を超えて
移動したらbuildを要求する（`triggersq = 0.25*skin^2`、
`src/neighbor.cpp:337-345, 2469-2519`）。Verletループはこの要求に対して
exchange、各`pre_exchange`/`pre_neighbor` hook、`neighbor->build()`を実行する
（`src/verlet.cpp:264-293`）。

boxが変化する場合、`check_distance()`はbox cornerの変位を計算し、利用可能な
skinを保守的に縮める（`src/neighbor.cpp:2469-2508`）。従って`fix deform`や
triclinic shearは、原子速度が同じでもrebuildを早め得る。

| 状態変化 | 実装から予想されるskinへの作用 |
|---|---|
| 温度・拡散係数・流速の上昇 | 原子変位が早く`skin/2`に達する。小skinのrebuild頻度が増えるので、より大きいskinが相対的に有利になり得る。 |
| 密度上昇 | 同一skinでも候補neighbor数が増え、Pair/Neigh/Commが増加する。小skinが有利になり得る。 |
| cluster/void/interface | rankごとの局所密度とneighbor数が異なる。global skinの最適値はdense領域のPair/Neighとdilute領域のrebuild頻度の交換になる。 |
| box deformation | corner変位で有効triggerが縮む。小skinの頻繁なrebuildが起きやすく、より大きいskinの相対価値が上がり得る。 |
| MPI rank増加・細長いsubdomain | skin由来のghost halo比率が増え、Comm・neighbor候補のコストが増す。最適skinはrank数・processor gridに依存し得る。 |
| 総粒子数のみ一様に増加 | Pair/Neighの絶対コストは増えるが、状態変化だけで最適skinが変わるとは限らない。密度、rank数、または速度の変化も必要。 |

`neigh_modify every`/`delay`はbuild判定の上限頻度を制約するだけで、`check yes`
では安全な変位判定を無効化しない。一方`check no`は物理的に危険な設定を許す
可能性があるため、本研究のperformance actionから除外すべきである。

## 2. `balance` と `fix balance` の実装

### 何をimbalanceとして測るか

基本のimbalanceは `max(load_rank) / mean(load_rank)` である。
`Balance::imbalance_factor()` は重みなしなら`atom->nlocal`、重みありなら各local
atomのweightの和をMPI_Allreduceし、この比を返す
（`src/balance.cpp:538-561`）。従ってdefaultの`balance`は**atom数balance**で、
Pair/Neigh/KSpace時間を直接測るわけではない。

重みは`balance`/`fix balance`の`weight` optionから組み立てられる
（`Balance::set_weights()`、`src/balance.cpp:523-535`）。主要な実装は以下である。

- `weight neigh`: rankのhalf/full neighbor数をlocal atom数で割ってper-atom重みと
  する（`src/imbalance_neigh.cpp:45-100`）。密な領域・多body近傍の偏りに対応する。
- `weight time`: Pair + Neigh + Bond + KSpaceの累積wall timeをlocal atom数で割る
  （`src/imbalance_time.cpp:55-103`）。計算量の偏りを直接反映できるが、timer精度を
  避けるため0.1秒未満では更新しない。
- `weight group`、`store`、`var` もある（`src/imbalance_group.cpp`、
  `src/imbalance_store.cpp`、`src/imbalance_var.cpp`）。異種粒子・反応領域・
  ユーザ計算量モデルに使える。

### 再分割方式とコスト

- `balance` は一回だけ実行するCommandで、必要ならLAMMPS再初期化、PBC/reset、
  atom migration、grid resetまで行う（`Balance::command()`、
  `src/balance.cpp:103-390`）。閾値未満なら再分割しない。
- `fix balance N thresh shift xyz Niter stopthresh` は`N` stepごとに
  `pre_exchange()`でimbalanceを測り、`thresh`超過時だけ再分割する
  （`FixBalance::pre_exchange()`、`src/fix_balance.cpp:229-266`）。`N=0`は初期評価のみ。
- `shift` はlogical processor gridのcut planeをx/y/z順に反復移動する。
  `Balance::shift()`はslice tally、MPI集約、最大`niter`回のadjustを行う
  （`src/balance.cpp:774-1020`）。brick/nonuniform decomposition向きである。
- `rcb` はrecursive coordinate bisectionでtiled subdomainを作る。
  `Balance::bisection()`はglobal bounding boxを求め、RCBでcutとatom送付先を決める
  （`src/balance.cpp:569-705`）。`comm_style brick`では使用できない
  （`src/balance.cpp:238`, `src/fix_balance.cpp:118`）。
- 再balanceの実コストはweight計算、global reductions、cut決定、`Irregular` atom
  migration、`domain->set_local_box()`、distributed gridとPair/KSpace gridのresetである
  （`FixBalance::rebalance()`、`src/fix_balance.cpp:284-348`）。短いphaseや小さい
  imbalanceでは、この一時コストが将来の短縮を上回る。

`FixBalance::rebalance()`はsubboxがneighbor skinより小さくならないか検査する
（`src/fix_balance.cpp:305-315`）。このためskinとbalanceは独立ではない。skinを大きく
すると細いdense slabへcutを寄せる自由度が小さく、balanceの実現可能な分割も制約される。

### どの状態でbalanceの価値が変わるか

| 状態変化 | atom-count balance | `weight neigh/time` balance | 再balanceの価値 |
|---|---|---|---|
| 一様な温度上昇 | 通常は変化しない | 全rank同等なら変化しない | 小さい |
| 局所加熱・衝撃波・moving indenter | atom数だけでは見えない | Pair/Neigh/KSpace時間の局所増加を捉え得る | 高い可能性 |
| cluster / droplet / void / interface移動 | rank atom数が偏る | neighbor数・pair計算量も偏る | 高い。位置移動に合わせdynamic fixが必要。 |
| deposition / GCMC / evaporation | particle countと密度分布が時間変化 | 反応・近傍の偏りも追跡可能 | 高い。固定一回balanceより`fix balance`が自然。 |
| box deformation / domain shape変化 | uniform密度ならatom数は均等でもsubbox形状が悪化し得る | Comm/KSpaceの偏りは標準weightに完全には出ない | 中〜高。processor gridもepisode-level候補。 |
| MPI rank数増加 | 小domain化で偏り・halo差の影響が増幅 | `time` weightの観測粒度も変わる | rank数が多いほど検討価値が上がる。 |

## 3. bundled workload候補（未実行）

### skinの状態依存性を狙う候補

1. `examples/indent/in.indent` — moving spherical indenterが局所高密度・塑性変形・
   非一様速度を作る。indenter接近/除去でrebuild頻度とdense contactのneighbor候補数の
   交換条件が変わる見込みがある。
2. `examples/deposit/in.deposit.atom` — `fix deposit`が上部slabへatomを追加する。
   粒子数・局所密度・interfaceが時間とともに成長し、skinの候補listコストとmigration
   頻度の双方が変化する。
3. `examples/shear/in.shear.void` — 初期voidと剪断がvoid形状、局所密度、速度場を変える。
   `neigh_modify delay 5`を含むため、safeな`check yes`条件に明示化して比較する候補。

### balanceの状態依存性を狙う候補

1. `examples/balance/in.balance` — 移動する2D粒子clusterを`comm_style tiled`と
   `fix ... balance 50 0.9 rcb`で追従する、最も直接的なbundled validation候補。
2. `examples/deposit/in.deposit.atom` — 成長するsurface slabでrankごとのatom数が偏る。
   atom-count、`weight time`、およびbalance頻度の比較に適する。
3. `examples/shear/in.shear.void` または `examples/indent/in.indent` — void/indenterにより
   hotspotが移動する。atom-countだけでなく`weight time`が有効かを検証する候補。

## 4. 状態 × control 仮説の要約

| 変化させる状態 | skinへの影響予想 | balanceへの影響予想 | 理由 | 候補workload |
|---|---|---|---|---|
| 温度・局所速度 | 高速原子でrebuild増。大skinが相対的に有利になり得る | 一様なら小、局所なら`weight time`で大 | `check_distance()`がskin/2移動を判定 | `indent`, `shear.void` |
| 密度・総粒子数の局所増加 | 候補neighbor数増で小skin寄り | atom数/neighbor重みのimbalance増 | cutoff+skinがlist/haloを拡張 | `deposit.atom` |
| cluster/void/interfaceの移動 | dense/dilute領域の交換条件が変化 | 高い。domain cutをcluster位置へ追従する価値 | 最大rank costが平均から乖離 | `balance/in.balance`, `shear.void` |
| box deformation | 有効skin trigger縮小、rebuild増 | subbox形状・Commが悪化し得る | box-corner変位をskinから差し引く | `ELASTIC_T/DEFORMATION`, `shear.void` |
| MPI rank数・domain shape | halo/Comm比で最適skinが変化し得る | rank増で偏りの影響増、RCB/shiftの選択も変化 | `cutneighmax`とlocal subbox形状が直接関与 | `balance/in.balance`, `deposit.atom` |
| Pair/Comm/Neigh/KSpace比 | Pair/Neigh優勢ならskin感度増 | `weight time`ならこの比のrank偏りを追跡 | time weightはPAIR/NEIGH/BOND/KSPACEを加算 | `indent`, `shear.void`, SPC/E派生 |

## 実験設計上の含意

内部状態のみでaction reversalを主張するには、同じrestartからabsolute actionを分岐し、
checkpointごとの`T(s_t,a)`を比較する必要がある。自然な均一LJ/SPC/Eでは既にその
反証的な測定でreversalを確認できなかった。

次に試すべきなのは、単なるthermo変動ではなく、**移動する空間的不均一性**を作る
`examples/balance/in.balance`である。まず`fix balance ... report`でimbalance時系列を
観測し、次に`shift`/`rcb`と頻度・閾値を分岐比較する。skinの実験は`indent`または
`deposit`のrestart分岐で、`check yes`を保った安全領域だけを扱うべきである。
