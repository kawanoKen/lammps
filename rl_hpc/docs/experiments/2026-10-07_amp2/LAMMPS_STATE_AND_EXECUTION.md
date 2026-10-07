# LAMMPSにおける状態・領域分割・同期の整理

本資料は、amp2上のShock/NEMD実験で扱う状態と、LAMMPSが粒子移動・MPI通信・
neighbor listをどのように処理するかを整理したものである。

## 1. 状態の分類

LAMMPSの状態は、大きく「物理状態」と「software・性能状態」に分けて考える。

### 1.1 物理状態

物理状態は、シミュレーション対象そのものの状態である。

| 状態量 | 意味 | 性能との関係 |
|:---|:---|:---|
| Temperature | 粒子の運動エネルギーに対応する温度 | 高温では粒子移動が速くなり、neighbor list再構築頻度が増え得る |
| Pressure | 系の圧力・応力状態 | shock frontの形成・通過や圧縮状態を表す |
| Density | 単位体積当たりの粒子数 | 高密度領域では近傍粒子数とPair計算量が増える |
| Energy | 運動・ポテンシャル・全エネルギー | action間で物理的妥当性が保たれているかを確認する |
| 粒子の空間分布 | cluster、void、interface、高密度・低密度領域など | MPI rank間の計算負荷不均衡を発生させる |
| Shock進行状態 | shock front位置、圧縮領域の広がり | シミュレーション内部の非定常性を表す |

MD stepやshock位置は状態の解析には使えるが、方策入力にすると時間進行を暗記する
可能性がある。一般化を重視する場合は、圧力、密度、不均衡、timingなど、実際の
計算状態を表す観測量を優先する。

### 1.2 Software・性能状態

software状態は、同じ物理状態をどのような計算配置と設定で処理しているかを表す。
性能状態は、その結果としてどの処理に時間を使っているかを表す。

| 状態量 | 意味 |
|:---|:---|
| 現在のconfiguration | neighbor skin、balance有無、neighbor-weight factorなど |
| MPI領域分割 | 各rankのsubdomain境界、特にx方向subdomain幅 |
| `Nlocal` | 各rankが正式に所有する粒子数 |
| `Nghost` | 隣接rankから通信で受け取った粒子コピー数 |
| `Neighs` | rankが保持するneighbor listの総entry数、すなわち近傍pair候補数 |
| Neighbor build | 区間内のneighbor list再構築回数とdangerous build数 |
| Pair time | 粒子間相互作用計算に要した時間 |
| Neigh time | neighbor list構築に要した時間 |
| Comm time | ghost通信、粒子移管、力の返送などに要した時間 |
| Modify time | fixや時間積分などに要した時間 |
| 区間wall time | 固定MD stepsを完了するための実時間 |

状態の最小構成は、概念的には次のように表せる。

\[
s_t = (
T, P, \rho,
I_{\mathrm{atom}}, I_{\mathrm{neigh}}, I_{\mathrm{ghost}},
\text{subdomain形状}, \text{現在の設定},
T_{\mathrm{Pair}}, T_{\mathrm{Neigh}}, T_{\mathrm{Comm}},
\text{neighbor build率}, \Delta\tau
).
\]

直前区間のtimingはaction選択前に利用できる観測である。現在区間のwall timeは
実行終了後にrewardと次状態として得られる。

## 2. `Nlocal`、`Nghost`、`Neighs`

LAMMPSは空間をMPI rankごとのsubdomainへ分割する。全rankが全粒子を保持する
方式ではない。

```text
MPI rank 0       MPI rank 1       MPI rank 2
┌──────────┬──────────┬──────────┐
│ 担当領域  │ 担当領域  │ 担当領域  │
└──────────┴──────────┴──────────┘
```

### 2.1 `Nlocal`

`Nlocal`は、そのrankが正式に所有する粒子数である。粒子がsubdomain境界を越えると、
所有権は新しいrankへ移る。

`Nlocal`は時間積分や粒子単位の処理量に関係する。ただし、高密度領域では1粒子
当たりのneighbor数が多いため、`Nlocal`が同じでも計算量が同じとは限らない。

### 2.2 `Nghost`

`Nghost`は、隣接rankが所有する粒子のうち、自rankの力計算に必要な粒子のコピー数
である。ghost atomは新しい物理粒子ではなく、通信上の複製である。

rank境界付近のlocal atomは隣rankの粒子とも相互作用するため、その位置情報をghost
として受け取る必要がある。おおむねcutoffとskinで決まる通信範囲が広いほど、
`Nghost`も増える。

`Nghost`は主に通信量、メモリ使用量、neighbor list構築量に関係する。

### 2.3 `Neighs`

`Neighs`は粒子数ではなく、neighbor listに登録された近傍関係の総数である。

```text
local atom A -> [B, C, D]
local atom E -> [F, G]
local atom H -> [I, J, K, L]
```

この例のneighbor entry数は `3 + 2 + 4 = 9` である。参照先にはlocal atomとghost
atomの両方が含まれ得る。

したがって、次の等式は成立しない。

\[
N_{\mathrm{neigh}} = N_{\mathrm{local}} + N_{\mathrm{ghost}}.
\]

概念的にはPair計算量は`Neighs`と強く関係する。ただし、half listかfull listか、
pair style、Newton設定、CPU/GPU実装によってentry数と実コストの関係は変わる。

## 3. Neighbor listの所有方法

概念的には各local atomに近傍リストがある。ただし、各粒子が独立したオブジェクト
としてリストを持つのではない。各MPI rankが、自rankのlocal atomを起点とする
neighbor listを、効率的な配列・ページ構造としてまとめて保持する。

```text
MPI rank r
├─ local atom A -> neighbor IDs [...]
├─ local atom B -> neighbor IDs [...]
├─ local atom C -> neighbor IDs [...]
└─ ghost atoms  -> 上記リストから参照されるコピー
```

代表的な形式は次の2つである。

- Half list：A-B pairを一度だけ登録し、Newtonの第三法則を利用する。
- Full list：AのlistにB、BのlistにAを登録する。重複計算は増えるが、GPU等で並列化
  しやすい場合がある。

`Neighs`の絶対値を比較するときは、同じlist方式と実装で比較する必要がある。

## 4. 不均衡を表す指標

### 4.1 Atom imbalance

\[
I_{\mathrm{atom}}
=
\frac{\max_r N_{\mathrm{local},r}}
     {\operatorname{mean}_r N_{\mathrm{local},r}}.
\]

- 1.0：全rankのlocal atom数が均等
- 1.5：最大rankが平均の1.5倍
- 3.0：最大rankが平均の3倍

これは粒子数の偏りであり、計算負荷の偏りそのものではない。

### 4.2 Neighbor imbalance

\[
I_{\mathrm{neigh}}
=
\frac{\max_r N_{\mathrm{neigh},r}}
     {\operatorname{mean}_r N_{\mathrm{neigh},r}}.
\]

Pair計算負荷をatom imbalanceより直接的に表す場合がある。shock後方の高密度領域は
1粒子当たりのneighbor数が多いため、neighbor重み付きbalanceでは、その領域を担当
するrankへ意図的に少ない粒子を割り当てる場合がある。そのためatom imbalanceが
大きくても、計算負荷は改善している可能性がある。

### 4.3 Ghost imbalance

\[
I_{\mathrm{ghost}}
=
\frac{\max_r N_{\mathrm{ghost},r}}
     {\operatorname{mean}_r N_{\mathrm{ghost},r}}.
\]

通信・境界処理負荷の偏りを表す。ただし、通信相手数、メッセージサイズ、待ち時間
まではこの値だけでは分からない。

### 4.4 Timing imbalance

\[
I_{\mathrm{time}}
=
\frac{\max_r T_r}{\operatorname{mean}_r T_r}.
\]

Pair、Comm、総計算時間などについて計算でき、実際の待ち時間に近い指標である。
一方で、同期や計測区間の影響を受ける。

### 4.5 Relative imbalance

RL実装で用いる相対表現は次のとおりである。

\[
I_{\mathrm{relative}}
=
\frac{\max-\mathrm{mean}}{\mathrm{mean}}
=
\frac{\max}{\mathrm{mean}}-1.
\]

`max/mean = 1.5`と`relative imbalance = 0.5`は同じ偏りを異なる形式で表している。

## 5. 粒子移動の時間間隔

元のShock/NEMDは`units lj`、`timestep 0.0005`、`fix nve`を使用する。粒子は
500 stepsごとではなく、1 MD stepごとにVelocity Verlet法で更新される。

各stepでは概ね次の処理を行う。

```text
1. 速度を半step更新
2. 位置を1 step更新
3. ghost位置を通信、または粒子を新しいrankへ移管
4. 必要ならneighbor listを再構築
5. 粒子間力を計算
6. ghost側で計算した力を所有rankへ返す
7. 速度を残り半step更新
```

今回の制御間隔は500 MD stepsなので、1 decisionに対応する物理時間は

\[
500 \times 0.0005 = 0.25
\]

Lennard-Jones時間単位である。500 stepsは観測・制御間隔であり、その内部でも粒子
移動とMPI通信は毎step繰り返される。

## 6. Neighbor list再構築

今回の設定は次のとおりである。

```text
neigh_modify every 1 delay 0 check yes
```

LAMMPSは毎step、neighbor listを再構築すべきか確認するが、毎step必ず再構築する
わけではない。粒子が前回構築時から十分移動した場合、一般に最大移動量がskinの
半分程度へ達した場合に再構築する。

小さいskinはlistを短くしてPair計算を減らす一方、再構築を増やしやすい。大きい
skinは再構築を減らす一方、`Nghost`と`Neighs`を増やし、Pair・Commコストを増やす
可能性がある。

## 7. MPI通信と実質的な同期点

LAMMPSが毎stepの先頭と末尾に必ず全rank barrierを置くわけではない。しかし必要な
データが届くまで先へ進めないため、次の通信が実質的な同期点として働く。

### 7.1 通常step

neighbor listを再構築しないstepでは`comm->forward_comm()`により、所有rankから
ghost側へ最新の位置などを送る。

### 7.2 Neighbor list再構築step

- `comm->exchange()`：subdomain境界を越えた粒子の所有rankを変更する。
- `comm->borders()`：新しいghost atomを構築する。
- `neighbor->build()`：新しいlocal/ghost配置からneighbor listを構築する。

### 7.3 力計算後

Newton設定が有効な場合、`comm->reverse_comm()`によりghost側で計算した力を粒子の
所有rankへ返す。

### 7.4 全体統計と出力

temperature、pressure、energy、rank最大値・平均値などの全体系統計にはMPI collective
が必要になる。ただし、出力・集約頻度によっては毎step実行されない。

### 7.5 Balance実行時

`balance`を実行すると、領域境界の再計算、subdomain変更、粒子所有rankの変更、ghost
再構築、neighbor list再構築が発生する。そのためbalanceは単なるscalar parameter
変更ではなく、全rankへ影響する実行コストと持続的な状態変化を持つactionである。

速いrankは通信・同期点で遅いrankを待つ。このため最大負荷rankを減らすことが重要に
なる。

```text
速いrank ─────── 待機 ── 次step
遅いrank ────────────── 次step
```

## 8. Shock/NEMDでの関係

shockが伝播すると、圧縮領域の密度と1粒子当たりneighbor数が増える。その結果、
時間とともに次の量が変化する。

- rankごとの`Nlocal`
- `Nghost`
- `Neighs`
- atom・neighbor・ghost imbalance
- Pair、Neigh、Comm時間
- 最適なMPI領域境界

したがって、原子数だけを均等化するbalanceが常に最良とは限らない。`weight neigh`
はneighbor計算量の大きい領域へ高い重みを与え、その領域を担当するsubdomainを狭く
するための機構である。

要点をまとめると、次のとおりである。

- `Nlocal`：自rankが所有し、直接担当する粒子数
- `Nghost`：隣rankから計算のために受け取った粒子コピー数
- `Neighs`：local atomから参照する近傍pair候補の総数
- 物理状態：何をsimulateしているか
- software状態：その物理系をどの分割・設定で計算しているか
- 性能状態：Pair、Neigh、Commなど、どこへ実時間を使っているか

