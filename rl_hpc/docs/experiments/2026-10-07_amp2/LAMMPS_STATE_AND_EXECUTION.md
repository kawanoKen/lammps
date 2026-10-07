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
はrankごとの平均neighbor数から計算重みを作り、neighbor計算量の大きいrank側の
subdomainを狭くするための機構である。

要点をまとめると、次のとおりである。

- `Nlocal`：自rankが所有し、直接担当する粒子数
- `Nghost`：隣rankから計算のために受け取った粒子コピー数
- `Neighs`：local atomから参照する近傍pair候補の総数
- 物理状態：何をsimulateしているか
- software状態：その物理系をどの分割・設定で計算しているか
- 性能状態：Pair、Neigh、Commなど、どこへ実時間を使っているか

## 9. 制御パラメータの定義

ここでは更新方策や良否ではなく、LAMMPSへ与える制御パラメータと、実装内での定義を
整理する。主要な制御変数は概念的に次のように書ける。

\[
a_t = (s_t^{\mathrm{skin}}, M_t, b_t, f_t,
       \theta_t, K_t, \eta_t, d_t),
\]

ここで、skinを \(s^{\mathrm{skin}}\)、`every`を \(M\)、balanceの実行有無を
\(b\)、neighbor weight factorを \(f\)、balance開始thresholdを \(\theta\)、最大反復
回数を \(K\)、停止thresholdを \(\eta\)、balance対象方向を \(d\) と表す。

### 9.1 Neighbor skin：\(s^{\mathrm{skin}}\)

```text
neighbor SKIN bin
```

型は正の実数である。原子type \(i,j\) 間の力のcutoffを \(r^{\mathrm{cut}}_{ij}\) と
すると、neighbor listへ登録する距離cutoffは

\[
r^{\mathrm{neigh}}_{ij}
= r^{\mathrm{cut}}_{ij} + s^{\mathrm{skin}}
\]

である。粒子間距離を \(r_{pq}\) とすると、通常のlist登録条件は

\[
r_{pq}^2 \leq
\left(r^{\mathrm{cut}}_{ij}+s^{\mathrm{skin}}\right)^2
\]

となる。skinは力を実際に計算する物理cutoffではなく、次回のlist再構築までに粒子が
移動できる余裕幅である。

今回のbalance RLでは

\[
s^{\mathrm{skin}}=0.5
\]

へ固定しており、actionには含めていない。

### 9.2 `every`：\(M\)

```text
neigh_modify every M delay D check yes
```

\(M\)は正の整数で、neighbor list再構築を検討できるstep間隔である。前回のbuildから
の経過step数を \(A\)、`delay`を \(D\) とすると、通常の候補step条件は

\[
A \geq D
\quad\land\quad
A \bmod M = 0
\]

である。

`check yes`の場合、前回build時からの粒子 \(i\) の移動量を
\(\lVert\Delta\mathbf{x}_i\rVert\) とすると、さらに

\[
\max_i \lVert\Delta\mathbf{x}_i\rVert^2
>
\frac{(s^{\mathrm{skin}})^2}{4}
\]

が成立したときに再構築する。すなわち、少なくとも1粒子がskinの半分より大きく移動
したことを判定する。

今回の設定は

\[
M=1,\qquad D=0,\qquad \mathrm{check}=\mathrm{yes}
\]

であり、`every`も現在のbalance RL actionには含めていない。

### 9.3 Balance実行有無：\(b\)

one-shot balanceについて

\[
b\in\{0,1\}
\]

とする。

- \(b=0\)：`balance`を実行せず、現在のMPI partitionを保持する。
- \(b=1\)：指定されたbalanceパラメータでMPI partitionを再計算する。

今回のRLでは、これは`skip`または`balance`という離散actionに対応する。

### 9.4 Balanceの負荷と不均衡：\(L_r, I\)

rank \(r\) が所有するlocal atom集合を \(\mathcal{P}_r\)、原子 \(i\) のbalance weightを
\(w_i\) とすると、rank負荷は

\[
L_r = \sum_{i\in\mathcal{P}_r} w_i
\]

である。重みを指定しない場合は

\[
w_i=1,
\qquad
L_r=N_{\mathrm{local},r}
\]

となる。

MPI rank数を \(P\) とすると、LAMMPSのbalance imbalance factorは

\[
I =
\frac{\max_r L_r}
     {\frac{1}{P}\sum_{r=1}^{P}L_r}
\]

である。

### 9.5 Balance開始threshold：\(\theta\)

```text
balance THRESH ...
```

`THRESH`を \(\theta\) とする。初期不均衡 \(I_{\mathrm{init}}\) に対し、one-shot
`balance`は基本的に

\[
I_{\mathrm{init}} \geq \theta
\]

のとき実行対象となる。今回のコマンドでは

\[
\theta=1.0
\]

である。

### 9.6 Balance方向：\(d\)

```text
shift x
```

\(d\)は移動させるprocessor切断面の方向である。

\[
d \subseteq \{x,y,z\}
\]

今回のShock/NEMDでは

\[
d=\{x\}
\]

であり、x方向の切断面だけを変更する。

### 9.7 最大反復数：\(K\) と停止threshold：\(\eta\)

```text
shift x NITER STOPTHRESH
```

`NITER`を \(K\)、`STOPTHRESH`を \(\eta\) とする。shift法は最大 \(K\) 回の切断面探索を
行い、探索中の不均衡が

\[
I \leq \eta
\]

になれば停止する。

今回の設定は

\[
K=10,\qquad \eta=1.0
\]

である。実際にはsubdomain幅にskin由来の制約があるため、常に厳密な \(I=1\) を達成
できるとは限らない。

### 9.8 Neighbor weight factor：\(f\)

```text
weight neigh FACTOR
```

factorを \(f>0\) とする。まずrank \(r\) のneighbor entry総数を \(E_r\)、local atom数を
\(N_r=N_{\mathrm{local},r}\) とし、rank当たりの平均neighbor数を

\[
q_r=\frac{E_r}{N_r}
\]

と計算する。空rankは別扱いとなる。

全非空rankに対して

\[
q_{\min}=\min_r q_r,
\qquad
q_{\max}=\max_r q_r
\]

を求める。\(q_{\min}\neq q_{\max}\) の場合、LAMMPS実装がfactorを適用したrank重みは

\[
q'_r
=q_{\min}
+\frac{q_r-q_{\min}}{q_{\max}-q_{\min}}
 \left(fq_{\max}-q_{\min}\right)
\]

である。

そのrankが所有する各local atom \(i\in\mathcal{P}_r\) の既存weightは

\[
w'_i=w_i q'_r
\]

と更新される。したがって、各粒子固有のneighbor数を直接weightにするのではなく、
rank平均 \(q_r\) から作った同一の係数を、そのrank内の全local atomへ掛ける。

特別な場合は次のとおりである。

- \(f=1\) なら \(q'_r=q_r\) となる。
- \(f>1\) なら最大側の幅を拡大する。
- \(0<f<1\) なら最大側の幅を縮小する。
- \(q_{\min}=q_{\max}\) ならrank間差がないため、このneighbor weighting処理はskipされる。

今回のRLにおける範囲は

\[
f\in[0.5,1.5]
\]

である。反実仮想実験ではこの連続範囲から

\[
f\in\{0.50,0.75,1.00,1.25,1.50\}
\]

を評価した。

なお、この式では \(fq_{\max}<q_{\min}\) となるほど小さい \(f\) を選ぶと、変換後の
重み勾配が反転し得る。LAMMPSの入力条件は \(f>0\) だけだが、実験上の安全・妥当範囲
は別途定める必要がある。

### 9.9 `fix balance`の頻度：\(F\)

```text
fix ID all balance NFREQ THRESH shift x NITER STOPTHRESH ...
```

`NFREQ`を \(F\) とすると、`fix balance`は

\[
t \bmod F=0
\]

となるstepで不均衡を評価し、

\[
I_t>\theta
\]

なら再分割する。これはone-shot `balance`とは別の制御パラメータである。現在のRLは
`fix balance`の \(F\) や \(\theta\) をactionにはせず、500 stepsごとにone-shot
balanceの有無 \(b\) とfactor \(f\) を選ぶ。

### 9.10 現在のRL actionを数式で表したもの

現在のbalance-hybrid RLで実際に変化させる成分は

\[
a_t=(b_t,f_t),
\]

\[
b_t\in\{0,1\},
\qquad
f_t\in[0.5,1.5]\quad(b_t=1\text{のときのみ有効})
\]

である。その他は

\[
s^{\mathrm{skin}}=0.5,
\quad M=1,
\quad D=0,
\quad \theta=1.0,
\quad K=10,
\quad \eta=1.0,
\quad d=\{x\}
\]

へ固定している。
