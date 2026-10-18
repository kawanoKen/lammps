# 独立CPU contention観測：凍結方策のpaired live評価

## 評価条件

- workload：Shock/NEMD、491,520 atoms
- CPU/MPI：32 ranks、1 thread/rank
- episode：500 MD steps × 120 decisions
- external state：CPU 0–7に対するL0/L4、45–90 wall-clock秒で独立切替
- held-out physical seeds：47287、57287
- repeats：各seed 2回
- 方策：aware RL、blind RL、factor 0.8固定、factor 1.0固定
- 全方策のparameterを凍結し、探索・online updateなし
- 同じseed/repeatでは同じrestartと同じwall-clock contention traceを使用

全16 episodesが正常終了した。dangerous builds、crash、nonzero exitは0だった。16回の
baseline guardはすべて合格し、2,895個の実行中monitor samplesでstudy外の1%以上CPU processは
記録されなかった。

## 初回評価（各方策n=4）

| 方策 | Mean, s | SD, s | awareとの差（aware − 比較対象）, s | Paired 95% CI, s |
|:---|---:|---:|---:|---:|
| aware RL | 891.269 | 14.685 | -- | -- |
| blind RL | 895.813 | 15.366 | -4.544 (-0.51%) | [-17.727, +8.639] |
| fixed factor 0.8 | 928.888 | 16.518 | -37.619 (-4.05%) | [-62.883, -12.355] |
| fixed factor 1.0 | 918.559 | 17.920 | -27.290 (-2.97%) | [-52.091, -2.490] |

awareは4組中3組でblindより速く、平均0.51%短かった。しかし`n=4`でpaired CIが0を跨ぐため、
contention観測がblindより実行時間を改善したとはまだ結論できない。固定0.8、固定1.0に対する
awareの差は、この評価ではCIが0を跨がなかった。

## 観測状態に対するaction応答

action選択前のCPU 0–7 utilization meanを60%で分けた記述統計である。これはhidden L0/L4
labelではなく、方策が実際に受け取る直前区間の観測値に基づく。

| 方策 | 観測CPU状態 | n | Balance率 | Balance時factor平均 |
|:---|:---|---:|---:|---:|
| aware | low（平均util 17.7%） | 271 | 93.4% | 0.994 |
| aware | high（平均util 97.8%） | 209 | 85.2% | 1.318 |
| blind | low（平均util 17.4%） | 269 | 95.5% | 1.188 |
| blind | high（平均util 97.4%） | 211 | 94.8% | 1.217 |

awareは観測負荷に応じてfactorを大きく変えており、high contentionで大きいfactorを選ぶ。
blindのfactor平均はlow/highで近い。したがって、resource観測がactionへ反映されていることは
確認できた。ただしaction変化がblindに対する有意な総時間短縮を生んだかは未確定である。

## 結論と次の判定

今回確認できた因果鎖は次の途中までである。

```text
L0/L4を観測から識別可能
  -> aware方策のfactor選択が変化
  -> 固定factorより短時間
  -> blind方策より短時間かはn=4では未確定
```

次に必要なのは新しい方策の再学習ではなく、同じ凍結checkpointによる追加paired evaluationで
ある。現在のpaired差のばらつきでは、まず各方策`n=8`まで増やしてCIを再計算する。差が依然
不確定なら、効果量が小さいと判断し、無制限に反復数を増やさない。

## 追加paired評価（Aware / Blindをn=8へ拡張）

固定方策は再実行せず、同じ凍結Aware / Blind checkpointについて新しいschedule seedを使う
4 paired条件を追加した。追加8 episodesもすべて正常終了した。

| 方策 | n | Mean, s | SD, s |
|:---|---:|---:|---:|
| aware RL | 8 | 892.275 | 11.280 |
| blind RL | 8 | 893.852 | 11.486 |

paired差`aware - blind`は次の通りである。

$$
\overline{\Delta T}=-1.577\ \mathrm{s}\;(-0.18\%),
\qquad 95\%\ \mathrm{CI}=[-9.686,+6.532]\ \mathrm{s}.
$$

Awareは8組中5組で速かったが、平均効果は初回n=4の-0.51%から-0.18%へ縮小し、CIは0を
跨いだ。したがって、このworkload・外乱・状態設計では、CPU utilizationを明示的に入力する
ことがBlind RLより総時間を短縮するという証拠は得られなかった。

ただし、action応答の違いはn=8でも維持された。

| 方策 | 観測CPU状態 | n decisions | Balance率 | Balance時factor平均 |
|:---|:---|---:|---:|---:|
| aware | low | 529 | 92.4% | 0.981 |
| aware | high | 431 | 86.3% | 1.332 |
| blind | low | 526 | 95.8% | 1.177 |
| blind | high | 434 | 94.7% | 1.230 |

すなわちAwareはcontentionを識別してactionを変えたが、**CPU utilizationを明示的に追加した
ことによる**実用的改善はほぼ0だった。これはbalance factor制御そのものの効果が弱いことを
意味しない。過去の固定factor比較ではCPU contentionによる順位差が観測され、連続factorを
選ぶ学習方策も試した固定設定を上回っている。今回言えるのは、Pair/Comm timing等を含むBlind
stateがcontentionの影響を既に間接観測しており、追加CPU telemetryが冗長だった可能性が高い、
ということまでである。内部simulation phaseへの適応と外部CPU contentionへの適応の寄与は、
このAware対Blind比較だけでは分離できない。0.18%差を有意にする目的だけで反復数をさらに
増やすことは推奨しない。

追加分を含む全24 evaluation episodesでdangerous builds、failure、外部高負荷process検出は0、
baseline guardは24/24合格だった。
