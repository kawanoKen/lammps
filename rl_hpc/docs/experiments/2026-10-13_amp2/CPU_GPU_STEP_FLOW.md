# LAMMPSのCPU版とGPU版におけるMD stepの流れ

## 前提

CPU版とKOKKOS/CUDA版で、velocity-Verletによる物理計算の順序は基本的に同じである。
大きな違いは、処理をCPUとGPUのどちらで実行するか、MPI通信の前後でどこに同期と
memory transferが入るかである。

今回のGPU実験は次の構成である。

- MPI 32 ranks
- RTX 3090 × 4
- 8 ranks/GPU
- CUDA MPS有効
- CUDA-aware MPI無効：`gpu/aware off`

## 共通する1 MD step

```mermaid
flowchart TD
    A[step開始] --> B[速度を半step更新]
    B --> C[粒子位置を1 step更新]
    C --> D{neighbor listの<br/>再構築が必要か}
    D -- No --> E[ghost座標を更新]
    D -- Yes --> F[粒子migration]
    F --> G[ghost粒子を再生成]
    G --> H[neighbor listを再構築]
    H --> I[Pair forceを計算]
    E --> I
    I --> J[ghostに作用したforceを<br/>所有rankへ返送]
    J --> K[速度を残り半step更新]
    K --> L[fix・thermo・出力]
    L --> M[次step]
```

物理的には、おおむね次の順序である。

```text
v(t + dt/2)
  -> x(t + dt)
  -> communication / neighbor処理
  -> F(t + dt)
  -> v(t + dt)
```

## CPU版

CPU版では、座標・速度・力・neighbor listの主要データがCPU memoryにあり、各MPI rankが
自身のsubdomainをCPU上で処理する。

```mermaid
flowchart TD
    subgraph R[各MPI rank / CPU core]
        A[位置・速度更新] --> B{neighbor再構築?}
        B -- No --> C[ghost座標をpack]
        B -- Yes --> D[移動粒子をpack]
        D --> E[MPI粒子migration]
        E --> F[ghost粒子を交換]
        F --> G[CPUでneighbor list構築]
        G --> H[CPUでPair force計算]
        C --> I[MPI ghost座標交換]
        I --> H
        H --> J[MPI force返送]
        J --> K[速度更新・fix]
    end
```

rank `i` の代表的な負荷は次の量に依存する。

```text
Pair cost_i  ~ neighbor pair数_i
Neigh cost_i ~ neighbor list構築量_i
Comm cost_i  ~ ghost粒子数_i + subdomain境界面積_i
```

MPI区間の実行時間は、多くの場合、最も遅いrankに支配される。

```text
T_step ~= max_i(T_rank_i) + synchronization overhead
```

## KOKKOS/CUDA版

GPU版でもMPI process自体はCPU上に存在する。CPUがstep進行、MPI通信、GPU kernel起動を
制御し、粒子更新・neighbor list構築・Pair forceなどの対応kernelをGPUで実行する。

今回のOpen MPIはCUDA-awareではないため、MPI通信bufferはhost memoryを経由する。

```mermaid
flowchart TD
    subgraph HOST[CPU / MPI rank]
        A[step進行を制御] --> B[GPU kernelを起動]
        H2D[host bufferをGPUへ転送]
        MPI[MPI送受信]
    end

    subgraph GPU[GPU memory / KOKKOS kernels]
        C[位置・速度更新] --> D{neighbor再構築?}
        D -- No --> E[ghost通信bufferをpack]
        D -- Yes --> F[移動粒子bufferをpack]
        G[受信bufferをunpack] --> H[neighbor list構築]
        H --> I[Pair force計算]
        E --> I
        I --> J[force返送bufferをpack]
        K[受信forceを加算] --> L[速度更新・GPU側fix]
    end

    B --> C
    E --> SYNC1[GPU kernel完了待ち]
    F --> SYNC1
    SYNC1 --> D2H[GPUからhostへ転送]
    D2H --> MPI
    MPI --> H2D
    H2D --> G
    J --> SYNC2[GPU kernel完了待ち]
    SYNC2 --> D2H2[GPUからhostへ転送]
    D2H2 --> MPI2[MPI force返送]
    MPI2 --> H2D2[hostからGPUへ転送]
    H2D2 --> K
```

実装上、すべての処理で必ず図の通り個別copyが発生するとは限らない。しかし
`gpu/aware off`では、MPIが必要とするdevice dataをhost側から送受信できる状態にする
必要があり、GPU kernel、host/device transfer、MPIの間に依存関係と同期が生じる。

## neighbor listを再構築しないstep

```mermaid
flowchart LR
    A[位置更新] --> B[ghost座標更新]
    B --> C[既存neighbor listでPair計算]
    C --> D[force返送]
    D --> E[速度更新]
```

このstepでは粒子migration、spatial binの再作成、neighbor listの全再構築を省ける。

## neighbor listを再構築するstep

```mermaid
flowchart LR
    A[位置更新] --> B[再構築判定]
    B --> C[subdomainを越えた粒子を移動]
    C --> D[ghost粒子を再生成]
    D --> E[spatial binを再構築]
    E --> F[neighbor listを再構築]
    F --> G[Pair計算]
    G --> H[force返送]
```

skinを大きくするとlistに余裕ができ、一般には再構築間隔を延ばせる。一方で各粒子の
候補neighborが増えるため、Pair計算とlist memoryの負担が増える。

```text
大きいskin:
  rebuild回数は減りやすい
  neighbor listは長くなる
  各stepのPair候補数は増える

小さいskin:
  neighbor listは短くなる
  Pair候補数は減る
  rebuildが頻繁になりやすい
```

## balance actionを実行する場合

`balance ... shift x ...`はx方向のMPI subdomain境界を変更する。

```mermaid
flowchart TD
    A[現在のMPI分割] --> B[rank負荷を測定]
    B --> C[x方向の新しい境界を探索]
    C --> D[subdomain境界を変更]
    D --> E[所有rankが変わった粒子をmigration]
    E --> F[ghost粒子を再生成]
    F --> G[neighbor listを再構築]
    G --> H[新しい分割で500 steps実行]
```

balanceが有利になる条件は、action costよりも将来の短縮時間が大きいことである。

```text
balanceが有利
<=>
balance適用コスト
  < balance後に減少する1-step時間 × 効果が続くstep数
```

## 今回の32 MPI ranks / 4 GPU構成

概念的な割当は次の通りである。実際のrank番号とGPUの対応はlauncherのmappingに従う。

```mermaid
flowchart TB
    subgraph MPI[32 MPI ranks]
        R0[ranks 0-7]
        R1[ranks 8-15]
        R2[ranks 16-23]
        R3[ranks 24-31]
    end

    R0 --> G0[GPU 0]
    R1 --> G1[GPU 1]
    R2 --> G2[GPU 2]
    R3 --> G3[GPU 3]

    G0 --- M0[CUDA MPS]
    G1 --- M1[CUDA MPS]
    G2 --- M2[CUDA MPS]
    G3 --- M3[CUDA MPS]
```

CPU版ではrank不均衡が比較的そのままcore間の実行時間差になる。一方、GPU版では
8 ranksの負荷が1 GPU上へ集約される。

```mermaid
flowchart LR
    subgraph CPU版
        A1[rankごとのNlocal / Neighs / Nghost] --> A2[coreごとのPair / Neigh / Comm時間]
        A2 --> A3[最遅rankがstep時間を支配]
    end

    subgraph GPU版
        B1[rankごとのNlocal / Neighs / Nghost] --> B2[同じGPUを使う8 ranksの合計負荷]
        B2 --> B3[GPU kernel効率・MPS共有]
        B3 --> B4[host/device転送・MPI同期]
        B4 --> B5[最遅GPU/rank群がstep時間を支配]
    end
```

## どこで同期するか

```mermaid
sequenceDiagram
    participant R0 as MPI rank 0
    participant G0 as GPU 0
    participant MPI as MPI通信
    participant R1 as 隣接MPI rank

    R0->>G0: 位置更新kernelを起動
    G0-->>R0: 通信対象dataの準備完了
    Note over R0,G0: GPU kernel完了を待つ
    R0->>MPI: host bufferから座標を送信
    R1->>MPI: 隣接rankの座標を送信
    MPI-->>R0: ghost座標を受信
    R0->>G0: 受信dataをGPUへ反映
    R0->>G0: Pair force kernelを起動
    G0-->>R0: force buffer準備完了
    R0->>MPI: ghost forceを返送
    MPI-->>R0: local粒子のforceを受信
    R0->>G0: forceを反映して速度更新
```

主な待ち合わせは次で発生する。

- 粒子migration
- ghost座標のforward communication
- ghost forceのreverse communication
- thermoなどのMPI reduction
- GPU kernelの完了待ち
- host/device transfer
- 同一GPUを共有するrank間のGPU実行待ち

## CPU版とGPU版で最適balanceが一致しない理由

CPU版では、rankごとのneighbor数を均等化することが、Pair計算時間の均等化へ比較的
直接つながる。一方、GPU版ではrank単位の均衡だけでは不十分である。

```mermaid
flowchart TD
    A[balanceでrank負荷を均等化] --> B{GPUごとの合計負荷も<br/>均等になったか}
    B -- Yes --> C{通信・転送増加より<br/>計算短縮が大きいか}
    B -- No --> D[rankは均等でもGPU間不均衡が残る]
    C -- Yes --> E[balanceが有利]
    C -- No --> F[balance costを回収できない]
    D --> F
```

GPU版ではさらに次が性能へ影響する。

- GPUごとの担当粒子・neighbor pairの合計
- 同一GPUを共有する8 ranks間のkernel投入
- kernelサイズとGPU occupancy
- host/device transfer量
- GPU間・rank間の粒子migration
- CPU側MPI通信との同期
- GPU contention時のMPS共有率、clock、power

したがって、CPUで有効なrank-level balanceがGPUでも最適とは限らない。

## 観測すべき状態

CPU/GPUを共通に比較するには、少なくとも次を残す。

| 層 | 状態候補 |
|:---|:---|
| 物理 | temperature、pressure、density、shock位置 |
| rank分割 | `Nlocal`、`Nghost`、neighbor数のmax/mean |
| CPU性能 | Pair、Neigh、Comm時間、rank timing imbalance |
| GPU集約 | GPUごとの担当`Nlocal`・neighbor数・ghost数の合計 |
| GPU性能 | utilization、power、clock、kernel時間、memory使用量 |
| 通信 | migration量、ghost交換量、host/device転送時間 |

特にGPU実験では、rank単位の不均衡だけでなく、rankをGPU単位に集約した不均衡を
状態として扱う必要がある。
