# GPU Shock state × action characterization

MPI32 / 4 GPUs / CUDA MPS、各cell 5反復。actionはskipまたは`balance ... weight time factor`。

## Runtime matrix

|State|Regime|Best action|Mean s|Std s|CV %|Runner-up gap %|
|---:|---|---|---:|---:|---:|---:|
|S1|idle|time 0.75|1.626204|0.021853|1.34|0.09|
|S1|cpu_contention|time 0.75|1.666258|0.025739|1.54|0.74|
|S1|gpu_contention|time 1|3.134569|0.015296|0.49|0.24|
|S2|idle|time 0.5|1.706205|0.030125|1.77|0.01|
|S2|cpu_contention|time 0.75|1.724632|0.012004|0.70|0.19|
|S2|gpu_contention|time 1|3.242516|0.031546|0.97|0.02|
|S3|idle|time 0.75|1.834655|0.021836|1.19|0.26|
|S3|cpu_contention|time 0.75|1.858118|0.019533|1.05|0.96|
|S3|gpu_contention|time 0.5|3.400486|0.025815|0.76|0.10|
|S4|idle|time 0.5|2.001347|0.007483|0.37|0.26|
|S4|cpu_contention|time 0.75|2.018151|0.024683|1.22|0.58|
|S4|gpu_contention|time 0.5|3.580136|0.022689|0.63|0.51|
|S5|idle|skip|2.059069|0.016621|0.81|0.24|
|S5|cpu_contention|skip|2.048769|0.017066|0.83|1.65|
|S5|gpu_contention|time 0.5|3.624853|0.039280|1.08|0.06|
|S6|idle|skip|2.112000|0.011354|0.54|0.98|
|S6|cpu_contention|time 0.75|2.160674|0.029737|1.38|0.20|
|S6|gpu_contention|skip|3.857588|0.027657|0.72|0.55|
|S7|idle|skip|2.098199|0.010880|0.52|0.67|
|S7|cpu_contention|skip|2.140921|0.019578|0.91|0.18|
|S7|gpu_contention|skip|3.820748|0.025628|0.67|0.57|
|S8|idle|skip|2.063395|0.026520|1.29|0.55|
|S8|cpu_contention|skip|2.124610|0.010988|0.52|0.39|
|S8|gpu_contention|skip|3.895021|0.022498|0.58|0.50|

## Best Fixedと記述的oracle

|Regime|Best Fixed|Fixed total s|Oracle total s|Oracle gain %|
|---|---|---:|---:|---:|
|idle|time 0.5|15.614992|15.501073|0.730|
|cpu_contention|time 0.75|15.794982|15.742134|0.335|
|gpu_contention|time 0.5|28.654344|28.555917|0.343|

oracleは同じ測定値上で選択・評価した記述値であり、未知状態への汎化性能ではない。
