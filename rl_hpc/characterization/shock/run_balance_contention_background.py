#!/usr/bin/env python3
"""File-logged supervisor for the approved matched contention experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = Path(__file__).resolve().parent
CHILD = None


def terminate_tree(process):
    try:
        tree = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        tree = []
    for item in tree:
        try:
            item.terminate()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.terminate()
    _, alive = psutil.wait_procs(tree, timeout=5)
    for item in alive:
        try:
            item.kill()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=5)
    _, alive = psutil.wait_procs(alive, timeout=5)
    if any(p.status() != psutil.STATUS_ZOMBIE for p in alive):
        raise RuntimeError('Residual experiment process after cleanup')


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def rows_at(directory):
    file = directory/'measurements.jsonl'
    lines = file.read_text().splitlines() if file.exists() else []
    result = []
    for i, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1:
                raise
    return result


def run_stage(command, directory, stage, status_path, state, expected):
    global CHILD
    with (directory/f'{stage}_driver.log').open('a') as log:
        CHILD = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        count = 0
        progress = time.monotonic()
        while True:
            rows = rows_at(directory)
            if len(rows) != count:
                count = len(rows)
                progress = time.monotonic()
            state.update(stage=stage, completed=count, planned=expected,
                         safe_completed=sum(r['safe'] for r in rows),
                         last_key=rows[-1]['key'] if rows else None,
                         runner_pid=CHILD.pid, updated=time.time())
            save(status_path, state)
            if any(not r['safe'] for r in rows):
                raise RuntimeError('Unsafe trial; all further stages stopped')
            rc = CHILD.poll()
            if rc is not None:
                state['stage_returncode'] = rc
                if rc != 0 or len(rows) != expected:
                    raise RuntimeError(f'{stage} failed: returncode={rc}, trials={len(rows)}')
                CHILD = None
                return
            if time.monotonic()-progress > 240:
                raise RuntimeError(f'{stage}: no completed trial for 240 seconds; stopped')
            time.sleep(2)


def make_report(main):
    docs = ROOT/'rl_hpc/docs/experiments/2026-10-08_ampb'
    analysis = main/'analysis'
    a = json.loads((analysis/'analysis.json').read_text())
    shutil.copyfile(analysis/'MATRIX.md', docs/'CONTENTION_MATRIX_V3.md')
    for name in ('contention_runtimes', 'contention_imbalance', 'contention_pca'):
        shutil.copyfile(analysis/f'{name}.png', docs/f'{name}_v3.png')
    shutil.copyfile(analysis/'pca_metadata.json', docs/'contention_pca_v3_metadata.json')
    lines = ['# CPU contention比較：監督付きバックグラウンド再測定v3', '',
             '同じ8 checkpoint × 6 actions × idle/contention × 各5回。各反復の96条件をseed20261008でshuffleし、MPI32/OMP1、skin0.5、500 stepsを順次実行した。共通weight neigh1.0のpartition再構成は計測外、contentionは既存RL CpuJitterのCPU0–7の8 workersを計測中のみactiveにした。', '',
             'smoke12試行とmain480試行が安全に完了。試行の安全性・外部負荷監視・checkpoint SHAはrunnerで確認し、停止時は後続段階を実行しない。新しい監督プロセスは240秒進行なしでも停止する。元のfork、RL負荷実装と実験条件は変更していない。', '',
             '[全5測定値・mean/std/CV・報酬・次状態の完全matrix](CONTENTION_MATRIX_V3.md)。CIは同じ反復ブロック差の無補正df4 t区間であり、48比較を同時に保証しない。Pooled分布は状態/action間の違いを含む。', '',
             '|量|idle mean|contention mean|変化 %|','|---|---:|---:|---:|']
    if list(main.glob('resume_audit-*.json')):
        lines[2:2] = ['VSCodeの一時的CPU活動による監視停止後、ユーザーの再開指示に従って安全な283試行を保持し、残り197試行を元のランダム順で継続した。旧manifest・中断ログとresume_auditを保持。計測条件の変更はなく、監視とrunnerのSHAのみ履歴として記録する。VSCode extensionHostは約10秒の累積2.5 CPU秒超で停止、それ以外の外部ユーザープロセスは従来の0.5 CPU秒/checkで停止する。日をまたいだ測定の時間変動を含むため、細かい順位差は慎重に解釈する。', '']
    for metric in ('runtime_seconds','reward','atom_imbalance','neighbor_imbalance','ghost_imbalance','pair_per_step','neigh_per_step','comm_per_step'):
        x=a['equally_weighted_mixture']['idle'][metric]['mean'];y=a['equally_weighted_mixture']['contention'][metric]['mean']
        lines.append(f'|{metric}|{x:.9g}|{y:.9g}|{(y-x)/abs(x)*100:+.3f}|')
    label=lambda f:'skip' if f is None else f'{f:.2f}'
    lines += ['', '|State|idle mean-best|runner-up gap s|contention mean-best|runner-up gap s|','|---|---|---:|---|---:|']
    for phase in range(1,9):
        fields=[]
        for regime in ('idle','contention'):
            cs=sorted([c for c in a['cells'] if c['phase']==phase],key=lambda c:c['statistics'][regime]['runtime_seconds']['mean'])
            fields += [label(cs[0]['factor']),f"{cs[1]['statistics'][regime]['runtime_seconds']['mean']-cs[0]['statistics'][regime]['runtime_seconds']['mean']:.6f}"]
        lines.append(f'|S{phase}|'+'|'.join(fields)+'|')
    lines += ['', '|Regime|Best Fixed|8-state mean sum s|Oracle s|差 s|削減率 %|','|---|---|---:|---:|---:|---:|']
    for regime in ('idle','contention'):
        totals=[(f,sum(c['statistics'][regime]['runtime_seconds']['mean'] for c in a['cells'] if c['factor']==f)) for f in (None,.5,.75,1,1.25,1.5)]
        best=min(totals,key=lambda t:t[1]);oracle=sum(min(c['statistics'][regime]['runtime_seconds']['mean'] for c in a['cells'] if c['phase']==p) for p in range(1,9))
        lines.append(f'|{regime}|{label(best[0])}|{best[1]:.6f}|{oracle:.6f}|{best[1]-oracle:.6f}|{(best[1]-oracle)/best[1]*100:.3f}|')
    lines += ['', 'oracleは同じデータから選択・評価した記述値。学習方策の性能や不偏な汎化性能ではない。各条件5回では分布の裾や小さい差を精密に評価できない。今回の比較は同一checkpoint/actionからの条件付き次状態であり、連続軌跡の訪問状態分布ではない。', '',
              '![全5測定のruntime](contention_runtimes_v3.png)', '',
              '![次状態不均衡](contention_imbalance_v3.png)', '',
              '![同じidle-fit軸でのPCA](contention_pca_v3.png)', '',
              '環境・正確なbuild設定は[ampb報告](README.md)を参照。先行する中断データは混ぜずに保存。v3元データは `rl_hpc/characterization/data/portable_balance_contention_main_v3/`、監督statusは `rl_hpc/characterization/data/portable_balance_contention_background_v3/status.json`。生成データ等をGitへ追加していない。']
    (docs/'CONTENTION_V3.md').write_text('\n'.join(lines)+'\n')
    index=docs/'CONTENTION.md'
    text=index.read_text()
    index.write_text('[監督付き再測定v3の完了結果](CONTENTION_V3.md)（各5回、main480試行）。以下は先行実行の記録。\n\n'+text)


def main():
    global CHILD
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    status_path=output/'status.json'
    if status_path.exists():
        if not args.resume:
            raise RuntimeError('existing status requires explicit resume')
        shutil.copyfile(status_path,output/f'status.before-resume-{time.time_ns()}.json')
    state=dict(status='running',supervisor_pid=os.getpid(),started=time.time())
    save(status_path,state)
    try:
        for stage,expected in [('smoke',12),('main',480)]:
            directory=ROOT/f'rl_hpc/characterization/data/portable_balance_contention_{stage}_v3'
            directory.mkdir(exist_ok=True)
            if args.resume and stage == 'smoke':
                rows=rows_at(directory)
                if len(rows)!=12 or any(not r['safe'] or r['returncode']!=0 for r in rows):
                    raise RuntimeError('previous smoke is incomplete or unsafe')
                continue
            command=[sys.executable,str(SCRIPTS/'fork_balance_contention.py'),
                     '--checkpoints',str(ROOT/'rl_hpc/characterization/data/portable_representative_checkpoints_v1'),
                     '--output',str(directory),'--phase',stage,'--repetitions','5',
                     '--seed','20261008','--cpu-cores','0-7','--monitor-host']
            if args.resume:
                command.append('--resume')
            run_stage(command,directory,stage,status_path,state,expected)
        state.update(stage='analysis',updated=time.time());save(status_path,state)
        main_dir=ROOT/'rl_hpc/characterization/data/portable_balance_contention_main_v3'
        env={**os.environ,'MPLCONFIGDIR':'/tmp/ampb-contention-matplotlib'}
        with (output/'analysis_driver.log').open('w') as log:
            rc=subprocess.run([sys.executable,str(SCRIPTS/'analyze_balance_contention.py'),
                               '--input',str(main_dir),'--output',str(main_dir/'analysis')],
                              cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
        if rc != 0:raise RuntimeError(f'Analysis returncode {rc}')
        make_report(main_dir)
        state.update(status='complete',stage='complete',updated=time.time(),report=str(ROOT/'rl_hpc/docs/experiments/2026-10-08_ampb/CONTENTION_V3.md'))
    except BaseException as exc:
        state.update(status='stopped',error=f'{type(exc).__name__}: {exc}',updated=time.time())
        raise
    finally:
        if CHILD is not None:
            terminate_tree(CHILD)
        save(status_path,state)


if __name__=='__main__':
    signal.signal(signal.SIGINT,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
