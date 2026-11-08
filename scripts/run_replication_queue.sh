#!/usr/bin/env bash
# Full replication on this machine: 4 lanes x 1 worker (measured optimum for the
# i5-12450H laptop: 4 concurrent runs scale ~3.6x, 6 runs are slower in total).
# Lanes hold disjoint jobs; every run is resumable (finished JSONs are skipped).
#
#   nohup setsid ./scripts/run_replication_queue.sh > results/raw/queue.log 2>&1 &
#
# Synthetic jobs are split in 7 interleaved shards: lanes L1-L3 take two each,
# the heavy lane H takes one after Airlines/Poker/CovType (balances finish times).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TREES="HT HLAST HLAST_D"
ALL="ARTE SRP ARF LevBag ARE"

run() { echo "=== $(date '+%F %T') [$LANE] $*"; $PY scripts/run_experiments.py --trees $TREES -j 1 "$@"; }

( LANE=H
  run --datasets Airlines --ensembles $ALL --fresh-workers
  run --datasets Poker CovType --ensembles ARF LevBag ARE
  run --datasets synthetic --ensembles $ALL --shard 6/7
  echo "=== $(date '+%F %T') [H] done" ) &

( LANE=L1
  run --datasets Rialto INSECTSi INSECTSa --ensembles ARF LevBag ARE
  run --datasets synthetic --ensembles $ALL --shard 0,1/7
  echo "=== $(date '+%F %T') [L1] done" ) &

( LANE=L2
  run --datasets synthetic --ensembles $ALL --shard 2,3/7
  echo "=== $(date '+%F %T') [L2] done" ) &

( LANE=L3
  run --datasets synthetic --ensembles $ALL --shard 4,5/7
  echo "=== $(date '+%F %T') [L3] done" ) &

wait
echo "=== $(date '+%F %T') queue finished"
