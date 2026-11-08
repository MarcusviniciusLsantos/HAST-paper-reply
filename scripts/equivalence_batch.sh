#!/usr/bin/env bash
# Python-vs-Java equivalence batch with 100-member ensembles.
#   DATASETS="covtypeNorm poker-lsn" N=10000 P=2 ./scripts/equivalence_batch.sh
cd "$(dirname "$0")/.."
N=${N:-10000}
DATASETS=${DATASETS:-"covtypeNorm poker-lsn INSECTS-abrupt_balanced airlines"}
OUTF=results/equivalence/batch_100_N${N}.txt
jobs=()
for d in $DATASETS; do
  for e in LevBag ARF SRP ARTE; do
    for t in HT HLAST HLAST_D; do
      jobs+=("$e $t data/real/$d.arff $N 100")
    done
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P ${P:-2} -L 1 bash -c '.venv/bin/python -W ignore scripts/check_ensemble.py "$0" "$1" "$2" "$3" "$4"' >> "$OUTF" 2>&1
echo DONE >> "$OUTF"
