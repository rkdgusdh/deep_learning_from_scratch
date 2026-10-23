#!/bin/bash
# 사용법: ./run_queue.sh queue_file  (한 줄에 train.py 인자 한 세트, 순서대로 실행)
cd "$(dirname "$0")"
mkdir -p logs
export VECLIB_MAXIMUM_THREADS=${THREADS:-4}
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  name=$(echo "$line" | sed -E 's/.*--name ([^ ]+).*/\1/')
  if grep -q "^$name," results.csv 2>/dev/null; then echo "skip $name"; continue; fi
  eval ${PYTHON:-python3} train.py $line > logs/$name.log 2>&1
done < "$1"
