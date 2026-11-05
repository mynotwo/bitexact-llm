# source: pick a free GPU among 0-3 (0 MiB, no compute process); prints it
free() { [ "$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" = 0 ] &&
         [ "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -c .)" = 0 ]; }
pick() { while true; do for g in 0 1 2 3; do if free $g; then sleep 2; free $g && { echo $g; return; }; fi; done; sleep 60; done; }
