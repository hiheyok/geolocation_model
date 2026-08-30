# Overnight run: data scale vs epochs

Selection criterion is greedy-decode **val median km**, not val loss. Beam ranking is on s0-s2 (`--score-steps 3`), the shipping depth.


## Arms

| arm | release | images | epochs | steps | selected ep | val km (greedy, sel) | test km k=2 | test <25km | s/epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Beam width sweep (val)

| arm | k=1 | k=2 | k=4 |
|---|---:|---:|---:|

## System utilisation by stage

| stage | samples | GPU % (peak) | VRAM MB | shared MB | W | degC | CPU % | RAM GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| embed_dino | 34 | 87 (100) | 4705 | 231 | 183 | 60 | 25 | 18.9 |
| embed_siglip | 12 | 75 (100) | 4123 | 240 | 197 | 59 | 27 | 18.3 |

`shared MB` is GPU memory that spilled into system DRAM. The desktop baseline here is 230-290 MB; materially above that means a process is over-committed and every access is crossing PCIe while nvidia-smi still reports 100% utilisation.


## Stage timings

| stage | minutes |
|---|---:|
