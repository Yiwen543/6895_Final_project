# Nova LLM Quantization Benchmark

| Model        | Type Acc | Cmd Acc | Avg (ms) | P95 (ms) | Size (GB) |
|--------------|----------|---------|----------|----------|-----------|
| 1.5B-Q3_K_M  |     15% |     20% |     5007 |    14051 |      0.8 |
| 1.5B-Q4_K_M  |     30% |      0% |     2728 |     5337 |      1.1 |
| 1.5B-Q5_K_M  |     50% |     80% |     4028 |     6399 |      1.3 |
| 3B-Q3_K_M    |     85% |     80% |     3887 |     7581 |      1.5 |
| 3B-Q4_K_M    |     75% |    100% |     5712 |     9982 |      2.0 |

_Metrics: Type Acc = correct intent type / 20 cases; Cmd Acc = correct device+action / 5 direct_command cases; Avg/P95 latency in ms (3 runs median per case)._
