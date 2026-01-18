# Test SGLANG performance with flash attention



## test_fa.py

Example output on Nvidia 3090

```
Test Configuration:
  Single prompt token length: 141
  Batch size: 4
  Max new tokens per sample: 128
  Warm-up runs: 3 | Test runs: 10

Starting warm-up runs (3 runs)...
Warm-up completed

Starting performance testing (10 runs)...
  Run 1/10: 0.7312s | Throughput: 700.23 tokens/s
  Run 2/10: 0.6298s | Throughput: 812.94 tokens/s
  Run 3/10: 0.6300s | Throughput: 812.71 tokens/s
  Run 4/10: 0.6293s | Throughput: 813.55 tokens/s
  Run 5/10: 0.6300s | Throughput: 812.71 tokens/s
  Run 6/10: 0.6323s | Throughput: 809.71 tokens/s
  Run 7/10: 0.6296s | Throughput: 813.22 tokens/s
  Run 8/10: 0.6292s | Throughput: 813.72 tokens/s
  Run 9/10: 0.6321s | Throughput: 810.01 tokens/s
  Run 10/10: 0.6287s | Throughput: 814.42 tokens/s

===== Flash Attention Performance Summary =====
Average run time: 0.6402 seconds
Average throughput: 799.72 tokens/second
Total generated tokens: 5,120.0

Test completed | Resources cleaned up

```

## test_fa_async.py

Example output on Nvidia 3090

```
Test Configuration:
  Single prompt token length: 141
  Batch size: 4
  Max new tokens per sample: 128
  Warm-up runs: 3 | Test runs: 10

Starting asynchronous warm-up runs (3 runs)...
Warm-up completed

Starting asynchronous performance testing (10 runs)...
  Run 1/10: 0.6769s | Throughput: 756.37 tokens/s
  Run 2/10: 0.6397s | Throughput: 800.37 tokens/s
  Run 3/10: 0.6278s | Throughput: 815.53 tokens/s
  Run 4/10: 0.6287s | Throughput: 814.44 tokens/s
  Run 5/10: 0.6278s | Throughput: 815.57 tokens/s
  Run 6/10: 0.6268s | Throughput: 816.86 tokens/s
  Run 7/10: 0.6308s | Throughput: 811.62 tokens/s
  Run 8/10: 0.6306s | Throughput: 811.97 tokens/s
  Run 9/10: 0.6307s | Throughput: 811.86 tokens/s
  Run 10/10: 0.6295s | Throughput: 813.29 tokens/s

===== Async Flash Attention Performance Summary =====
Average run time: 0.6349 seconds
Average throughput: 806.39 tokens/second
Total generated tokens: 5,120.0

Test completed | Resources cleaned up

```