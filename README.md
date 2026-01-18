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