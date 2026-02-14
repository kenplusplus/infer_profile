import re
import argparse
from typing import List, Tuple, Optional

def extract_throughput_values(log_content: str) -> Tuple[List[float], List[str]]:
    """
    Extract gen throughput values and corresponding log lines from log content

    Args:
        log_content: String containing the log lines

    Returns:
        Tuple of (list of throughput values, list of corresponding log lines)
    """
    # Regular expression pattern to match "gen throughput (token/s): XXXX.XX" and capture the value
    pattern = r'(.*gen throughput \(token/s\): (\d+\.\d+).*)'

    # Find all matches in the log content
    matches = re.findall(pattern, log_content)

    # Separate values and lines
    throughput_values = []
    corresponding_lines = []

    for full_line, value_str in matches:
        try:
            value = float(value_str)
            throughput_values.append(value)
            corresponding_lines.append(full_line.strip())
        except ValueError:
            # Skip invalid numeric values (defensive programming)
            continue

    return throughput_values, corresponding_lines

def calculate_statistics(values: List[float]) -> Tuple[float, float, float, int, float, float, Optional[int]]:
    """
    Calculate comprehensive statistics for throughput values

    Args:
        values: List of throughput values

    Returns:
        Tuple of (average, total_sum, std_dev, count, max_value, min_value, max_index)
    """
    if not values:
        return 0.0, 0.0, 0.0, 0, 0.0, 0.0, None

    count = len(values)
    total_sum = sum(values)
    average = total_sum / count

    # Calculate standard deviation
    variance = sum((x - average) **2 for x in values) / count
    std_dev = variance **0.5

    # Calculate max/min values and max index
    max_value = max(values)
    min_value = min(values)
    max_index = values.index(max_value)

    return average, total_sum, std_dev, count, max_value, min_value, max_index

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Extract and calculate gen throughput statistics from logs (with max/min values)')
    parser.add_argument('--input', '-i', type=str, help='Path to log file (if not provided, uses sample log in script)')
    parser.add_argument('--output', '-o', type=str, help='Path to save results (optional)')
    args = parser.parse_args()

    # Sample log content (your provided logs)
    sample_log = """
[2026-02-14 10:47:39] Decode batch. #running-req: 32, #token: 175104, token usage: 0.60, cuda graph: False, gen throughput (token/s): 1077.81, #queue-req: 16,
[2026-02-14 10:47:40] Decode batch. #running-req: 32, #token: 177152, token usage: 0.61, cuda graph: False, gen throughput (token/s): 1073.58, #queue-req: 16,
[2026-02-14 10:47:41] Decode batch. #running-req: 32, #token: 179200, token usage: 0.62, cuda graph: False, gen throughput (token/s): 1068.62, #queue-req: 16,
[2026-02-14 10:47:42] Decode batch. #running-req: 32, #token: 179200, token usage: 0.62, cuda graph: False, gen throughput (token/s): 1064.84, #queue-req: 16,
[2026-02-14 10:47:44] Decode batch. #running-req: 32, #token: 181248, token usage: 0.63, cuda graph: False, gen throughput (token/s): 1060.84, #queue-req: 16,
[2026-02-14 10:47:45] Decode batch. #running-req: 32, #token: 181248, token usage: 0.63, cuda graph: False, gen throughput (token/s): 1057.22, #queue-req: 16,
[2026-02-14 10:47:46] Decode batch. #running-req: 32, #token: 183296, token usage: 0.63, cuda graph: False, gen throughput (token/s): 1052.22, #queue-req: 16,
[2026-02-14 10:47:47] Decode batch. #running-req: 32, #token: 185344, token usage: 0.64, cuda graph: False, gen throughput (token/s): 1048.41, #queue-req: 16,
[2026-02-14 10:47:48] Decode batch. #running-req: 32, #token: 185344, token usage: 0.64, cuda graph: False, gen throughput (token/s): 1044.06, #queue-req: 16,
[2026-02-14 10:47:50] Decode batch. #running-req: 32, #token: 187392, token usage: 0.65, cuda graph: False, gen throughput (token/s): 1040.11, #queue-req: 16,
[2026-02-14 10:47:51] Decode batch. #running-req: 32, #token: 189440, token usage: 0.65, cuda graph: False, gen throughput (token/s): 1036.88, #queue-req: 16,
[2026-02-14 10:47:52] Decode batch. #running-req: 32, #token: 189440, token usage: 0.65, cuda graph: False, gen throughput (token/s): 1031.88, #queue-req: 16,
[2026-02-14 10:47:53] Decode batch. #running-req: 32, #token: 191488, token usage: 0.66, cuda graph: False, gen throughput (token/s): 1028.10, #queue-req: 16,
[2026-02-14 10:47:55] Decode batch. #running-req: 32, #token: 191488, token usage: 0.66, cuda graph: False, gen throughput (token/s): 1024.48, #queue-req: 16,
[2026-02-14 10:47:56] Decode batch. #running-req: 32, #token: 193536, token usage: 0.67, cuda graph: False, gen throughput (token/s): 1020.09, #queue-req: 16,
[2026-02-14 10:47:57] Decode batch. #running-req: 32, #token: 195584, token usage: 0.67, cuda graph: False, gen throughput (token/s): 1016.69, #queue-req: 16,
[2026-02-14 10:47:58] Decode batch. #running-req: 32, #token: 195584, token usage: 0.67, cuda graph: False, gen throughput (token/s): 1012.40, #queue-req: 16,
[2026-02-14 10:48:00] Decode batch. #running-req: 32, #token: 197632, token usage: 0.68, cuda graph: False, gen throughput (token/s): 1008.92, #queue-req: 16,
[2026-02-14 10:48:01] Decode batch. #running-req: 32, #token: 199680, token usage: 0.69, cuda graph: False, gen throughput (token/s): 1004.10, #queue-req: 16,
    """

    # Read log content (from file or sample)
    if args.input:
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                log_content = f.read()
            print(f"✅ Successfully read log file: {args.input}")
        except Exception as e:
            print(f"❌ Error reading log file: {e}")
            return
    else:
        log_content = sample_log
        print("ℹ️ Using sample log content from script")

    # Extract throughput values and corresponding lines
    throughput_values, corresponding_lines = extract_throughput_values(log_content)

    if not throughput_values:
        print("❌ No gen throughput values found in log content")
        return

    # Calculate comprehensive statistics
    average, total_sum, std_dev, count, max_value, min_value, max_index = calculate_statistics(throughput_values)

    # Get log line for max value (if available)
    max_line = corresponding_lines[max_index] if max_index is not None else "N/A"

    # Print detailed results
    print("\n" + "="*80)
    print("📊 Gen Throughput Comprehensive Statistics")
    print("="*80)
    print(f"Total number of samples:       {count}")
    print(f"Sum of all values:             {total_sum:.2f} token/s")
    print(f"Average throughput:            {average:.2f} token/s")
    print(f"Standard deviation:            {std_dev:.2f} token/s")
    print(f"🔺 Maximum throughput:         {max_value:.2f} token/s")  # 重点标注最大值
    print(f"🔻 Minimum throughput:         {min_value:.2f} token/s")
    print("="*80)
    print(f"📝 Log line with maximum value:\n{max_line}")
    print("="*80)
    print("\n📋 All extracted values (sorted descending):")
    # Sort values with their original indices to keep traceability
    sorted_values = sorted(enumerate(throughput_values), key=lambda x: x[1], reverse=True)
    for idx, val in sorted_values:
        print(f"  {val:.2f} token/s (log line {idx+1})")

    # Save results to file if specified
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write("Gen Throughput Comprehensive Statistics\n")
                f.write("="*80 + "\n")
                f.write(f"Total number of samples:       {count}\n")
                f.write(f"Sum of all values:             {total_sum:.2f} token/s\n")
                f.write(f"Average throughput:            {average:.2f} token/s\n")
                f.write(f"Standard deviation:            {std_dev:.2f} token/s\n")
                f.write(f"Maximum throughput:            {max_value:.2f} token/s\n")
                f.write(f"Minimum throughput:            {min_value:.2f} token/s\n")
                f.write("="*80 + "\n")
                f.write(f"Log line with maximum value:\n{max_line}\n")
                f.write("="*80 + "\n")
                f.write("\nAll extracted values (sorted descending):\n")
                for idx, val in sorted_values:
                    f.write(f"  {val:.2f} token/s (log line {idx+1})\n")
            print(f"\n✅ Results saved to: {args.output}")
        except Exception as e:
            print(f"\n❌ Error saving results: {e}")

if __name__ == "__main__":
    main()