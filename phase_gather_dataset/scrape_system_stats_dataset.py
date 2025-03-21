import psutil
import datetime
import requests  # pip install requests
import time
import os
import json
import csv
import io

# Configuration
NGINX_STATUS_URL = "http://127.0.0.1/stub_status"
OUTPUT_FILE = "/etc/zabbix/templates/host_anomaly_detection/data/system_stats.csv"
STATE_FILE = "/etc/zabbix/templates/host_anomaly_detection/data/.nginx_requests_state.json"  # To store last requests count & timestamp

def get_system_usage():
    """
    Returns overall system CPU and memory usage.
    CPU usage is measured over a short interval.
    """
    cpu_usage = psutil.cpu_percent(interval=0.1)
    mem_usage = psutil.virtual_memory().percent
    return cpu_usage, mem_usage

def get_top_processes_info():
    """
    Uses psutil to retrieve the top 5 processes sorted by CPU and memory usage.
    Returns:
        top_cpu: list of tuples (process_command, cpu_percent)
        top_mem: list of tuples (process_command, mem_percent)
    """
    procs = list(psutil.process_iter(['pid', 'name', 'cmdline']))

    # Initialize per-process CPU percent measurements.
    for proc in procs:
        try:
            proc.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    time.sleep(0.1)

    cpu_info_list = []
    mem_info_list = []
    for proc in procs:
        try:
            cpu = proc.cpu_percent(interval=None)
            mem = proc.memory_percent()
            # Use the full command line if available; otherwise, use the process name.
            cmd = " ".join(proc.info.get('cmdline', [])) if proc.info.get('cmdline') else proc.info.get('name', '')
            cpu_info_list.append((cmd, cpu))
            mem_info_list.append((cmd, mem))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    top_cpu = sorted(cpu_info_list, key=lambda x: x[1], reverse=True)[:5]
    top_mem = sorted(mem_info_list, key=lambda x: x[1], reverse=True)[:5]
    return top_cpu, top_mem

def get_nginx_stub_status(url=NGINX_STATUS_URL):
    """
    Fetch the nginx stub status page and parse:
      - active_connections (int)
      - total_requests (int)  -> cumulative requests
    If there's any error, returns (0, 0).
    """
    try:
        response = requests.get(url, timeout=2)
        response.raise_for_status()
    except requests.RequestException:
        return 0, 0

    text = response.text.strip()
    if not text:
        return 0, 0

    lines = text.split('\n')
    if len(lines) < 3:
        return 0, 0

    active_connections = 0
    total_requests = 0

    try:
        for line in lines:
            if line.startswith("Active connections:"):
                parts = line.split()
                if len(parts) >= 3:
                    active_connections = int(parts[2])
                break
    except (ValueError, IndexError):
        active_connections = 0

    try:
        line_with_requests = lines[2].strip()
        parts = line_with_requests.split()
        if len(parts) == 3:
            total_requests = int(parts[2])
    except (ValueError, IndexError):
        total_requests = 0

    return active_connections, total_requests

def compute_requests_per_second(current_requests):
    """
    Reads the last known requests count and timestamp from the STATE_FILE.
    Returns the computed requests per second and updates the state file
    with the current count and timestamp.
    If there is no prior state, returns 0.0.
    """
    now = time.time()
    if not os.path.isfile(STATE_FILE):
        with open(STATE_FILE, 'w') as f:
            json.dump({"last_requests": current_requests, "last_time": now}, f)
        return 0.0

    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            last_requests = state.get("last_requests", 0)
            last_time = state.get("last_time", now)
    except (json.JSONDecodeError, FileNotFoundError):
        last_requests = current_requests
        last_time = now

    elapsed = now - last_time if now > last_time else 1.0
    diff_requests = current_requests - last_requests
    if diff_requests < 0:
        diff_requests = 0

    rps = diff_requests / elapsed

    with open(STATE_FILE, 'w') as f:
        json.dump({"last_requests": current_requests, "last_time": now}, f)

    return rps

def main():
    # 1. Get overall system usage.
    cpu_usage, mem_usage = get_system_usage()

    # 2. Get top 5 processes by CPU and memory usage.
    top_cpu, top_mem = get_top_processes_info()

    # 3. Get NGINX status.
    nginx_active_connections, nginx_total_requests = get_nginx_stub_status()
    nginx_requests_ps = compute_requests_per_second(nginx_total_requests)

    # Prepare CSV fields.
    # Flatten top CPU info into name and usage pairs.
    top_cpu_fields = []
    for i in range(5):
        if i < len(top_cpu):
            proc_name, proc_val = top_cpu[i]
            proc_val = f"{proc_val:.2f}"
        else:
            proc_name, proc_val = "", ""
        top_cpu_fields.extend([proc_name, proc_val])

    # Flatten top Memory info into name and usage pairs.
    top_mem_fields = []
    for i in range(5):
        if i < len(top_mem):
            proc_name, proc_val = top_mem[i]
            proc_val = f"{proc_val:.2f}"
        else:
            proc_name, proc_val = "", ""
        top_mem_fields.extend([proc_name, proc_val])

    now_str = datetime.datetime.now().isoformat()

    # Compose the CSV row items:
    # [timestamp; overall_cpu; (top 5 CPU process names & usages); overall_mem; (top 5 mem process names & usages);
    #  nginx_active_connections; nginx_requests_ps]
    row_items = [
        now_str,
        f"{cpu_usage:.2f}",
        *top_cpu_fields,  # 10 fields (5 name,usage pairs)
        f"{mem_usage:.2f}",
        *top_mem_fields,  # 10 fields (5 name,usage pairs)
        str(nginx_active_connections),
        f"{nginx_requests_ps:.2f}"
    ]

    # Use the csv module to generate a CSV line that properly quotes fields containing semicolons.
    output = io.StringIO()
    csv_writer = csv.writer(output, delimiter=";", quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(row_items)
    csv_line = output.getvalue()
    output.close()

    # Print the CSV line to the console and append it to the output file.
    print(csv_line, end="")
    with open(OUTPUT_FILE, "a", newline="") as f:
        f.write(csv_line)

if __name__ == "__main__":
    main()