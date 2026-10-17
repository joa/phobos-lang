"""Side-by-side metrics from ncu reports exported as raw CSV.

    python scripts/ncu_compare.py NAME=REPORT.csv...

Each CSV is `ncu --import R.ncu-rep --page raw --csv --print-units base`.
Prints the metrics that separate a latency-bound GEMM from an issue-, shared-
or DRAM-bound one, one column per report.
"""

import csv
import sys

METRICS = [
    ("duration us", "gpu__time_duration.sum", 1e-3),
    ("SM clock MHz", "smsp__cycles_elapsed.avg.per_second", 1e-6),
    ("registers/thread", "launch__registers_per_thread", 1),
    ("static smem B/block", "launch__shared_mem_per_block_static", 1),
    ("blocks/SM limit (regs)", "launch__occupancy_limit_registers", 1),
    ("achieved occupancy %", "sm__warps_active.avg.pct_of_peak_sustained_active", 1),
    ("waves/SM", "launch__waves_per_multiprocessor", 1),
    ("issue active %", "sm__issue_active.avg.pct_of_peak_sustained_elapsed", 1),
    ("block size", "launch__block_size", 1),
    ("grid size", "launch__grid_size", 1),
    ("FMA pipe %", "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active", 1),
    ("eligible warps/cycle", "smsp__warps_eligible.avg.per_cycle_active", 1),
    ("active warps/scheduler", "smsp__warps_active.avg.per_cycle_active", 1),
    ("all inst (M)", "smsp__inst_executed.sum", 1e-6),
    ("stall long scoreboard", "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", 1),
    ("stall barrier", "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio", 1),
    ("stall MIO throttle", "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio", 1),
    ("stall short scoreboard", "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio", 1),
    ("stall math pipe throttle", "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", 1),
    ("stall not selected", "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio", 1),
    ("stall selected", "smsp__average_warps_issue_stalled_selected_per_issue_active.ratio", 1),
    ("stall wait", "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio", 1),
    ("stall dispatch", "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio", 1),
    ("stall LG throttle", "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio", 1),
    ("stall no instruction", "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio", 1),
    ("shared ld wavefronts (M)", "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", 1e-6),
    ("shared ld bank conflicts (M)", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", 1e-6),
    ("shared st wavefronts (M)", "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum", 1e-6),
    ("shared st bank conflicts (M)", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", 1e-6),
    ("LSU shared pipe %", "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed", 1),
    ("MIO issue %", "sm__mio_inst_issued.avg.pct_of_peak_sustained_elapsed", 1),
    ("L1 hit %", "l1tex__t_sector_hit_rate.pct", 1),
    ("L2 hit %", "lts__t_sector_hit_rate.pct", 1),
    ("DRAM read MB", "dram__bytes_read.sum", 1e-6),
    ("DRAM read % of peak", "dram__bytes_read.sum.pct_of_peak_sustained_elapsed", 1),
    ("local ld/st bytes", "l1tex__t_bytes_pipe_lsu_mem_local.sum", 1),
]


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    # Header row, a units row, then one row per kernel.
    header, values = rows[0], rows[-1]
    return dict(zip(header, values))


def fmt(v, scale):
    try:
        x = float(v.replace(",", "")) * scale
    except (ValueError, AttributeError):
        return v if v else "-"
    return f"{x:.0f}" if abs(x) >= 1000 else f"{x:.3g}"


def main(args):
    cols = [a.split("=", 1) for a in args]
    data = [(name, load(path)) for name, path in cols]
    print("| metric | " + " | ".join(n for n, _ in data) + " |")
    print("|---|" + "---|" * len(data))
    for label, key, scale in METRICS:
        print(f"| {label} | " + " | ".join(fmt(d.get(key, ""), scale) for _, d in data) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
