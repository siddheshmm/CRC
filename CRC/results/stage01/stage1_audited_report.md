# Stage 1 Flow Baseline & Hydraulic Audit Report
**Dataset:** `stage1_baseline/baseline` (Started 2026-10-27)

## 1. Measured Physical Hydraulics
- **Measured Flow Rate ($Q$):** **4.00 mL/min** (gravimetric runs: [4.1, 3.9] mL/min)
- **Vial Volume ($V$):** **10.0 mL**
- **Theoretical Residence Time ($	au = V/Q$):** **2.50 min** (150.0 s)
- **Empirical Step-Relaxation $	au$:** **1.99 min** (119.4 s)
- **Effective Mixing Volume ($V_{\text{eff}}$):** **7.96 mL** (79.6% chamber efficiency)
- **Rule 4 Target Step Length ($T_s = \tau / 3$):** **39.8 s to 50.0 s** (Nominal $60.0\text{ s}$ was a protocol deviation)

## 2. Gate Evaluations
| Criterion | Gate Limit | Measured Value | Result |
| :--- | :--- | :--- | :--- |
| Window Stability (1.47 - 1.51) | $\ge 30\text<built-in function min>$ continuous | 63.0 min (99.3% in window) | **PASS** |
| Overall Drift Rate | $\le 0.20\% / 10\text<built-in function min>$ | $-0.025 \pm 0.006\% / 10\text<built-in function min>$ | **PASS** |
| 10h Run Drift Guideline | $\le 0.05\% / 10\text<built-in function min>$ | $-0.025\% / 10\text<built-in function min>$ | **PASS** |
| **Recent Drift Acceleration (Last 20 min)** | $\le 0.05\% / 10\text<built-in function min>$ | **$-0.060 \pm 0.029\% / 10\text<built-in function min>$ (curvature $p = 6 \times 10^{-7}$)** | **FAIL** |
| Measurement Noise $\sigma$ | $\le 0.003$ | Median $0.0017$, worst bin $0.0021$ | **PASS** |

## 3. Inter-Session Carrier Baseline Shifts
| Session / Stage | Date | Median $F_0$ (555/480 nm) | Within-Run Noise $\sigma$ | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 0 (Carrier Check)** | 2026-10-26 | 1.5045 | 0.00097 | Fresh buffer batch 1 |
| **Stage 1 (Flow Baseline)** | 2026-10-27 | 1.4740 | 0.00170 | Continuous flow baseline |
| **Stage 4 Run 1** | 2026-10-28 | 1.4809 | 0.00160 | Benchmark run 1 |
| **Stage 6 (Hold-out)** | 2026-10-29 | 1.4842 | 0.00150 | Seed 43 hold-out run |
| **Stage 5 (ESP Run)** | 2026-10-30 | 1.4866 | 0.00130 | 7.5h slug contraction run |

*Note: Per-run standardization masks these inter-day baseline shifts ($1.474$ to $1.505$). A true zero-shot model trained on Stage 4 and deployed across days must account for this $\Delta F_0 \approx 0.012$ ($7\sigma$) offset.*
