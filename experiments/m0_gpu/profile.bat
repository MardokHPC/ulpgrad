@echo off
rem Nsight Compute probe of the latency-bound sweep (one warm launch per variant).
rem
rem The L2 sweep runs 50 launches per variant (2 rounds x 25), and variant v
rem occupies launches [50v, 50v+49], so --launch-skip 50v+25 is a warm one.
rem Variant ids are the order of the enum in exp_bench.cu:
rem   0 ulp_exp   1 ulp_exp_estrin  2 libdevice_expf  3 fast___expf  4 sfu_ex2_approx
rem   5 fast_p3   6 fast_p4         7 fast_sfu        8 copy
rem
rem Writes stalls.csv (metric rows from ncu's CSV mode, one block per variant).
rem On this machine the counters are readable without admin. The one metric that
rem is not: smsp__warp_issue_stalled_fixed_latency_* reports n/a on sm_75, so the
rem dependency stalls are read from smsp__warp_issue_stalled_wait_* instead.
setlocal
cd /d "%~dp0"
set NCU=C:\Program Files\NVIDIA Corporation\Nsight Compute 2023.1.0\target\windows-desktop-win7-x64\ncu.exe
if not exist "%NCU%" (
  echo ERROR: ncu.exe not found at "%NCU%"
  exit /b 1
)
set "METRICS=smsp__cycles_elapsed.sum,smsp__inst_executed.sum,smsp__inst_executed_pipe_fma.sum,smsp__inst_executed_pipe_alu.sum,smsp__inst_executed_pipe_xu.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct"
if exist stalls.csv del /q stalls.csv
for %%v in (0 1 2 3 4 5 6 7 8) do call :one %%v
echo wrote stalls.csv
endlocal
exit /b 0

:one
set /a SKIP=50*%1+25
echo profiling variant %1 ^(launch-skip %SKIP%^)
"%NCU%" --csv --target-processes all --kernel-name regex:kernel_repeat --launch-skip %SKIP% --launch-count 1 --metrics %METRICS% exp_bench.exe ..\m0_exp\ref.bin "ncu probe" _part_ncu.json 2>nul | findstr /C:"smsp__" >> stalls.csv
exit /b 0
