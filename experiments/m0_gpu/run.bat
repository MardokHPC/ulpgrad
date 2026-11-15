@echo off
rem Build and run the M0 GPU spike.
rem
rem Three builds of the same file, because the two flags that matter are
rem compile-time:
rem
rem   ftz=true  fmad=true    CUDA defaults, the canonical build
rem   ftz=false fmad=true    subnormals honoured, to see what FTZ costs
rem   ftz=true  fmad=false   contraction off, to prove the 1 ulp bound does not
rem                          depend on the compiler fusing anything (all the
rem                          products here are explicit __fmaf_rn already)
rem
rem Usage: run.bat            all three, then merge the parts into report.json
rem        run.bat merge      only merge existing parts (no rebuild)
rem
rem Prerequisite: ..\m0_exp\ref.bin (the 1,009,866-point reference sample).
rem Run ..\m0_exp\run.bat once if it is missing; it is not committed.
setlocal
pushd "%~dp0"

rem nvcc 12.1 cannot compile this: MSVC 14.44's STL refuses CUDA < 12.4 outright
rem (yvals_core.h: "error STL1002: Unexpected compiler version, expected CUDA
rem 12.4 or newer"), and --allow-unsupported-compiler does not reach that guard.
rem Discovery picks the newest toolkit installed; set NVCC to force another one.
if not defined NVCC call "%~dp0..\find_toolchain.bat"
if errorlevel 1 exit /b 1
if not exist "%NVCC%" (
  echo ERROR: nvcc not found. Install a CUDA toolkit, or set NVCC to its full path.
  exit /b 1
)

set "REF=..\m0_exp\ref.bin"
if not exist "%REF%" (
  echo ERROR: %REF% is missing. Run experiments\m0_exp\run.bat once to generate it.
  exit /b 1
)

if /i "%~1"=="merge" goto merge

call "%VCVARS%" >nul
if errorlevel 1 exit /b 1

rem MSVC 14.44 is newer than nvcc 12.1 knows about; without this nvcc refuses
rem to start.
set "NVCC_PREPEND_FLAGS=--allow-unsupported-compiler"

rem -Xcompiler takes a comma-separated list, not a space-separated one.
set "NVFLAGS=-arch=sm_75 -std=c++17 -O3 -Xcompiler /O2,/W4,/EHsc"

echo [1/4] build ftz=true  fmad=true  ^(canonical^)
"%NVCC%" %NVFLAGS% -ftz=true  -fmad=true  -o exp_bench.exe            exp_bench.cu
if errorlevel 1 exit /b 1

echo [2/4] build ftz=false fmad=true
"%NVCC%" %NVFLAGS% -ftz=false -fmad=true  -o exp_bench_ftzfalse.exe   exp_bench.cu
if errorlevel 1 exit /b 1

echo [3/4] build ftz=true  fmad=false
"%NVCC%" %NVFLAGS% -ftz=true  -fmad=false -o exp_bench_fmadfalse.exe  exp_bench.cu
if errorlevel 1 exit /b 1

echo [4/4] run
exp_bench.exe           "%REF%" "ftz=true  fmad=true"  _part_ftz_true.json  > run-ftz-true.log 2>&1
if errorlevel 1 (
  type run-ftz-true.log
  exit /b 1
)
type run-ftz-true.log

exp_bench_ftzfalse.exe  "%REF%" "ftz=false fmad=true"  _part_ftz_false.json > run-ftz-false.log 2>&1
if errorlevel 1 (
  type run-ftz-false.log
  exit /b 1
)

exp_bench_fmadfalse.exe "%REF%" "ftz=true  fmad=false" _part_fmad_false.json > run-fmad-false.log 2>&1
if errorlevel 1 (
  type run-fmad-false.log
  exit /b 1
)

:merge
exp_bench.exe --merge report.json _part_ftz_true.json _part_ftz_false.json _part_fmad_false.json
if errorlevel 1 exit /b 1
del /q _part_*.json >nul 2>&1

echo done: report.json ^(+ run-ftz-true.log, run-ftz-false.log, run-fmad-false.log^)
popd
endlocal
