@echo off
rem Build and run the M0 exp spike.
rem
rem Usage: run.bat [precise^|strict] [avx2] [degree]
rem
rem   precise   /fp:precise (default, the canonical build)
rem   strict    /fp:strict, to show the number does not move
rem   avx2      adds /arch:AVX2; without it MSVC targets SSE2, where fmaf() is
rem             a library call instead of one instruction (see REPORT.md)
rem   degree    polynomial degree for the fit, default 6 (report name gets a
rem             -degN suffix when it differs from 6)
rem
rem The reference sample is regenerated only when ref.bin is missing: delete it
rem to force a full regeneration (~45 s). Coefficients are always regenerated,
rem because a stale coeffs.h is exactly the kind of thing that makes a
rem measurement lie.
setlocal
pushd "%~dp0"

set "FP_MODE=precise"
if /i "%~1"=="strict" set "FP_MODE=strict"
set "ARCH="
if /i "%~2"=="avx2" set "ARCH=/arch:AVX2"
set "DEGREE=%~3"
if "%DEGREE%"=="" set "DEGREE=6"
set "REPORT=report"
if /i "%FP_MODE%"=="strict" set "REPORT=report-strict"
if defined ARCH set "REPORT=%REPORT%-avx2"
if not "%DEGREE%"=="6" set "REPORT=%REPORT%-deg%DEGREE%"
set "REPORT=%REPORT%.json"

rem Both toolchains are discovered, not hardcoded: set VCVARS or NVCC to override.
if not defined VCVARS call "%~dp0..\find_toolchain.bat"
if errorlevel 1 exit /b 1

call "%VCVARS%" >nul
if errorlevel 1 exit /b 1

echo [1/3] coefficients, degree %DEGREE%
python fit_coeffs.py --degree %DEGREE%
if errorlevel 1 exit /b 1

echo [2/3] reference sample
if exist ref.bin goto build
python gen_reference.py
if errorlevel 1 exit /b 1

:build
echo [3/3] build and measure, fp mode %FP_MODE% %ARCH%
cl /nologo /O2 /fp:%FP_MODE% /std:c++17 /W4 /WX /EHsc %ARCH% /DM0_DEGREE=%DEGREE% /Fe:m0_exp.exe measure.cpp
if errorlevel 1 exit /b 1
m0_exp.exe ref.bin manifest.txt %REPORT%
if errorlevel 1 exit /b 1

echo done: %REPORT%
popd
endlocal
