@echo off
rem Build and run the device-side check. Needs a CUDA toolkit; needs a GPU to run,
rem but only nvcc to compile. CI uses the compile half of this.
rem
rem The golden reference directory is the first argument and it is required by
rem the program: this script only defaults it to the copy inside the repository,
rem relative to its own directory.
rem
rem Set VCVARS or NVCC to override the discovery in ..\find_toolchain.bat.
setlocal
pushd "%~dp0"
set "REFERENCE=%~1"
if not defined REFERENCE set "REFERENCE=..\..\data\reference"
if not defined VCVARS call "%~dp0..\..\experiments\find_toolchain.bat"
if errorlevel 1 exit /b 1
if not exist "%NVCC%" (
  echo ERROR: nvcc not found. Install a CUDA toolkit, or set NVCC to its full path.
  exit /b 1
)

call "%VCVARS%" >nul
if errorlevel 1 exit /b 1

set "NVCC_PREPEND_FLAGS=--allow-unsupported-compiler"
"%NVCC%" -arch=sm_75 -std=c++17 -O2 -I "%~dp0..\..\include" -o exp_smoke.exe exp_smoke.cu
if errorlevel 1 exit /b 1

exp_smoke.exe "%REFERENCE%"
set "RC=%errorlevel%"
if "%RC%"=="77" (
  echo [ SKIPPED ] no CUDA device: compile check only
  popd
  endlocal & exit /b 0
)
popd
endlocal & exit /b %RC%
