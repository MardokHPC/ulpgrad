@echo off
rem Locate the toolchains the experiments need and export them to the caller:
rem
rem   VCVARS   full path to vcvars64.bat from a Visual Studio installation
rem   NVCC     full path to nvcc.exe from a CUDA toolkit
rem
rem Set either one before calling this file to override discovery. Nothing here is
rem tied to one machine: a clone elsewhere should be able to run the harness, or the
rem numbers are not reproducible and the harness is decoration.
rem
rem Two pairings matter and are encoded below on purpose:
rem   * MSVC 14.44 (VS 2022) with CUDA 12.6 is the pair this repository was measured
rem     with. VS 18's 14.51 crashes nvcc 12.6's cudafe++, and MSVC older than 14.44
rem     refuses the CUDA 12.6 STL headers with STL1002.
rem   * CUDA 12.4 is the floor with modern MSVC, so discovery walks versions
rem     newest-first. CUDA_PATH often pins an older toolkit and is only a last resort.
setlocal EnableExtensions

if not defined VCVARS call :find_vcvars
if not defined NVCC call :find_nvcc
goto export

:find_vcvars
rem VS 2022 first, by explicit path: vswhere's -version range syntax needs brackets,
rem and a bracket expression inside the for-in backticks closes the block early, so
rem the query dies quietly and the fallback wins. Explicit paths are boring and work.
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if defined VCVARS goto :eof
if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if defined VCVARS goto :eof
if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
if defined VCVARS goto :eof

rem Anything else vswhere knows about, newest first.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto vcvars_fallback
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
if exist "%VCVARS%" goto :eof

:vcvars_fallback
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
goto :eof

:find_nvcc
for %%v in (13.1 13.0 12.9 12.8 12.7 12.6 12.5 12.4 12.3 12.2 12.1 12.0 11.8) do if not defined NVCC if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%%v\bin\nvcc.exe" set "NVCC=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%%v\bin\nvcc.exe"
if defined NVCC goto :eof
if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
goto :eof

:export
if not defined VCVARS (
    echo ERROR: vcvars64.bat not found. Install the MSVC C++ toolset, or set VCVARS
    echo        to the full path of vcvars64.bat before calling this file.
    exit /b 1
)
endlocal & set "VCVARS=%VCVARS%" & set "NVCC=%NVCC%"
exit /b 0
