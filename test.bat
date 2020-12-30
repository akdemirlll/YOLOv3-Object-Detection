@echo off
REM Test model build.
for /F "tokens=1,2 delims== " %%a in (.\.conf) do (
  set %%a=%%b
)
echo %modelweights%
python .\test\test_yolov3.py
