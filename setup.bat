@echo off
REM Run this script to download and build the model.

for /F "tokens=1,2 delims== " %%a in (.\.conf) do (
  set %%a=%%b
)

set arg=%1

IF "%arg%"=="" (
  set arg=normal
) ELSE (
  set arg=%arg%
)


IF %arg%==--params (
  echo Params parsed.
  goto parsparam
)


IF %arg%==--help (
  echo setup.sh [OPTION]
  echo Download weights and setup Yolo-V3 model.
  echo Options:
  echo -f or --force   : Remove existing model and re-run setup.
  echo -h or --help    : Display help message.
  exit
)

IF %arg%==-h (
  echo setup.sh [OPTION]
  echo Download weights and setup Yolo-V3 model.
  echo Options:
  echo -f or --force   : Remove existing model and re-run setup.
  echo -h or --help    : Display help message.
  pause
  exit
)

IF %arg%==-f (
  echo Removing saved weights...
  rm %modelweights%
  rm %rawweights%
)

IF exist %modelweights% (
  echo Weights are found.
  echo Testing setup.
  .\test.bat
  goto success
)

IF not exist %rawweights% (
  echo Downloading model weights...
  curl %weightsurl% -o %raww%
  move %raww% %weightsdir%\.
  IF exist %rawweights% (
    echo Model weights successfully downloaded.
  ) ELSE (
    echo Failed to fetch model weights.
    goto end
  )
) ELSE (
  echo Model weights already downloaded.
)

IF not exist %modelweights% (
  echo Setting up model...
  python .\%buildscript%
  IF not exist %modelweights% (
    echo Failed to setup model.
    goto end
  ) ELSE (
    echo Done.
  )

)

echo Testing setup...
.\test.bat
goto success


:end
echo Model setup failed.
exit 1

:success
echo Model build successful.
exit 0

:parsparam
