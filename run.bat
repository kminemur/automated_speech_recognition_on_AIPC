@echo off
setlocal

if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found. Run setup.bat first.
    exit /b 1
)

call ".venv\Scripts\activate.bat"

rem Prefer local OpenVINO IR if --model-id is not provided explicitly.
set "HAS_MODELID="
set "HAS_DEVICE="

:arg_loop
if "%~1"=="" goto after_arg_scan
if /I "%~1"=="--model-id" set "HAS_MODELID=1"
if /I "%~1"=="--device" set "HAS_DEVICE=1"
shift
goto arg_loop
:after_arg_scan

rem Prefer the locally exported Whisper tiny IR directory.
set "IR_DIR=%~dp0whisper-tiny-ov"
set "MODEL_ARG="
if not defined HAS_MODELID (
    if exist "%IR_DIR%\openvino_model.xml" (
        echo Using local IR model at "%IR_DIR%".
        set "MODEL_ARG=--model-id \"%IR_DIR%\""
    ) else (
        echo No local IR found at "%IR_DIR%"; falling back to default model-id ^(openai/whisper-tiny^).
    )
)

rem Allow selecting device via DEVICE env var when --device is not passed.
set "DEVICE_ARG="
if not defined HAS_DEVICE (
    if defined DEVICE (
        echo Using device "%DEVICE%" from environment.
        set "DEVICE_ARG=--device %DEVICE%"
    )
)

call python app.py %MODEL_ARG% %DEVICE_ARG% %*

endlocal
