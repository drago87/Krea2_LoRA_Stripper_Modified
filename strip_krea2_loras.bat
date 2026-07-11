@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   Krea2 LoRA Batch Stripper
echo ============================================
echo.
echo This will scan a folder AND all its sub-folders for Krea2 LoRAs
echo and create size-reduced copies (text-conditioning layers only).
echo Originals are never overwritten -- a new *_stripped.safetensors
echo file is written alongside each original, plus (if present) a patched
echo *_stripped.metadata.json and a copied *_stripped.jpeg preview.
echo Missing .jpeg / .metadata.json sidecars are silently skipped.
echo.
echo Right after you provide the folder path, you'll be asked whether to
echo MOVE the original files out to another folder. If yes, after each
echo file's _stripped outputs are written, the original .safetensors
echo (+ optional .jpeg + .metadata.json) is moved into the destination
echo folder, preserving sub-folder structure.
echo.

set "LORA_PATH="
set /p LORA_PATH="Paste the full path to your LoRA folder: "

:: Strip any double quotes the user may have typed -- we always re-quote
:: when invoking Python, so leaving user quotes in would produce double
:: quotes like ""D:\path\Krea 2"" and break paths that contain spaces.
set "LORA_PATH=%LORA_PATH:"=%"

:: Strip a single trailing backslash, which can also cause quoting issues
:: on Windows (e.g. "D:\foo\" -> the closing quote gets escaped).
if "%LORA_PATH:~-1%"=="\" set "LORA_PATH=%LORA_PATH:~0,-1%"

if "%LORA_PATH%"=="" (
    echo.
    echo ERROR: No path entered.
    pause
    exit /b 1
)

if not exist "%LORA_PATH%" (
    echo.
    echo ERROR: That path does not exist:
    echo   "%LORA_PATH%"
    pause
    exit /b 1
)

echo.
echo Running on: "%LORA_PATH%"
echo.

python "%~dp0batch_strip_krea2.py" "%LORA_PATH%"

echo.
echo ============================================
echo Done. Press any key to close.
echo ============================================
pause >nul
