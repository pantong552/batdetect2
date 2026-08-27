@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

title BatDetect2 - Noise Annotation JSON Generator

echo ======================================================================
echo   BatDetect2 Noise / Background Annotation Generator
echo ======================================================================
echo.

set "TARGET_DIR=%~1"
if not "%TARGET_DIR%"=="" goto START_PROCESS

echo Please drag and drop your noise audio folder into this window,
echo or press Enter directly to scan current directory (%~dp0).
echo.
set "USER_IN="
set /p "USER_IN=Folder Path: "

set "TARGET_DIR=."
if defined USER_IN (
    set "TARGET_DIR=!USER_IN!"
)

:START_PROCESS
:: Remove surrounding quotes
set "TARGET_DIR=%TARGET_DIR:"=%"
if "%TARGET_DIR%"=="" set "TARGET_DIR=."

echo.
echo [*] Scanning audio folder: "%TARGET_DIR%"
echo.

set /a COUNT=0
set /a SKIPPED=0

for /r "%TARGET_DIR%" %%F in (*.wav *.WAV *.flac *.FLAC *.mp3) do (
    set "AUDIO_FILE=%%~fF"
    set "JSON_FILE=%%~dpnF%%~xF.json"

    if exist "!JSON_FILE!" (
        echo [SKIP] Already exists: %%~nxF.json
        set /a SKIPPED+=1
    ) else (
        (
            echo {
            echo   "annotated": true,
            echo   "issues": false,
            echo   "notes": "Background noise / negative sample (auto-generated)",
            echo   "annotation": []
            echo }
        ) > "!JSON_FILE!"

        echo [CREATE] %%~nxF -^> %%~nxF.json
        set /a COUNT+=1
    )
)

echo.
echo ======================================================================
echo   Processing Complete!
echo   [+] Created negative annotations: !COUNT!
echo   [!] Skipped existing annotations: !SKIPPED!
echo ======================================================================
echo.
echo Press any key to close this window.
pause >nul
