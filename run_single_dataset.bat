@echo off
REM Script to run experiment on a single dataset - Windows version
REM Usage: run_single_dataset.bat [DATASET_NAME]

setlocal enabledelayedexpansion

REM Default parameters
if "%1"=="" (
    set DATASET=ECG200
) else (
    set DATASET=%1
)

set DEVICE=cuda
set OUTPUT_DIR=results
set NUM_EPOCHS=1001
set LR=0.0002
set K_LIST=2 3 4 5 6
set HIDDEN_SIZE_LIST=10 50 100 400 800 1200 1600 2000

echo 🔍 Single Dataset Experiment (Windows)
echo Dataset: %DATASET%
echo Device: %DEVICE%
echo Epochs: %NUM_EPOCHS%
echo Output: %OUTPUT_DIR%
echo.

REM Create output directory
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

REM Run experiment
echo Starting experiment at: %date% %time%
echo.

python main.py --dataset "%DATASET%" --K_list %K_LIST% --hidden_size_list %HIDDEN_SIZE_LIST% --num_epochs %NUM_EPOCHS% --lr %LR% --device %DEVICE% --output_dir %OUTPUT_DIR% --verbose

set EXIT_CODE=%errorlevel%

echo.
echo Finished at: %date% %time%

if %EXIT_CODE% equ 0 (
    echo ✅ SUCCESS: %DATASET% completed successfully
    echo.
    echo Results saved in:
    echo   - RI scores: %OUTPUT_DIR%\ri_%DATASET%.csv
    echo   - NMI scores: %OUTPUT_DIR%\nmi_%DATASET%.csv
    echo   - Summary: %OUTPUT_DIR%\summary_%DATASET%.csv
) else (
    echo ❌ FAILED: %DATASET% failed with exit code %EXIT_CODE%
)

pause