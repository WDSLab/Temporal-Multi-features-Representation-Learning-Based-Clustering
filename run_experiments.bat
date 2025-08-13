@echo off
REM VMD-LSTM Time Series Clustering Experiments - Windows Batch Script
REM Run experiments on multiple UCR datasets

setlocal enabledelayedexpansion

REM Set default parameters
set DEVICE=cuda
set OUTPUT_DIR=results
set NUM_EPOCHS=1001
set LR=0.0002

REM VMD decomposition modes to test
set K_LIST=2 3 4 5 6

REM LSTM hidden sizes to test  
set HIDDEN_SIZE_LIST=10 50 100 400 800 1200 1600 2000

REM Create output directory
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

REM Dataset list (36 datasets)
set datasets=ArrowHead Beef BeetleFly BirdChicken Car ChlorineConcentration Coffee DiatomsizeReduction Dist.phal.outl.agegroup Dist.phal.outl.correct ECG200 ECGFiveDays GunPoint Ham Herring Lighting2 Meat Mid.phal.outl.agegroup Mid.phal.outl.correct Mid.phal.TW MoteStrain OSULeaf Plane Prox.phal.outl.ageGroup Prox.phal.TW SonyAIBORobotSurface SonyAIBORobotSurfaceII SwedishLeaf Symbols ToeSegmentation1 ToeSegmentation2 TwoLeadECG TwoPatterns Wafer Wine WordsSynonyms

echo ================================================================
echo 🚀 VMD-LSTM Time Series Clustering Experiments (Windows)
echo ================================================================
echo Output directory: %OUTPUT_DIR%
echo Device: %DEVICE%
echo Epochs per experiment: %NUM_EPOCHS%
echo K values: %K_LIST%
echo Hidden sizes: %HIDDEN_SIZE_LIST%
echo.

REM Record start time
echo Experiment started at: %date% %time%
echo %date% %time% > %OUTPUT_DIR%\experiment_start.txt

REM Initialize counters
set /a COMPLETED=0
set /a FAILED=0
set /a TOTAL=0

REM Count total datasets
for %%d in (%datasets%) do (
    set /a TOTAL+=1
)

echo Total datasets: !TOTAL!
echo.

REM Delete previous failed datasets log
if exist %OUTPUT_DIR%\failed_datasets.txt del %OUTPUT_DIR%\failed_datasets.txt

REM Run experiments for each dataset
set /a CURRENT=0
for %%d in (%datasets%) do (
    set /a CURRENT+=1
    echo ==================================================
    echo [!CURRENT!/!TOTAL!] Processing dataset: %%d
    echo Start time: %time%
    echo ==================================================
    
    python main.py --dataset "%%d" --K_list %K_LIST% --hidden_size_list %HIDDEN_SIZE_LIST% --num_epochs %NUM_EPOCHS% --lr %LR% --device %DEVICE% --output_dir %OUTPUT_DIR% --verbose
    
    if !errorlevel! equ 0 (
        echo ✅ SUCCESS: %%d completed
        set /a COMPLETED+=1
    ) else (
        echo ❌ FAILED: %%d failed with exit code !errorlevel!
        echo %%d >> %OUTPUT_DIR%\failed_datasets.txt
        set /a FAILED+=1
    )
    
    echo End time: %time%
    echo Completed: !COMPLETED!, Failed: !FAILED!
    echo Completed: !COMPLETED!, Failed: !FAILED! > %OUTPUT_DIR%\progress.txt
    echo.
)

REM Generate final summary
echo ==================================================
echo 🎉 All experiments completed!
echo ==================================================
echo Start time: (see %OUTPUT_DIR%\experiment_start.txt)
echo End time: %date% %time%
echo Total datasets: !TOTAL!
echo Completed successfully: !COMPLETED!
echo Failed: !FAILED!

if !FAILED! gtr 0 (
    echo.
    echo Failed datasets:
    if exist %OUTPUT_DIR%\failed_datasets.txt (
        type %OUTPUT_DIR%\failed_datasets.txt
    ) else (
        echo None
    )
)

echo.
echo Generating summary report...
python -c "import sys; sys.path.append('src'); from utils import create_summary_report; create_summary_report('%OUTPUT_DIR%')" 2>nul || echo Summary report generation skipped

echo.
echo Results saved in: %OUTPUT_DIR%
echo Check individual dataset results: %OUTPUT_DIR%\*_{dataset_name}.csv
echo Overall summary: %OUTPUT_DIR%\all_results.csv
echo Best results: %OUTPUT_DIR%\best_results.csv

pause