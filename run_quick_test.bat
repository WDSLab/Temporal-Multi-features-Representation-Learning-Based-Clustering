@echo off
REM Quick test script - Windows version
REM Runs single experiment per dataset for fast testing

setlocal enabledelayedexpansion

REM Set parameters for quick testing
set DEVICE=cuda
set OUTPUT_DIR=results_quick_test
set NUM_EPOCHS=101
set LR=0.0002

REM Create output directory
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

REM Dataset list (36 datasets)
set datasets=ArrowHead Beef BeetleFly BirdChicken Car ChlorineConcentration Coffee DiatomsizeReduction Dist.phal.outl.agegroup Dist.phal.outl.correct ECG200 ECGFiveDays GunPoint Ham Herring Lighting2 Meat Mid.phal.outl.agegroup Mid.phal.outl.correct Mid.phal.TW MoteStrain OSULeaf Plane Prox.phal.outl.ageGroup Prox.phal.TW SonyAIBORobotSurface SonyAIBORobotSurfaceII SwedishLeaf Symbols ToeSegmentation1 ToeSegmentation2 TwoLeadECG TwoPatterns Wafer Wine WordsSynonyms

echo 🔬 Quick Test Mode - Single run per dataset (Windows)
echo Epochs per experiment: %NUM_EPOCHS%
echo Output directory: %OUTPUT_DIR%
echo.

set /a COMPLETED=0
set /a FAILED=0
set /a CURRENT=0
set /a TOTAL=0

REM Count total datasets
for %%d in (%datasets%) do (
    set /a TOTAL+=1
)

echo Total datasets: !TOTAL!
echo Start time: %date% %time%
echo.

for %%d in (%datasets%) do (
    set /a CURRENT+=1
    echo [!CURRENT!/!TOTAL!] Testing: %%d
    
    python main.py --dataset "%%d" --single_run --num_epochs %NUM_EPOCHS% --lr %LR% --device %DEVICE% --output_dir %OUTPUT_DIR% --verbose
    
    if !errorlevel! equ 0 (
        echo ✅ %%d OK
        set /a COMPLETED+=1
    ) else (
        echo ❌ %%d FAILED
        echo %%d >> %OUTPUT_DIR%\failed_datasets.txt
        set /a FAILED+=1
    )
    echo.
)

echo 🎉 Quick test completed!
echo End time: %date% %time%
echo Completed: !COMPLETED!, Failed: !FAILED!

if !FAILED! gtr 0 (
    echo.
    echo Failed datasets:
    if exist %OUTPUT_DIR%\failed_datasets.txt (
        type %OUTPUT_DIR%\failed_datasets.txt
    )
)

pause