@echo off
REM Automated training script for remaining C-MAPSS datasets
REM This will train TCN and BiLSTM on FD003 and FD004

echo ============================================================
echo NASA C-MAPSS Automated Training Pipeline
echo ============================================================
echo.
echo This script will train:
echo   - FD003: TCN + BiLSTM
echo   - FD004: TCN + BiLSTM
echo.
echo Estimated time: ~90 minutes
echo ============================================================
echo.

echo [1/4] Training TCN on FD003...
python train_advanced.py --model tcn --dataset FD003
if %errorlevel% neq 0 (
    echo ERROR: TCN FD003 training failed!
    pause
    exit /b 1
)

echo.
echo [2/4] Training BiLSTM on FD003...
python train_advanced.py --model bilstm --dataset FD003
if %errorlevel% neq 0 (
    echo ERROR: BiLSTM FD003 training failed!
    pause
    exit /b 1
)

echo.
echo [3/4] Training TCN on FD004...
python train_advanced.py --model tcn --dataset FD004
if %errorlevel% neq 0 (
    echo ERROR: TCN FD004 training failed!
    pause
    exit /b 1
)

echo.
echo [4/4] Training BiLSTM on FD004...
python train_advanced.py --model bilstm --dataset FD004
if %errorlevel% neq 0 (
    echo ERROR: BiLSTM FD004 training failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo ALL TRAINING COMPLETE!
echo ============================================================
echo.
echo Results saved to: checkpoints\
dir /b checkpoints\*.json
echo.
pause
