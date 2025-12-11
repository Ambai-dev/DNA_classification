@echo off
echo ============================================
echo Creating Gene Data Visualizations...
echo ============================================
cd /d C:\new_kaggle
python visualizations\create_visualizations.py
echo.
echo Done! Check visualizations\plots\ folder
pause

