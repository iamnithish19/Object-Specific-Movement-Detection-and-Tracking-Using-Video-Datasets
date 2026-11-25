@echo off
echo ======================================================================
echo  STARTING LIVE OBJECT DETECTION & TRACKING WEB SERVER
echo ======================================================================
echo.
echo Server URL: http://localhost:5000
echo Open this URL in your web browser (Chrome, Edge, Firefox, Brave)
echo Press Ctrl+C in this terminal to stop the server anytime.
echo.
.\.venv\Scripts\python.exe -u app.py
pause
