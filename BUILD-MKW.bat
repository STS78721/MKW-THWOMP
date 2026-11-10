@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not defined MKW_GAME (
  echo [ERROR] Set MKW_GAME to your legally dumped 1.8.0 folder containing exefs\ and romfs\.
  exit /b 1
)
if not exist "%MKW_GAME%\exefs\main" (
  echo [ERROR] Missing %MKW_GAME%\exefs\main
  exit /b 1
)
if not defined NUMBER_OF_PROCESSORS set "NUMBER_OF_PROCESSORS=4"

if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv || exit /b 1
.venv\Scripts\python.exe -m pip install -r tools\requirements.txt zstandard || exit /b 1

.venv\Scripts\python.exe tools\mkw_cli.py prepare --repo . --dump "%MKW_GAME%" --i-confirm-update-180 || exit /b 1
if not exist "build\xlate\totk_xlate.exe" .venv\Scripts\python.exe tools\mkw_cli.py xlate --repo . || exit /b 1
if not exist "generated\mkw180\layout.json" .venv\Scripts\python.exe tools\mkw_cli.py generate --repo . --dump "%MKW_GAME%" --jobs %NUMBER_OF_PROCESSORS% || exit /b 1

if defined MKW_PACKED_ROMFS (
  .venv\Scripts\python.exe tools\mkw_select_romfs.py --repo . --packed "%MKW_PACKED_ROMFS%" || exit /b 1
) else (
  .venv\Scripts\python.exe tools\mkw_cli.py pack --repo . --dump "%MKW_GAME%" || exit /b 1
)

.venv\Scripts\python.exe tools\mkw_cli.py build --repo . --dump "%MKW_GAME%" --jobs %NUMBER_OF_PROCESSORS% || exit /b 1
echo.
echo Built build\mkw180\mkw.exe from your local dump.
echo Run RUN-MKW.bat to start it.
exit /b 0
