@echo off
setlocal enabledelayedexpansion

:: ============================================================
::  MLPVehiclePrice -- Build and Run Docker Image (Windows)
::  Auteur : Axel Bouchaud-Roche | UCO B2 S4
:: ============================================================

set IMAGE_NAME=mlp-vehicle-price-axelbcr
set VERSION=1.0.0
set PORT=8001

echo.
echo  ==================================================
echo   MLPVehiclePrice -- Docker Build Script
echo   Image : %IMAGE_NAME%:%VERSION%
echo  ==================================================
echo.

:: --- Verifier que Docker est installe ---
docker --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Docker n'est pas detecte sur ce systeme.
    echo.
    echo  Installation automatique de Docker Desktop...
    echo.

    :: Tenter avec winget (Windows Package Manager)
    winget --version >nul 2>&1
    if errorlevel 1 (
        echo [INFO] winget non disponible, telechargement direct de l'installeur...
        goto :download_docker
    )

    echo [INFO] Installation via winget...
    winget install -e --id Docker.DockerDesktop --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo [AVERTISSEMENT] winget a echoue, telechargement direct...
        goto :download_docker
    )
    goto :docker_installed

    :download_docker
    echo [INFO] Telechargement de Docker Desktop via PowerShell...
    set "DOCKER_INSTALLER=%TEMP%\DockerDesktopInstaller.exe"

    powershell -Command "Write-Host 'Telechargement en cours (environ 500 Mo)...'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://desktop.docker.com/win/main/amd64/Docker%%20Desktop%%20Installer.exe' -OutFile '%DOCKER_INSTALLER%' -UseBasicParsing"

    if not exist "%DOCKER_INSTALLER%" (
        echo.
        echo [ERREUR] Le telechargement a echoue.
        echo          Installez Docker Desktop manuellement :
        echo          https://www.docker.com/products/docker-desktop
        pause
        exit /b 1
    )

    echo [INFO] Lancement de l'installeur Docker Desktop...
    echo          Suivez les instructions de l'installeur.
    echo.
    start /wait "" "%DOCKER_INSTALLER%" install --quiet

    :: Nettoyage
    del "%DOCKER_INSTALLER%" >nul 2>&1

    :docker_installed
    echo.
    echo  ==================================================
    echo   Docker Desktop a ete installe.
    echo.
    echo   IMPORTANT : Vous devez maintenant :
    echo     1. Redemarrer votre PC (si demande)
    echo     2. Lancer Docker Desktop
    echo     3. Attendre que Docker soit pret (icone verte)
    echo     4. Relancer ce script : build_docker.bat
    echo  ==================================================
    echo.
    pause
    exit /b 0
)

:: --- Verifier que Docker daemon tourne ---
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Docker est installe mais le daemon ne repond pas.
    echo          Lancez Docker Desktop et attendez qu'il soit pret,
    echo          puis relancez ce script.
    echo.
    echo          Tentative d'ouverture de Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe" 2>nul
    pause
    exit /b 1
)

echo [OK] Docker detecte et operationnel.
echo.

:: --- Verifier les fichiers requis ---
if not exist "Dockerfile" (
    echo [ERREUR] Dockerfile introuvable dans le repertoire courant.
    echo          Lancez ce script depuis la racine du projet MLPVehiclePrice.
    pause
    exit /b 1
)

if not exist "API\vehicle_price_model.pkl" (
    echo [ERREUR] Modele introuvable : API\vehicle_price_model.pkl
    echo          Entrainez le modele d'abord via main.ipynb.
    pause
    exit /b 1
)

if not exist "frontend\index.html" (
    echo [ERREUR] Frontend introuvable : frontend\index.html
    pause
    exit /b 1
)

echo [OK] Fichiers requis presents.
echo.

:: --- Arreter et supprimer un container existant ---
for /f "tokens=*" %%i in ('docker ps -q --filter "name=%IMAGE_NAME%" 2^>nul') do (
    echo [INFO] Arret du container existant...
    docker stop %%i >nul 2>&1
)
for /f "tokens=*" %%i in ('docker ps -aq --filter "name=%IMAGE_NAME%" 2^>nul') do (
    echo [INFO] Suppression du container existant...
    docker rm %%i >nul 2>&1
)

:: --- Build de l'image ---
echo ==================================================
echo  CONSTRUCTION DE L'IMAGE DOCKER
echo  %IMAGE_NAME%:%VERSION%
echo ==================================================
echo.

docker build -t %IMAGE_NAME%:%VERSION% -t %IMAGE_NAME%:latest .

if errorlevel 1 (
    echo.
    echo [ERREUR] Le build a echoue. Verifiez les logs ci-dessus.
    pause
    exit /b 1
)

echo.
echo [OK] Image construite avec succes !
echo.

:: --- Afficher l'image ---
echo ==================================================
echo  IMAGES DOCKER
echo ==================================================
docker images %IMAGE_NAME%
echo.

:: --- Demander si on lance le container ---
set /p RUN_CHOICE="Lancer le container maintenant ? (O/N) : "
if /i "%RUN_CHOICE%"=="O" goto :run_container
if /i "%RUN_CHOICE%"=="Y" goto :run_container
goto :end

:run_container
echo.
echo [INFO] Lancement du container sur le port %PORT%...
echo.

docker run -d --name %IMAGE_NAME% -p %PORT%:%PORT% %IMAGE_NAME%:%VERSION%

if errorlevel 1 (
    echo [ERREUR] Impossible de lancer le container.
    pause
    exit /b 1
)

echo.
echo  ==================================================
echo   Container lance avec succes !
echo.
echo   Interface Web : http://localhost:%PORT%
echo   API Docs      : http://localhost:%PORT%/docs
echo   API Endpoint  : POST /predict/full
echo.
echo   Arreter : docker stop %IMAGE_NAME%
echo  ==================================================
echo.

:: --- Ouvrir le navigateur ---
start http://localhost:%PORT%

:end
echo.
echo Termine.
pause
