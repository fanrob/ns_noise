# 1. Скачать и установить Python 3.9 (если не установлен)
$pythonInstaller = "python-3.9.13-amd64.exe"
$pythonUrl = "https://www.python.org/ftp/python/3.9.13/$pythonInstaller"

if (-not (Get-Command python3.9 -ErrorAction SilentlyContinue)) {
    Invoke-WebRequest -Uri $pythonUrl -OutFile $pythonInstaller
    Start-Process -Wait -FilePath .\$pythonInstaller -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1"
    Remove-Item $pythonInstaller
}

# 2. Создать виртуальное окружение
python3.9 -m venv venv

# 3. Активировать виртуальное окружение
. .\venv\Scripts\Activate.ps1

# 4. Установить пакеты из requirements.txt
pip install --upgrade pip
pip install -r requirements.txt