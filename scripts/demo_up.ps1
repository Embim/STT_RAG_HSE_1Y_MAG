<#
.SYNOPSIS
  Поднимает демо DS Navigator на этой машине: docker-сервисы (Weaviate + FRIDA),
  затем FastAPI с кастомным фронтом на http://localhost:8001.

.DESCRIPTION
  Порядок важен: VectorStoreManager в src/system/llm/llm_services.py подключается
  к Weaviate НА ИМПОРТЕ, поэтому Weaviate должен быть готов ДО старта API.
  Скрипт ждёт готовности Weaviate, потом запускает uvicorn (foreground).

  Гейт доступа: если задан -Token (или $env:DEMO_ACCESS_TOKEN), ручка /forward и
  ingest требуют этот токен. Без токена API открыт (для локальной отладки).
  Для публичной выдачи через туннель токен ОБЯЗАТЕЛЕН.

.PARAMETER Token
  Ключ доступа. Если не задан — сгенерируется случайный и будет показан.

.PARAMETER WithAsr
  Дополнительно поднять Whisper (asr-whisper) — нужно только если хочешь
  демонстрировать живой ингест аудио/видео. Для поиска не требуется.

.PARAMETER NoDocker
  Не трогать docker (сервисы уже подняты) — только запустить API.

.EXAMPLE
  .\scripts\demo_up.ps1
  .\scripts\demo_up.ps1 -Token "hse2026" -WithAsr
#>
param(
  [string]$Token = $env:DEMO_ACCESS_TOKEN,
  [switch]$WithAsr,
  [switch]$NoDocker
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# ── .env check ────────────────────────────────────────────────────────
if (-not (Test-Path (Join-Path $root ".env"))) {
  Write-Host "[!] Нет файла .env в корне проекта." -ForegroundColor Red
  Write-Host "    Создай его (см. .env_example): нужны LLM_MODEL, LLM_API_KEY_1/2/3 (OpenRouter)." -ForegroundColor Red
  exit 1
}

# ── access token ──────────────────────────────────────────────────────
if (-not $Token) {
  $bytes = New-Object 'System.Byte[]' 10
  (New-Object System.Random).NextBytes($bytes)
  $Token = -join ($bytes | ForEach-Object { '{0:x2}' -f $_ })
  Write-Host "[i] Сгенерирован токен доступа (никому, кроме демо, не давай):" -ForegroundColor Yellow
}
$env:DEMO_ACCESS_TOKEN = $Token

# ── docker services ───────────────────────────────────────────────────
if (-not $NoDocker) {
  $profiles = @("--profile","vdb","--profile","emb")
  if ($WithAsr) { $profiles += @("--profile","asr-whisper") }
  Write-Host "[*] Поднимаю docker: $($profiles -join ' ')" -ForegroundColor Cyan
  Push-Location $root
  docker compose @profiles up -d
  Pop-Location

  # Weaviate должен быть готов до импорта API
  Write-Host "[*] Жду готовности Weaviate (http://localhost:8080) ..." -ForegroundColor Cyan
  $ready = $false
  for ($i = 0; $i -lt 60; $i++) {
    try {
      $r = Invoke-WebRequest -Uri "http://localhost:8080/v1/.well-known/ready" -UseBasicParsing -TimeoutSec 3
      if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { Start-Sleep -Seconds 2 }
  }
  if (-not $ready) {
    Write-Host "[!] Weaviate не ответил за ~2 мин. Проверь 'docker compose logs weaviate'." -ForegroundColor Red
    exit 1
  }
  Write-Host "[+] Weaviate готов." -ForegroundColor Green
  Write-Host "[i] FRIDA (Infinity) грузится в фоне — первый запрос может подождать модель." -ForegroundColor DarkGray
}

# ── API ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  Токен доступа : $Token" -ForegroundColor Green
Write-Host "  Локально      : http://localhost:8001/?key=$Token" -ForegroundColor Green
Write-Host "  Туннель наружу: новый терминал -> .\scripts\demo_tunnel.ps1" -ForegroundColor Green
Write-Host ""
Write-Host "[*] Запускаю FastAPI (Ctrl+C для остановки) ..." -ForegroundColor Cyan

Set-Location (Join-Path $root "src")
uv run uvicorn api.main:app --host 0.0.0.0 --port 8001
