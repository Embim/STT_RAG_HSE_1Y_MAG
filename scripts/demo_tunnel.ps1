<#
.SYNOPSIS
  Открывает публичный HTTPS-туннель на локальный API (по умолчанию :8001)
  через cloudflared. Отдаёт случайный URL вида https://<...>.trycloudflare.com.

.DESCRIPTION
  Запускать в ОТДЕЛЬНОМ терминале, после того как demo_up.ps1 поднял API.
  Один туннель отдаёт и сайт, и API (один origin) — CORS не нужен.

  Требуется cloudflared:
    winget install --id Cloudflare.cloudflared
  (аккаунт Cloudflare для quick-туннеля НЕ нужен)

  URL меняется при каждом запуске. Для стабильного адреса см. docs/hosting_demo.md
  (именованный туннель Cloudflare).

.PARAMETER Port
  Локальный порт API. По умолчанию 8001.
#>
param([int]$Port = 8001)

$ErrorActionPreference = "Stop"

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
  Write-Host "[!] cloudflared не найден." -ForegroundColor Red
  Write-Host "    Установка: winget install --id Cloudflare.cloudflared" -ForegroundColor Yellow
  Write-Host "    (или скачай .exe: https://github.com/cloudflare/cloudflared/releases)" -ForegroundColor Yellow
  exit 1
}

Write-Host "[*] Открываю публичный HTTPS-туннель на http://localhost:$Port ..." -ForegroundColor Cyan
Write-Host "[i] Дай комиссии ссылку вида https://<...>.trycloudflare.com" -ForegroundColor DarkGray
Write-Host "[i] Не забудь добавить ?key=<твой токен> к ссылке." -ForegroundColor DarkGray
Write-Host ""

cloudflared tunnel --url "http://localhost:$Port"
