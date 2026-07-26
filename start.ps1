Set-Location $PSScriptRoot

if (Test-Path .env) {
    Get-Content .env | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#")) { return }
        $parts = $line -split '=', 2
        if ($parts.Length -eq 2) {
            [System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim())
        }
    }
}

$proxyHost = if ($env:PROXY_HOST) { $env:PROXY_HOST } else { "127.0.0.1" }
$proxyPort = if ($env:PROXY_PORT) { $env:PROXY_PORT } else { "8082" }

uvicorn main:app --host $proxyHost --port $proxyPort
