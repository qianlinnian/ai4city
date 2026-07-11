param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

$root = Split-Path -Parent $PSScriptRoot
$envInfo = conda env list --json | ConvertFrom-Json
$envPath = $envInfo.envs | Where-Object { (Split-Path -Leaf $_) -eq 'ai4city-mas' } | Select-Object -First 1

if (-not $envPath) {
    throw "Conda environment 'ai4city-mas' was not found."
}

$python = Join-Path $envPath 'python.exe'
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $root 'src'

try {
    & $python -m ai4city_mas @CliArgs
    exit $LASTEXITCODE
}
finally {
    $env:PYTHONPATH = $previousPythonPath
}
