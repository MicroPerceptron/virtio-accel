#requires -Version 5.1
<#
.SYNOPSIS
Clone PR 208, build an ARM64 Python wheel, and require real QNN HTP execution.
.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File .\test-hexagon.ps1 -QnnSdkRoot 'C:\Qualcomm\AIStack\QAIRT\2.49.0.260730'
.NOTES
Requires Git, Rust (1.85+ with the aarch64-pc-windows-msvc target), native ARM64
Python 3.10+, Visual Studio ARM64 C++ build tools, and the complete QAIRT SDK.
Uses a new directory and virtual environment. Does not install drivers or SDKs.
Environment changes are process-local and restored. Logs contain no SDK binaries.
The default revision follows PR 208; report.json records the exact commit tested.
#>
[CmdletBinding()]
param(
    [string]$QnnSdkRoot = '',
    [string]$AdspLibraryPath = '',
    [string]$Python = 'python',
    [string]$Revision = 'refs/pull/208/head',
    [string]$WorkDirectory = (Join-Path ([IO.Path]::GetTempPath()) ('virtio-accel-pr208-' + [guid]::NewGuid().ToString('N')))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw 'Run this script on the Windows Snapdragon machine.'
}
if (Test-Path -LiteralPath $WorkDirectory) {
    throw "WorkDirectory already exists; choose a new directory: $WorkDirectory"
}
$WorkDirectory = [IO.Path]::GetFullPath($WorkDirectory)
$logs = Join-Path $WorkDirectory 'logs'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
$report = [ordered]@{
    status = 'running'; revision = $Revision; commit = $null
    started_utc = [DateTime]::UtcNow.ToString('o'); os = [Environment]::OSVersion.ToString()
    python = $null; sdk = $null; dsp_path = $null; error = $null; steps = [ordered]@{}
}
$environmentNames = @('PATH', 'PYTHONPATH', 'CARGO_TARGET_DIR', 'CARGO_BUILD_TARGET',
    'CARGO_INCREMENTAL', 'VIRTIO_ACCEL_HEXAGON', 'VIRTIO_ACCEL_QNN_SDK_ROOT',
    'VIRTIO_ACCEL_QNN_LIB_DIR', 'ADSP_LIBRARY_PATH', 'VIRTIO_ACCEL_PYTHON_TEST_DEVICE',
    'VIRTIO_ACCEL_PYTHON_TEST_PROVIDER')
$savedEnvironment = @{}
foreach ($name in $environmentNames) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$pushed = $false
$exitCode = 1

function Invoke-Logged {
    param([string]$Step, [string]$Executable, [string[]]$Arguments, [switch]$AllowFailure)
    Write-Host "[$Step] $Executable $($Arguments -join ' ')"
    $log = Join-Path $logs ($Step + '.log')
    $timer = [Diagnostics.Stopwatch]::StartNew()
    # Windows PowerShell 5.1 wraps native stderr as ErrorRecord. Capture it as
    # text, then check the actual process exit code (not stderr's presence).
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Executable @Arguments 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log | Out-Host
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
        $timer.Stop()
    }
    $report.steps[$Step] = [ordered]@{ exit_code = $code; seconds = $timer.Elapsed.TotalSeconds }
    if ($code -ne 0 -and -not $AllowFailure) {
        throw "$Step failed (exit $code). See $log"
    }
}

try {
    $pythonExe = (Get-Command $Python -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
    Invoke-Logged '01-python' $pythonExe @('-c', "import json,sys,sysconfig; print(json.dumps({'executable':sys.executable,'version':list(sys.version_info[:3]),'platform':sysconfig.get_platform()}))")
    $report.python = Get-Content (Join-Path $logs '01-python.log') -Raw | ConvertFrom-Json
    if ($report.python.platform -ne 'win-arm64') {
        throw "Native ARM64 Python is required; this interpreter reports $($report.python.platform). Pass -Python C:\path\to\ARM64\python.exe (x64 emulation cannot run the HTP wheel)."
    }
    if ($report.python.version[0] -ne 3 -or $report.python.version[1] -lt 10) {
        throw 'Python 3.10 or newer is required.'
    }
    foreach ($tool in @('git', 'cargo', 'rustc')) {
        Get-Command $tool -CommandType Application -ErrorAction Stop | Out-Null
        Invoke-Logged ("02-$tool") $tool @('--version')
    }
    if (-not $QnnSdkRoot) { $QnnSdkRoot = $env:VIRTIO_ACCEL_QNN_SDK_ROOT }
    if (-not $QnnSdkRoot) { $QnnSdkRoot = $env:QNN_SDK_ROOT }
    if (-not $QnnSdkRoot) { throw 'Pass -QnnSdkRoot or set VIRTIO_ACCEL_QNN_SDK_ROOT / QNN_SDK_ROOT to the complete QAIRT SDK.' }
    $QnnSdkRoot = (Resolve-Path -LiteralPath $QnnSdkRoot).Path
    $report.sdk = $QnnSdkRoot
    $lib = Join-Path $QnnSdkRoot 'lib\aarch64-windows-msvc'
    foreach ($file in @('QnnHtp.lib', 'QnnHtp.dll')) {
        if (-not (Test-Path -LiteralPath (Join-Path $lib $file))) { throw "SDK is missing $lib\$file" }
    }
    # The build script validates the complete public header set as well.
    if (-not $AdspLibraryPath) { $AdspLibraryPath = $env:ADSP_LIBRARY_PATH }
    if (-not $AdspLibraryPath) {
        $AdspLibraryPath = Join-Path $QnnSdkRoot 'lib\hexagon-v73\unsigned'
        if (-not (Test-Path -LiteralPath $AdspLibraryPath)) { throw 'Pass -AdspLibraryPath for the DSP libraries matching your device, or set ADSP_LIBRARY_PATH.' }
    }
    $report.dsp_path = $AdspLibraryPath
    $env:VIRTIO_ACCEL_HEXAGON = '1'
    $env:VIRTIO_ACCEL_QNN_SDK_ROOT = $QnnSdkRoot
    $env:VIRTIO_ACCEL_QNN_LIB_DIR = $lib
    $env:ADSP_LIBRARY_PATH = $AdspLibraryPath
    $env:PATH = "$lib;$env:PATH"
    $env:PYTHONPATH = ''
    $env:CARGO_INCREMENTAL = '0'
    $env:CARGO_BUILD_TARGET = 'aarch64-pc-windows-msvc'
    $env:CARGO_TARGET_DIR = Join-Path $WorkDirectory 'target'
    $env:VIRTIO_ACCEL_PYTHON_TEST_DEVICE = 'npu'
    $env:VIRTIO_ACCEL_PYTHON_TEST_PROVIDER = 'hexagon'
    $repo = Join-Path $WorkDirectory 'repo'
    Invoke-Logged '03-clone' 'git' @('clone', '--filter=blob:none', '--no-checkout', 'https://github.com/MicroPerceptron/virtio-accel.git', $repo)
    Push-Location $repo
    $pushed = $true
    Invoke-Logged '04-fetch' 'git' @('fetch', '--depth=1', 'origin', $Revision)
    Invoke-Logged '05-checkout' 'git' @('checkout', '--detach', 'FETCH_HEAD')
    Invoke-Logged '06-revision' 'git' @('rev-parse', 'HEAD')
    $report.commit = (Get-Content (Join-Path $logs '06-revision.log') -Raw).Trim()
    Invoke-Logged '07-venv' $pythonExe @('-m', 'venv', (Join-Path $WorkDirectory 'venv'))
    $venvPython = Join-Path $WorkDirectory 'venv\Scripts\python.exe'
    Invoke-Logged '08-build-tools' $venvPython @('-m', 'pip', 'install', 'maturin>=1.8,<2')
    $dist = Join-Path $WorkDirectory 'dist'
    Invoke-Logged '09-wheel' $venvPython @('-m', 'maturin', 'build', '--release', '--manifest-path', 'python/Cargo.toml', '--interpreter', $venvPython, '--target', 'aarch64-pc-windows-msvc', '--no-default-features', '--features', 'python,hexagon', '--out', $dist)
    $wheels = @(Get-ChildItem -LiteralPath $dist -Filter '*.whl')
    if ($wheels.Count -ne 1) { throw "Expected one built wheel, found $($wheels.Count)." }
    Invoke-Logged '10-install' $venvPython @('-m', 'pip', 'install', $wheels[0].FullName)
    Invoke-Logged '11-dependencies' $venvPython @('-m', 'pip', 'freeze')
    Invoke-Logged '12-discovery' $venvPython @('-c', "import virtio_accel as va; print(va.devices()); print(va.device_diagnostics()); assert va.devices('npu', provider='hexagon'), 'Required Hexagon NPU unavailable'")
    Invoke-Logged '13-rust-tests' 'cargo' @('test', '-p', 'virtio-accel-hexagon', '--all-targets', '--', '--test-threads=1') -AllowFailure
    Invoke-Logged '14-python-tests' $venvPython @('-m', 'unittest', 'discover', '-s', 'python/tests', '-v') -AllowFailure
    if (@($report.steps.Values | Where-Object { $_.exit_code -ne 0 }).Count -ne 0) {
        throw 'One or more test suites failed. Both Rust and Python logs were retained.'
    }
    $report.status = 'passed'
    $exitCode = 0
} catch {
    $report.status = 'failed'
    $report.error = $_.Exception.Message
    Write-Host $report.error -ForegroundColor Red
} finally {
    if ($pushed) { Pop-Location }
    foreach ($name in $environmentNames) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
    }
    $report['finished_utc'] = [DateTime]::UtcNow.ToString('o')
    $report | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $logs 'report.json')
    $zip = Join-Path $WorkDirectory 'hexagon-results.zip'
    try { Compress-Archive -Path (Join-Path $logs '*') -DestinationPath $zip -ErrorAction Stop }
    catch { Write-Warning "Could not create ZIP; the logs are still in $logs" }
    Write-Host "Result: $($report.status). Report: $logs\report.json"
    Write-Host "Relay the report and logs (or $zip). The checkout and build are retained at $WorkDirectory"
}
exit $exitCode
