param(
    [string]$HexagonSdkRoot = $env:HEXAGON_SDK_ROOT,
    [string]$HexagonToolsRoot = $env:HEXAGON_TOOLS_ROOT,
    [string]$WindowsSdkBin = $env:WINDOWS_SDK_BIN,
    [string]$Inf2Cat = $env:INF2CAT_EXE,
    [string]$SignTool = $env:SIGNTOOL_EXE,
    [string]$Certificate = $env:HEXAGON_HTP_CERT,
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Crate = Split-Path -Parent (Split-Path -Parent $Here)
if (-not $HexagonSdkRoot) { throw "HEXAGON_SDK_ROOT is required" }
if (-not $HexagonToolsRoot) {
    $HexagonToolsRoot = Join-Path $HexagonSdkRoot "tools\HEXAGON_Tools\19.0.07"
}
if (-not $Certificate) { throw "HEXAGON_HTP_CERT is required" }
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $Crate "target\direct-htp-package" }

if (-not $Inf2Cat -and $WindowsSdkBin) {
    $Inf2Cat = Join-Path $WindowsSdkBin "x86\Inf2Cat.exe"
}
if (-not $SignTool -and $WindowsSdkBin) {
    $SignTool = Join-Path $WindowsSdkBin "arm64\signtool.exe"
}
$KitsBin = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"
if (-not $Inf2Cat -or -not (Test-Path -LiteralPath $Inf2Cat)) {
    $Inf2Cat = Get-ChildItem -LiteralPath $KitsBin -Filter Inf2Cat.exe -File -Recurse |
        Where-Object { (Split-Path -Leaf $_.DirectoryName) -eq "x86" } |
        Sort-Object FullName -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $SignTool -or -not (Test-Path -LiteralPath $SignTool)) {
    $SignTool = Get-ChildItem -LiteralPath $KitsBin -Filter signtool.exe -File -Recurse |
        Where-Object { (Split-Path -Leaf $_.DirectoryName) -eq "arm64" } |
        Sort-Object FullName -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $Inf2Cat -or -not (Test-Path -LiteralPath $Inf2Cat)) {
    throw "Inf2Cat.exe was not found; set INF2CAT_EXE or WINDOWS_SDK_BIN"
}
if (-not $SignTool -or -not (Test-Path -LiteralPath $SignTool)) {
    throw "arm64 signtool.exe was not found; set SIGNTOOL_EXE or WINDOWS_SDK_BIN"
}

$BuildDirectory = Join-Path $Crate "target\direct-htp-v73"
$Toolchain = Join-Path $Here "htp\cmake-toolchain.cmake"
cmake --fresh -S (Join-Path $Here "htp") -B $BuildDirectory -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain" `
    "-DHEXAGON_SDK_ROOT=$HexagonSdkRoot" `
    "-DHEXAGON_TOOLS_ROOT=$HexagonToolsRoot" `
    -DDSP_VERSION=v73 -DPREBUILT_LIB_DIR=toolv19_v73
if ($LASTEXITCODE -ne 0) { throw "HTP CMake configuration failed with exit code $LASTEXITCODE" }
cmake --build $BuildDirectory --config Release
if ($LASTEXITCODE -ne 0) { throw "HTP build failed with exit code $LASTEXITCODE" }

New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
Copy-Item -LiteralPath (Join-Path $BuildDirectory "libvirtio-accel-htp-v73.so") -Destination $OutputDirectory -Force
Copy-Item -LiteralPath (Join-Path $Here "libvirtio-accel-htp.inf") -Destination $OutputDirectory -Force
& $Inf2Cat "/driver:$OutputDirectory" /os:10_25H2_ARM64
& $SignTool sign /fd sha256 /f $Certificate `
    (Join-Path $OutputDirectory "libvirtio-accel-htp.cat")
& $SignTool verify /v /pa `
    (Join-Path $OutputDirectory "libvirtio-accel-htp.cat")
