param(
    [string]$Compiler = "g++",
    [string]$OutputDirectory = "build/gcc-plugin"
)

$ErrorActionPreference = "Stop"
$pluginDirectory = (& gcc -print-file-name=plugin).Trim()
$includeDirectory = Join-Path $pluginDirectory "include"
$header = Join-Path $includeDirectory "gcc-plugin.h"
if (-not (Test-Path -LiteralPath $header)) {
    throw "GCC plugin development headers were not found at $includeDirectory"
}

$resolvedOutputDirectory = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "..\$OutputDirectory")
)
New-Item -ItemType Directory -Force -Path $resolvedOutputDirectory | Out-Null
$sourcePath = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "..\self_compiler\gcc_plugin\ai_native_plugin.cc")
)

$targets = @(
    @{ Name = "ai_native_c.dll"; ImportLibrary = "cc1.exe.a" },
    @{ Name = "ai_native_cpp.dll"; ImportLibrary = "cc1plus.exe.a" }
)

foreach ($target in $targets) {
    $outputPath = Join-Path $resolvedOutputDirectory $target.Name
    $importLibrary = Join-Path $pluginDirectory $target.ImportLibrary
    if (-not (Test-Path -LiteralPath $importLibrary)) {
        throw "GCC frontend import library was not found: $importLibrary"
    }

    & $Compiler `
        -std=c++17 `
        -shared `
        -fno-rtti `
        -fno-exceptions `
        "-I$includeDirectory" `
        -o $outputPath `
        $sourcePath `
        $importLibrary
    if ($LASTEXITCODE -ne 0) {
        throw "GCC plugin build failed for $($target.Name) with exit code $LASTEXITCODE"
    }
    Write-Output $outputPath
}
