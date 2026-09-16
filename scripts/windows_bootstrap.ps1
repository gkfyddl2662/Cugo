[CmdletBinding()]
param(
    [switch]$SkipInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "`n========== $Message ==========" -ForegroundColor Cyan
}

function Invoke-Checked {
    param(
        [string]$File,
        [string[]]$Arguments = @()
    )

    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $File $($Arguments -join ' ')"
    }
}

function Refresh-ProcessPath {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machinePath;$userPath"
}

function Get-VsWherePath {
    $candidate = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $candidate) {
        return $candidate
    }
    return $null
}

function Get-VsInstallationPath {
    $vswhere = Get-VsWherePath
    if (-not $vswhere) {
        return $null
    }

    $path = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or -not $path) {
        return $null
    }
    return ($path | Select-Object -First 1)
}

function Ensure-BuildTools {
    $vsPath = Get-VsInstallationPath
    if ($vsPath) {
        return $vsPath
    }

    if ($SkipInstall) {
        throw "Visual Studio C++ Build Tools were not found and -SkipInstall was requested."
    }

    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "winget.exe was not found. Install Visual Studio 2022 Build Tools with the C++ workload, then rerun this script."
    }

    Write-Step "Installing Visual Studio 2022 C++ Build Tools"
    Invoke-Checked $winget.Source @(
        "install",
        "--id", "Microsoft.VisualStudio.2022.BuildTools",
        "-e",
        "--source", "winget",
        "--accept-source-agreements",
        "--accept-package-agreements",
        "--override", "--wait --passive --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
    )

    $vsPath = Get-VsInstallationPath
    if (-not $vsPath) {
        throw "Visual Studio Build Tools installation finished, but a C++ toolchain could not be located."
    }
    return $vsPath
}

function Import-VsEnvironment {
    param([string]$VsPath)

    $devCmd = Join-Path $VsPath "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path $devCmd)) {
        throw "VsDevCmd.bat was not found at $devCmd"
    }

    $command = "call `"$devCmd`" -no_logo -arch=x64 -host_arch=x64 >nul && set"
    $lines = & $env:ComSpec /d /s /c $command
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to import the Visual Studio x64 developer environment."
    }

    foreach ($line in $lines) {
        if ($line -match '^([^=][^=]*)=(.*)$') {
            Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
        }
    }
}

function Find-CMake {
    param([string]$VsPath)

    $command = Get-Command cmake.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $env:ProgramFiles "CMake\bin\cmake.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "CMake\bin\cmake.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\CMake\bin\cmake.exe"),
        (Join-Path $VsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

function Ensure-CMake {
    param([string]$VsPath)

    $cmake = Find-CMake $VsPath
    if ($cmake) {
        return $cmake
    }

    if ($SkipInstall) {
        throw "CMake was not found and -SkipInstall was requested."
    }

    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "winget.exe was not found. Install CMake 3.24 or newer, then rerun this script."
    }

    Write-Step "Installing CMake"
    Invoke-Checked $winget.Source @(
        "install",
        "--id", "Kitware.CMake",
        "-e",
        "--source", "winget",
        "--accept-source-agreements",
        "--accept-package-agreements",
        "--silent"
    )

    Refresh-ProcessPath
    $cmake = Find-CMake $VsPath
    if (-not $cmake) {
        throw "CMake installation finished, but cmake.exe could not be located."
    }
    return $cmake
}

function Find-Ninja {
    param([string]$VsPath)

    $command = Get-Command ninja.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidate = Join-Path $VsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
    if (Test-Path $candidate) {
        return $candidate
    }
    return $null
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildDir = Join-Path $RepoRoot "build"

if (-not (Test-Path (Join-Path $RepoRoot ".git"))) {
    throw "This script must be run from a Cugo Git checkout. Resolved repository root: $RepoRoot"
}

Write-Step "Repository"
Write-Host "Repository: $RepoRoot"
Invoke-Checked "git.exe" @("-C", $RepoRoot, "status", "--short")
Invoke-Checked "git.exe" @("-C", $RepoRoot, "log", "--oneline", "-5")

Write-Step "CUDA device and toolkit"
$nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    Invoke-Checked $nvidiaSmi.Source @()
} else {
    Write-Warning "nvidia-smi.exe was not found in PATH."
}

$nvcc = Get-Command nvcc.exe -ErrorAction SilentlyContinue
if (-not $nvcc) {
    throw "nvcc.exe was not found. Install or repair the NVIDIA CUDA Toolkit before continuing."
}
Invoke-Checked $nvcc.Source @("--version")

Write-Step "MSVC host compiler"
$vsPath = Ensure-BuildTools
Write-Host "Visual Studio: $vsPath"
Import-VsEnvironment $vsPath

$cl = Get-Command cl.exe -ErrorAction SilentlyContinue
if (-not $cl) {
    throw "cl.exe is still unavailable after importing the Visual Studio developer environment."
}
& $env:ComSpec /d /s /c "cl 2>&1"

Write-Step "CMake"
$cmake = Ensure-CMake $vsPath
$cmakeDir = Split-Path -Parent $cmake
$ctest = Join-Path $cmakeDir "ctest.exe"
if (-not (Test-Path $ctest)) {
    throw "ctest.exe was not found next to CMake: $ctest"
}
Invoke-Checked $cmake @("--version")

$ninja = Find-Ninja $vsPath
if ($ninja) {
    $generator = "Ninja"
    $ninjaDir = Split-Path -Parent $ninja
    if (($env:Path -split ';') -notcontains $ninjaDir) {
        $env:Path = "$ninjaDir;$env:Path"
    }
    Write-Host "Generator: Ninja"
    Invoke-Checked $ninja @("--version")
} else {
    $generator = "NMake Makefiles"
    Write-Host "Generator: NMake Makefiles"
}

Write-Step "Clean configure"
if (Test-Path $BuildDir) {
    Remove-Item -Recurse -Force $BuildDir
}

Invoke-Checked $cmake @(
    "-S", $RepoRoot,
    "-B", $BuildDir,
    "-G", $generator,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_TESTING=ON",
    "-DCUGO_ENABLE_CUDA=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=native"
)

Write-Step "Release build"
if ($generator -eq "Ninja") {
    Invoke-Checked $cmake @("--build", $BuildDir, "--parallel")
} else {
    Invoke-Checked $cmake @("--build", $BuildDir)
}

Write-Step "Correctness tests"
Invoke-Checked $ctest @("--test-dir", $BuildDir, "--output-on-failure")

Write-Step "Direct test output"
$coreTest = Join-Path $BuildDir "cugo_core_test.exe"
$cudaTest = Join-Path $BuildDir "cugo_cuda_core_test.exe"

if (-not (Test-Path $coreTest)) {
    throw "CPU test executable was not produced: $coreTest"
}
if (-not (Test-Path $cudaTest)) {
    throw "CUDA test executable was not produced: $cudaTest"
}

Invoke-Checked $coreTest @()
Invoke-Checked $cudaTest @()

Write-Step "Final state"
Invoke-Checked "git.exe" @("-C", $RepoRoot, "status", "--short")
Invoke-Checked "git.exe" @("-C", $RepoRoot, "log", "--oneline", "-3")

Write-Host "`nCUGO WINDOWS CUDA BOOTSTRAP: PASS" -ForegroundColor Green
