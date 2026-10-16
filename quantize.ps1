# Prefer the environment explicitly activated by the user.
if ($env:CONDA_PREFIX) {
    if ($env:CONDA_DEFAULT_ENV -eq 'base') {
        throw 'Activate the onnx-edge-quantizer Conda environment first.'
    }
    $taskPythonExecutable = Join-Path $env:CONDA_PREFIX 'python.exe'
} elseif ($env:VIRTUAL_ENV) {
    $taskPythonExecutable = Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe'
} else {
    $taskPythonCandidates = @(
        (Join-Path $PSScriptRoot '.venv312\Scripts\python.exe'),
        (Join-Path $PSScriptRoot '.venv\Scripts\python.exe')
    )
    $taskPythonExecutable = $taskPythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $taskPythonExecutable) {
    throw 'Activate the onnx-edge-quantizer Conda environment and run python scripts/install.py first. See README.md.'
}
if (-not (Test-Path -LiteralPath $taskPythonExecutable)) {
    throw "Python was not found in the selected environment: $taskPythonExecutable"
}
& $taskPythonExecutable -m onnx_edge_quantizer @args
exit $LASTEXITCODE
