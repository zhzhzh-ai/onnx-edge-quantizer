# Use the installed project environment; all arguments pass directly to the CLI.
$taskPythonCandidates = @(
    (Join-Path $PSScriptRoot '.venv312\Scripts\python.exe'),
    (Join-Path $PSScriptRoot '.venv\Scripts\python.exe')
)
$taskPythonExecutable = $taskPythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $taskPythonExecutable) {
    throw 'Create a Python 3.12 virtual environment and install this project first. See README.md.'
}
& $taskPythonExecutable -m onnx_edge_quantizer @args
exit $LASTEXITCODE
