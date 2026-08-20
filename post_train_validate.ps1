param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingProcessId,

    [string]$ExperimentName = "exp_ca_5ch_grouped_reviewed_v1"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$experimentDir = Join-Path $projectRoot "runs\segment\land_seg\$ExperimentName"
$checkpoint = Join-Path $experimentDir "weights\best.pt"
$trainingLog = Join-Path $projectRoot "logs\train_grouped_reviewed.stdout.log"
$statusPath = Join-Path $projectRoot "logs\post_train_validation.status.json"
$validationStdout = Join-Path $projectRoot "logs\post_train_validation.stdout.log"
$validationStderr = Join-Path $projectRoot "logs\post_train_validation.stderr.log"
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$validator = Join-Path $projectRoot "validate_5ch.py"

function Save-Status {
    param(
        [string]$State,
        [string]$Message
    )

    [PSCustomObject]@{
        state = $State
        message = $Message
        updated_at = (Get-Date).ToString("o")
        training_pid = $TrainingProcessId
        experiment = $experimentDir
        checkpoint = $checkpoint
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

try {
    Save-Status -State "waiting" -Message "Waiting for training to finish."
    if (Get-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue) {
        Wait-Process -Id $TrainingProcessId
    }
    Start-Sleep -Seconds 5

    if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
        throw "Training ended without best.pt."
    }
    if (-not (Test-Path -LiteralPath $trainingLog -PathType Leaf)) {
        throw "Training stdout log is missing."
    }
    if ((Get-Content -LiteralPath $trainingLog -Raw) -notmatch "Results saved to") {
        throw "Training did not reach a normal completion marker; automatic validation was skipped."
    }

    Save-Status -State "validating" -Message "Validating the valid split."
    & $python $validator `
        --checkpoint $checkpoint `
        --split val `
        --output (Join-Path $projectRoot "reports\Data_v3_model_validation\val") `
        --imgsz 1024 `
        --batch 1 `
        --workers 0 `
        --device 0 1>> $validationStdout 2>> $validationStderr
    if ($LASTEXITCODE -ne 0) {
        throw "Valid-split validation failed with exit code $LASTEXITCODE."
    }

    Save-Status -State "validating" -Message "Validating the test split."
    & $python $validator `
        --checkpoint $checkpoint `
        --split test `
        --output (Join-Path $projectRoot "reports\Data_v3_model_validation\test") `
        --imgsz 1024 `
        --batch 1 `
        --workers 0 `
        --device 0 1>> $validationStdout 2>> $validationStderr
    if ($LASTEXITCODE -ne 0) {
        throw "Test-split validation failed with exit code $LASTEXITCODE."
    }

    Save-Status -State "complete" -Message "Training and both validations completed successfully."
}
catch {
    Save-Status -State "failed" -Message $_.Exception.Message
    $_ | Out-String | Add-Content -LiteralPath $validationStderr -Encoding UTF8
    exit 1
}
