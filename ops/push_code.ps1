param(
    [Parameter(Mandatory = $true)]
    [string]$Message,
    [string]$Remote = "origin",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))

git -C $RepoRoot rev-parse --is-inside-work-tree | Out-Null
git -C $RepoRoot add -A
python (Join-Path $PSScriptRoot "check_repository.py") --staged
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

git -C $RepoRoot diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "没有需要提交的变更。"
    exit 0
}

git -C $RepoRoot commit -m $Message
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
git -C $RepoRoot push --set-upstream $Remote $Branch
exit $LASTEXITCODE
