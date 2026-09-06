param(
    [Parameter(Mandatory = $true)]
    [string]$Url,
    [string]$Remote = "origin"
)

$ErrorActionPreference = "Stop"
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$remotes = @(git -C $RepoRoot remote)
if ($remotes -contains $Remote) {
    git -C $RepoRoot remote set-url $Remote $Url
} else {
    git -C $RepoRoot remote add $Remote $Url
}
git -C $RepoRoot remote -v
