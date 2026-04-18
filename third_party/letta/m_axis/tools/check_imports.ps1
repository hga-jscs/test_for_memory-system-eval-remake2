param(
  [string]$TargetDir = "m_axis/src"
)

$denyPatterns = @(
  "letta\.agents",
  "letta\.server",
  "server\.rest_api",
  "tool_executor",
  "summarizer"
)

Write-Host "[check_imports] scanning: $TargetDir"

if (-not (Test-Path $TargetDir)) {
  Write-Host "[check_imports] ERROR: target dir not found: $TargetDir"
  exit 2
}

$found = $false
foreach ($pat in $denyPatterns) {
  $hits = Select-String -Path (Join-Path $TargetDir "*.py") -Pattern $pat -AllMatches -ErrorAction SilentlyContinue
  if ($hits) {
    Write-Host "[check_imports] DENY pattern matched: $pat"
    $hits | ForEach-Object { Write-Host ("  " + $_.Path + ":" + $_.LineNumber + "  " + $_.Line.Trim()) }
    $found = $true
  }
}

if ($found) {
  Write-Host "[check_imports] FAILED"
  exit 1
}

Write-Host "[check_imports] OK"
exit 0
