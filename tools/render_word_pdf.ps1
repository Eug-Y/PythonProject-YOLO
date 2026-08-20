param(
    [Parameter(Mandatory=$true)][string]$InputDocx,
    [Parameter(Mandatory=$true)][string]$OutputPdf
)

$ErrorActionPreference = "Stop"
$word = $null
$doc = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($InputDocx, $false, $true)
    $doc.ExportAsFixedFormat($OutputPdf, 17)
    Write-Output ("PDF_CREATED=" + $OutputPdf)
    Write-Output ("PAGES=" + $doc.ComputeStatistics(2))
}
finally {
    if ($null -ne $doc) { $doc.Close($false) }
    if ($null -ne $word) { $word.Quit() }
    if ($null -ne $doc) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($doc) }
    if ($null -ne $word) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
