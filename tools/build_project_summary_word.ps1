param(
    [string]$SourceJson = "D:\PythonProject-YOLO\reports\project_summary_word_source.json",
    [string]$OutputDocx = ""
)

$ErrorActionPreference = "Stop"
$wdAlignParagraphLeft = 0
$wdAlignParagraphCenter = 1
$wdAlignParagraphRight = 2
$wdCollapseEnd = 0
$wdPageBreak = 7
$wdLineStyleSingle = 1
$wdBorderBottom = -3
$wdBorderTop = -1
$wdBorderLeft = -2
$wdBorderRight = -4
$wdBorderHorizontal = -5
$wdBorderVertical = -6
$wdRowHeightAtLeast = 1
$wdPaperLetter = 2
$wdOrientPortrait = 0
$wdStyleNormal = -1
$wdStyleTitle = -63
$wdStyleHeading1 = -2
$wdStyleHeading2 = -3
$wdStyleCaption = -35
$wdStyleListBullet = -49
$wdStyleListNumber = -50

function Rgb([int]$r, [int]$g, [int]$b) { return $r + 256 * $g + 65536 * $b }
$navy = Rgb 21 55 73
$blue = Rgb 30 104 145
$teal = Rgb 33 133 138
$lightBlue = Rgb 230 240 245
$pale = Rgb 244 248 250
$line = Rgb 194 210 219
$dark = Rgb 35 49 58
$muted = Rgb 92 110 120
$white = Rgb 255 255 255
$green = Rgb 228 242 235

$data = [IO.File]::ReadAllText($SourceJson, [Text.Encoding]::UTF8) | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace($OutputDocx)) {
    $OutputDocx = Join-Path (Split-Path -Parent $SourceJson) ([string]$data.output_file)
}
$word = $null
$doc = $null

function Set-RunFont($range, [double]$size, [int]$color, [bool]$bold = $false) {
    $range.Font.Name = "Calibri"
    $range.Font.NameFarEast = "Microsoft YaHei"
    $range.Font.Size = $size
    $range.Font.Color = $color
    $range.Font.Bold = $(if ($bold) { -1 } else { 0 })
}

function Add-Para($text, [object]$style = $null, [int]$align = 0, [double]$before = 0, [double]$after = 5, [double]$lineSpacing = 15) {
    $selection = $word.Selection
    $selection.EndKey(6) | Out-Null
    $start = $selection.Start
    if ($null -ne $style) { $selection.Style = $style }
    $selection.TypeText([string]$text)
    $finish = $selection.End
    $selection.TypeParagraph()
    $p = $doc.Range($start, $finish).Paragraphs.Item(1)
    $p.Alignment = $align
    $p.Format.SpaceBefore = $before
    $p.Format.SpaceAfter = $after
    if ($lineSpacing -gt 0) { $p.Format.LineSpacing = $lineSpacing }
    return $p
}

function Add-SectionHeading($text) {
    $p = Add-Para $text $wdStyleHeading1 $wdAlignParagraphLeft 14 7 18
    Set-RunFont $p.Range 16 $navy $true
    $p.KeepWithNext = -1
    $p.Borders.Item($wdBorderBottom).LineStyle = $wdLineStyleSingle
    $p.Borders.Item($wdBorderBottom).Color = $teal
    $p.Borders.Item($wdBorderBottom).LineWidth = 12
    return $p
}

function Add-Body($text) {
    $p = Add-Para $text $wdStyleNormal $wdAlignParagraphLeft 0 5 16
    Set-RunFont $p.Range 10.2 $dark $false
    return $p
}

function Add-ListItem($text, [bool]$numbered = $false) {
    $style = $(if ($numbered) { $wdStyleListNumber } else { $wdStyleListBullet })
    $p = Add-Para $text $style $wdAlignParagraphLeft 0 3 15
    Set-RunFont $p.Range 10 $dark $false
    $p.Format.LeftIndent = 18
    $p.Format.FirstLineIndent = -10
    $word.Selection.Style = $wdStyleNormal
    return $p
}

function Set-CellText($cell, $text, [bool]$header = $false) {
    $cell.Range.Text = [string]$text
    $cell.VerticalAlignment = 1
    $cell.Range.ParagraphFormat.SpaceAfter = 0
    $cell.Range.ParagraphFormat.SpaceBefore = 0
    $cell.Range.ParagraphFormat.LineSpacing = 13
    if ($header) {
        $cell.Shading.BackgroundPatternColor = $navy
        Set-RunFont $cell.Range 8.6 $white $true
    } else {
        Set-RunFont $cell.Range 8.7 $dark $false
    }
}

function Add-DataTable($spec) {
    $rows = $spec.rows.Count + 1
    $cols = $spec.headers.Count
    $range = $doc.Range($doc.Content.End - 1, $doc.Content.End - 1)
    $table = $doc.Tables.Add($range, $rows, $cols)
    $table.AllowAutoFit = $false
    $table.Rows.AllowBreakAcrossPages = 0
    $table.Rows.Item(1).HeadingFormat = -1
    $table.Borders.InsideLineStyle = $wdLineStyleSingle
    $table.Borders.OutsideLineStyle = $wdLineStyleSingle
    $table.Borders.InsideColor = $line
    $table.Borders.OutsideColor = $line
    $table.Rows.HeightRule = $wdRowHeightAtLeast
    $table.LeftPadding = 5
    $table.RightPadding = 5
    $table.TopPadding = 4
    $table.BottomPadding = 4
    for ($c = 1; $c -le $cols; $c++) {
        Set-CellText $table.Cell(1, $c) $spec.headers[$c - 1] $true
        if ($null -ne $spec.widths) { $table.Columns.Item($c).Width = [double]$spec.widths[$c - 1] * 72 }
    }
    for ($r = 2; $r -le $rows; $r++) {
        for ($c = 1; $c -le $cols; $c++) {
            Set-CellText $table.Cell($r, $c) $spec.rows[$r - 2][$c - 1] $false
            if (($r % 2) -eq 1) { $table.Cell($r, $c).Shading.BackgroundPatternColor = $pale }
        }
    }
    $after = $doc.Paragraphs.Add($doc.Range($doc.Content.End - 1, $doc.Content.End - 1))
    $after.Format.SpaceAfter = 4
    return $table
}

function Add-MetricCards($cards) {
    $range = $doc.Range($doc.Content.End - 1, $doc.Content.End - 1)
    $table = $doc.Tables.Add($range, 2, 2)
    $table.AllowAutoFit = $false
    $table.Columns.Item(1).Width = 3.2 * 72
    $table.Columns.Item(2).Width = 3.2 * 72
    $table.Borders.Enable = 0
    $table.LeftPadding = 9
    $table.RightPadding = 9
    $table.TopPadding = 7
    $table.BottomPadding = 7
    for ($i = 0; $i -lt 4; $i++) {
        $row = [Math]::Floor($i / 2) + 1
        $col = ($i % 2) + 1
        $cell = $table.Cell($row, $col)
        $cell.Shading.BackgroundPatternColor = $(if ($i -eq 0) { $lightBlue } elseif ($i -eq 1) { $green } else { $pale })
        $cell.Range.Text = ""
        $cell.Range.ParagraphFormat.SpaceAfter = 0
        $p1 = $cell.Range.Paragraphs.Add()
        $p1.Range.Text = [string]$cards[$i][0]
        Set-RunFont $p1.Range 8.5 $muted $true
        $p2 = $cell.Range.Paragraphs.Add()
        $p2.Range.Text = [string]$cards[$i][1]
        Set-RunFont $p2.Range 17 $navy $true
        $p3 = $cell.Range.Paragraphs.Add()
        $p3.Range.Text = [string]$cards[$i][2]
        Set-RunFont $p3.Range 8.2 $muted $false
    }
    $after = $doc.Paragraphs.Add($doc.Range($doc.Content.End - 1, $doc.Content.End - 1))
    $after.Format.SpaceAfter = 2
}

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Add()
    $section = $doc.Sections.Item(1)
    $section.PageSetup.PaperSize = $wdPaperLetter
    $section.PageSetup.Orientation = $wdOrientPortrait
    $section.PageSetup.TopMargin = 72
    $section.PageSetup.BottomMargin = 50
    $section.PageSetup.LeftMargin = 72
    $section.PageSetup.RightMargin = 72
    $section.PageSetup.HeaderDistance = 28
    $section.PageSetup.FooterDistance = 24

    $normal = $doc.Styles.Item($wdStyleNormal)
    $normal.Font.Name = "Calibri"
    $normal.Font.NameFarEast = "Microsoft YaHei"
    $normal.Font.Size = 10.2
    $normal.ParagraphFormat.SpaceAfter = 5

    $header = $section.Headers.Item(1)
    $header.Range.Text = "MULTISPECTRAL INSTANCE SEGMENTATION  /  PROJECT BRIEF"
    Set-RunFont $header.Range 7.5 $muted $true
    $header.Range.ParagraphFormat.Alignment = $wdAlignParagraphRight
    $header.Range.Borders.Item($wdBorderBottom).LineStyle = $wdLineStyleSingle
    $header.Range.Borders.Item($wdBorderBottom).Color = $line

    $footer = $section.Footers.Item(1)
    $footer.Range.Text = [string]$data.footer
    Set-RunFont $footer.Range 7.5 $muted $false
    $footer.Range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
    $footerPageRange = $footer.Range.Duplicate
    $footerPageRange.Collapse($wdCollapseEnd)
    $doc.Fields.Add($footerPageRange, -1, "PAGE", $true) | Out-Null

    $eyebrow = Add-Para "PROJECT STATUS / MODEL REVIEW" $wdStyleNormal $wdAlignParagraphLeft 8 10 14
    Set-RunFont $eyebrow.Range 8.5 $teal $true
    $title = Add-Para $data.title $wdStyleTitle $wdAlignParagraphLeft 0 6 30
    Set-RunFont $title.Range 25 $navy $true
    $subtitle = Add-Para $data.subtitle $wdStyleNormal $wdAlignParagraphLeft 0 3 18
    Set-RunFont $subtitle.Range 12 $blue $false
    $updated = Add-Para $data.updated $wdStyleNormal $wdAlignParagraphLeft 0 13 13
    Set-RunFont $updated.Range 8.5 $muted $false

    $calloutRange = $doc.Range($doc.Content.End - 1, $doc.Content.End - 1)
    $callout = $doc.Tables.Add($calloutRange, 1, 1)
    $callout.AllowAutoFit = $false
    $callout.Columns.Item(1).Width = 6.4 * 72
    $callout.Cell(1,1).Shading.BackgroundPatternColor = $navy
    $callout.Cell(1,1).Range.Text = $data.status
    Set-RunFont $callout.Cell(1,1).Range 10.5 $white $true
    $callout.Cell(1,1).Range.ParagraphFormat.SpaceAfter = 0
    $callout.Cell(1,1).Range.ParagraphFormat.LineSpacing = 16
    $callout.LeftPadding = 12
    $callout.RightPadding = 12
    $callout.TopPadding = 10
    $callout.BottomPadding = 10
    $doc.Paragraphs.Add($doc.Range($doc.Content.End - 1, $doc.Content.End - 1)).Format.SpaceAfter = 5

    Add-MetricCards $data.metric_cards

    foreach ($sec in $data.sections) {
        if (([string]$sec.heading -like "7.*") -or ([string]$sec.heading -like "8.*")) {
            $breakRange = $doc.Range($doc.Content.End - 1, $doc.Content.End - 1)
            $breakRange.InsertBreak($wdPageBreak)
        }
        Add-SectionHeading $sec.heading | Out-Null
        if ($null -ne $sec.paragraphs) {
            foreach ($text in $sec.paragraphs) { Add-Body $text | Out-Null }
        }
        if ($null -ne $sec.table) { Add-DataTable $sec.table | Out-Null }
        if ($null -ne $sec.bullets) {
            foreach ($text in $sec.bullets) { Add-ListItem $text $false | Out-Null }
        }
        if ($null -ne $sec.numbered) {
            foreach ($text in $sec.numbered) { Add-ListItem $text $true | Out-Null }
        }
        if ($null -ne $sec.image) {
            $p = Add-Para "" $wdStyleNormal $wdAlignParagraphCenter 3 2 12
            $shape = $p.Range.InlineShapes.AddPicture($sec.image.path, $false, $true)
            $shape.LockAspectRatio = -1
            $shape.Width = [double]$sec.image.width_inches * 72
            $shape.AlternativeText = [string]$sec.image.alt
            $cap = Add-Para $sec.image.caption $wdStyleCaption $wdAlignParagraphCenter 1 6 13
            Set-RunFont $cap.Range 8.5 $muted $false
            $cap.KeepWithNext = 0
        }
    }

    $doc.Repaginate()
    $doc.SaveAs2($OutputDocx, 16)
    Write-Output ("DOCX_CREATED=" + $OutputDocx)
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
