# Schedule a reboot at 3:00 AM today (or tomorrow if past 3 AM)
$now = Get-Date
if ($now.Hour -ge 3) {
    $rebootDate = $now.AddDays(1).Date.AddHours(3)
} else {
    $rebootDate = $now.Date.AddHours(3)
}

$dateStr = $rebootDate.ToString("yyyy/MM/dd")
$timeStr = "03:00"

$logFile = "E:\RAG\researchradar\data\raw\reboot_schedule.txt"

"Scheduling reboot for $dateStr $timeStr" | Out-File $logFile

schtasks /Create /TN "ResearchRadar-Reboot" /SC ONCE /SD $dateStr /ST $timeStr /TR "shutdown /r /t 60" /RL HIGHEST /F 2>&1 | Out-File $logFile -Append

schtasks /Query /TN "ResearchRadar-Reboot" /V /FO LIST 2>&1 | Out-File $logFile -Append
