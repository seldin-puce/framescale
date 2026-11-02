# Authenticode signing hook. Tauri calls it once per file it signs
# (main exe, sidecars, NSIS installer).
#
# Uses Azure Artifact Signing (formerly Trusted Signing): certificates are
# HSM-backed and nothing ever sits on disk. Required environment (CI secrets):
#   AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET   (service principal)
#   FS_SIGN_DLIB      path to Azure.CodeSigning.Dlib.dll
#   FS_SIGN_METADATA  path to metadata.json {Endpoint, CodeSigningAccountName, CertificateProfileName}
#   FS_SIGNTOOL       (optional) explicit path to signtool.exe
#
# Local dev builds without these variables skip signing, with a warning.
param([Parameter(Mandatory = $true)][string]$File)
$ErrorActionPreference = 'Stop'

if (-not $env:FS_SIGN_DLIB -or -not $env:FS_SIGN_METADATA) {
    Write-Warning "Code signing skipped for '$File' (FS_SIGN_DLIB / FS_SIGN_METADATA not set)."
    exit 0
}

$signtool = $env:FS_SIGNTOOL
if (-not $signtool) {
    $signtool = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\signtool.exe" |
        Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $signtool) { throw "signtool.exe not found; set FS_SIGNTOOL." }

& $signtool sign /v /fd SHA256 `
    /tr "http://timestamp.acs.microsoft.com" /td SHA256 `
    /dlib $env:FS_SIGN_DLIB /dmdf $env:FS_SIGN_METADATA `
    $File
if ($LASTEXITCODE -ne 0) { throw "signtool failed ($LASTEXITCODE) for '$File'" }
