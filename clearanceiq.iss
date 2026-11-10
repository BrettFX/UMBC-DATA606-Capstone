; Inno Setup script for the standalone ClearanceIQ Windows installer.
; Wraps the PyInstaller onedir build (dist/ClearanceIQ/, produced by clearanceiq.spec) in a traditional
; wizard installer: the user picks the install folder (not locked to Program Files), gets a Start Menu
; entry, an optional desktop shortcut, and a proper uninstaller registered in Windows.
;
; Build locally, from the repo root, after `pyinstaller clearanceiq.spec` has produced dist/ClearanceIQ/:
;     iscc clearanceiq.iss /DAppVersion=0.1.0
; This writes installer-output\ClearanceIQ-Setup.exe. CI (.github/workflows/release.yml) runs the same
; command on windows-latest and uploads that file as a release asset alongside the portable .zip.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{AF86FDDC-728E-46AF-BB6D-7D5D06CEEE1D}
AppName=ClearanceIQ
AppVersion={#AppVersion}
AppPublisher=Brett Allen
AppPublisherURL=https://github.com/BrettFX/UMBC-DATA606-Capstone
DefaultDirName={autopf}\ClearanceIQ
DefaultGroupName=ClearanceIQ
DisableProgramGroupPage=yes
; Lets the installer run either per-machine (admin, Program Files) or per-user (no admin, any folder the
; user picks) -- the wizard's Destination Location page is enabled by default, so either way the user can
; browse to install wherever they want.
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=installer-output
OutputBaseFilename=ClearanceIQ-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName=ClearanceIQ

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "dist\ClearanceIQ\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\ClearanceIQ"; Filename: "{app}\ClearanceIQ.exe"
Name: "{group}\Uninstall ClearanceIQ"; Filename: "{uninstallexe}"
Name: "{autodesktop}\ClearanceIQ"; Filename: "{app}\ClearanceIQ.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\ClearanceIQ.exe"; Description: "Launch ClearanceIQ now"; Flags: nowait postinstall skipifsilent
