Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
projectDir = fso.GetParentFolderName(scriptDir)

pythonPath = projectDir & "\.venv\Scripts\python.exe"
appPath = projectDir & "\src\pav_detector\ui\workbench_app.py"

If Not fso.FileExists(pythonPath) Then
  shell.Popup ".venv not found. Run windows\install_windows.bat first.", 6, "PAV Detector", 48
  WScript.Quit 1
End If

cmd = "cmd /c cd /d """ & projectDir & """ && set PYTHONPATH=src && """ & pythonPath & """ -m streamlit run """ & appPath & """"

' 0 = hidden window, False = do not wait
shell.Run cmd, 0, False
