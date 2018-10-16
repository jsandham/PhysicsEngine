@echo off

if not defined DevEnvDir (
	call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" x64
)

:: copy engine lib to editor bin folder
copy "C:\Users\James\Documents\PhysicsEngine\engine\lib\engine.lib" "C:\Users\James\Documents\PhysicsEngine\editor\bin"
copy "C:\Users\James\Documents\PhysicsEngine\engine\lib\obj\win32_main.obj" "C:\Users\James\Documents\PhysicsEngine\editor\bin\obj"
copy "C:\Users\James\Documents\PhysicsEngine\engine\lib\obj\Manager.obj" "C:\Users\James\Documents\PhysicsEngine\editor\bin\obj"

:: create list of .obj files
set OBJ_FILES=
for /r %%v in (*.obj) do call :concat_obj %%v

:: link to create project exe
set Libs=user32.lib gdi32.lib xinput.lib opengl32.lib glew32.lib cudart.lib engine.lib
set LinkerFlags=/SUBSYSTEM:WINDOWS /OPT:REF /OPT:ICF /TLBID:1 /DYNAMICBASE /NXCOMPAT /MACHINE:X64 /OUT:main.exe /DEBUG

link %LinkerFlags% %OBJ_FILES% %Libs%

:: copy executable and dll's to project bin folder
copy "main.exe" "C:\Users\James\Documents\PhysicsEngine\sample_project\bin"
copy "cudart32_75.dll" "C:\Users\James\Documents\PhysicsEngine\sample_project\bin"
copy "cudart64_75.dll" "C:\Users\James\Documents\PhysicsEngine\sample_project\bin"
copy "glew32.dll" "C:\Users\James\Documents\PhysicsEngine\sample_project\bin"

goto :eof
:concat_obj
set OBJ_FILES=%OBJ_FILES% %1
goto :eof