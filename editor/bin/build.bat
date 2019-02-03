@echo off

copy "bundle.assets" "C:\Users\James\Documents\PhysicsEngine\sample_project\Demo\x64\Debug"
copy "bundle.assets" "C:\Users\James\Documents\PhysicsEngine\sample_project\Demo\x64\Release"

for /R C:\Users\James\Documents\PhysicsEngine\editor\bin %%f in (*.scene) do copy %%f C:\Users\James\Documents\PhysicsEngine\sample_project\Demo\x64\Debug
for /R C:\Users\James\Documents\PhysicsEngine\editor\bin %%f in (*.scene) do copy %%f C:\Users\James\Documents\PhysicsEngine\sample_project\Demo\x64\Release





rem if not defined DevEnvDir (
rem 	call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" x64
rem )

rem :: copy engine lib to editor bin folder
rem copy "C:\Users\James\Documents\PhysicsEngine\engine\lib\engine.lib" "C:\Users\James\Documents\PhysicsEngine\editor\bin"

rem :: create list of .obj files
rem set OBJ_FILES=
rem for /r %%v in (*.obj) do call :concat_obj %%v

rem :: link to create project exe
rem set Libs=user32.lib gdi32.lib xinput.lib ole32.lib opengl32.lib glew32.lib cudart.lib engine.lib
rem set LinkerFlags=/SUBSYSTEM:WINDOWS /OPT:REF /OPT:ICF /TLBID:1 /DYNAMICBASE /NXCOMPAT /MACHINE:X64 /OUT:main.exe /DEBUG

rem link %LinkerFlags% %OBJ_FILES% %Libs%

rem :: copy executable and dll's to project bin folder
rem copy "main.exe" "C:\Users\James\Documents\PhysicsEngine\sample_project\bin"
rem copy "bundle.assets" "C:\Users\James\Documents\PhysicsEngine\sample_project\bin"
rem copy "cudart32_75.dll" "C:\Users\James\Documents\PhysicsEngine\sample_project\bin"
rem copy "cudart64_75.dll" "C:\Users\James\Documents\PhysicsEngine\sample_project\bin"
rem copy "glew32.dll" "C:\Users\James\Documents\PhysicsEngine\sample_project\bin"

rem goto :eof
rem :concat_obj
rem set OBJ_FILES=%OBJ_FILES% %1
rem goto :eof