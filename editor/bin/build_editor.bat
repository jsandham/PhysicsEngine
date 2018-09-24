@echo off

if not defined DevEnvDir (
	call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" x64
)

set GLEW="C:\Program Files (x86)\glew-2.1.0\include"
set CUDA="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include"
set ENGINE="C:\Users\James\Documents\PhysicsEngine\engine\include"

set CompilerFlags=-Oi -W4 -wd4201 -wd4189 -wd4100 -wd4530 -wd4996 -wd4127 -wd4211 -Zi
set Libs=engine.lib glew32.lib opengl32.lib cudart.lib
set LinkerFlags=/SUBSYSTEM:CONSOLE /OPT:REF /OPT:ICF /TLBID:1 /DYNAMICBASE /NXCOMPAT /MACHINE:X64 /OUT:editor.exe /DEBUG

rem mkdir build
rem pushd build

:: compile editor c++ files to obj
cl /c /I%ENGINE% /I%GLEW% /I%CUDA% %CompilerFlags% ..\src\MeshLoader.cpp
cl /c /I%ENGINE% /I%GLEW% /I%CUDA% %CompilerFlags% ..\src\main.cpp

:: link to create editor exe
link %LinkerFlags% MeshLoader.obj main.obj %Libs%

popd