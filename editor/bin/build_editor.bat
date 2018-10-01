@echo off

if not defined DevEnvDir (
	call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" x64
)

set GLEW="C:\Program Files (x86)\glew-2.1.0\include"
set CUDA="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include"
set ENGINE="C:\Users\James\Documents\PhysicsEngine\engine\include"
set PROJECT="C:\Users\James\Documents\PhysicsEngine\sample_project\include"

set CompilerFlags=-Oi -W4 -wd4201 -wd4189 -wd4100 -wd4530 -wd4996 -wd4127 -wd4211 -Zi
set Libs=engine.lib glew32.lib opengl32.lib cudart.lib
set LinkerFlags=/SUBSYSTEM:CONSOLE /OPT:REF /OPT:ICF /TLBID:1 /DYNAMICBASE /NXCOMPAT /MACHINE:X64 /OUT:editor.exe /DEBUG

:: copy engine lib to editor bin folder
copy "C:\Users\James\Documents\PhysicsEngine\engine\lib\engine.lib" "C:\Users\James\Documents\PhysicsEngine\editor\bin"
copy "C:\Users\James\Documents\PhysicsEngine\engine\lib\win32_main.obj" "C:\Users\James\Documents\PhysicsEngine\editor\bin"
copy "C:\Users\James\Documents\PhysicsEngine\engine\lib\Manager.obj" "C:\Users\James\Documents\PhysicsEngine\editor\bin"

:: compile project c++ files to obj
cl /c /I%PROJECT% /I%ENGINE% /I%GLEW% /I%CUDA% %CompilerFlags% ..\..\sample_project\src\systems\LoadSystem.cpp
cl /c /I%PROJECT% /I%ENGINE% /I%GLEW% /I%CUDA% %CompilerFlags% ..\..\sample_project\src\systems\LogicSystem.cpp

:: compile editor c++ files to obj
cl /c /I%PROJECT% /I%ENGINE% /I%GLEW% /I%CUDA% %CompilerFlags% ..\src\MeshLoader.cpp
cl /c /I%PROJECT% /I%ENGINE% /I%GLEW% /I%CUDA% %CompilerFlags% ..\src\main.cpp

:: link to create editor exe
link %LinkerFlags% LoadSystem.obj LogicSystem.obj MeshLoader.obj main.obj %Libs%

popd