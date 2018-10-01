@echo off

if not defined DevEnvDir (
	call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" x64
)

rem set GLEW="C:\Program Files (x86)\glew-2.1.0\include"
rem set CUDA="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include"
rem set ENGINE="C:\Users\James\Documents\PhysicsEngine\engine\include"
rem set PROJECT="C:\Users\James\Documents\PhysicsEngine\sample_project\include"

rem set CompilerFlags=-Oi -W4 -wd4201 -wd4189 -wd4100 -wd4530 -wd4996 -wd4127 -wd4211 -Zi
set Libs=user32.lib gdi32.lib xinput.lib opengl32.lib glew32.lib cudart.lib engine.lib
set LinkerFlags=/SUBSYSTEM:WINDOWS /OPT:REF /OPT:ICF /TLBID:1 /DYNAMICBASE /NXCOMPAT /MACHINE:X64 /OUT:main.exe /DEBUG

:: compile project c++ files to obj  (I could just have this done when building the editor???)
:: cl /c /I%PROJECT% /I%ENGINE% /I%GLEW% /I%CUDA% %CompilerFlags% ..\..\sample_project\src\systems\LoadSystem.cpp
:: cl /c /I%PROJECT% /I%ENGINE% /I%GLEW% /I%CUDA% %CompilerFlags% ..\..\sample_project\src\systems\LogicSystem.cpp

:: link to create project exe
link %LinkerFlags% LoadSystem.obj LogicSystem.obj Manager.obj win32_main.obj %Libs%

:: copy executable and dll's to project bin folder
copy "main.exe" "C:\Users\James\Documents\PhysicsEngine\sample_project\bin"
copy "cudart32_75.dll" "C:\Users\James\Documents\PhysicsEngine\sample_project\bin"
copy "cudart64_75.dll" "C:\Users\James\Documents\PhysicsEngine\sample_project\bin"
copy "glew32.dll" "C:\Users\James\Documents\PhysicsEngine\sample_project\bin"

popd