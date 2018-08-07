@echo off

if not defined DevEnvDir (
	call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" x64
)

set ENGINE="C:\Users\James\Documents\PhysicsEngine\engine\include"

set CompilerFlags=-Oi -W4 -wd4201 -wd4189 -wd4100 -wd4530 -wd4996 -wd4127 -wd4211 -Zi
set Libs=engine.lib
set LinkerFlags=/SUBSYSTEM:CONSOLE /OPT:REF /OPT:ICF /TLBID:1 /DYNAMICBASE /NXCOMPAT /MACHINE:X64

mkdir build
pushd build

:: compile editor c++ files to obj
cl /c /I%ENGINE% %CompilerFlags% ..\editor\src\MeshLoader.cpp
cl /c /I%ENGINE% %CompilerFlags% ..\editor\src\main.cpp

:: link 
link %LinkerFlags% MeshLoader.obj main.obj %Libs%

popd