@echo off

if not defined DevEnvDir (
	call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" x64
)

set CompilerFlags=-Oi -W4 -wd4201 -wd4189 -wd4100 -wd4530 -wd4996 -Zi
set LinkerFlags=/SUBSYSTEM:CONSOLE /OPT:REF /OPT:ICF /TLBID:1 /DYNAMICBASE /NXCOMPAT /MACHINE:X64

mkdir build
pushd build

:: compile editor c++ files to obj
cl /c %CompilerFlags% ..\src\editor\main.cpp

:: link 
link %LinkerFlags% main.obj

popd