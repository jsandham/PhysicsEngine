@echo off

if not defined DevEnvDir (
	call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
)

rem set GLEW="C:\Program Files (x86)\glew-2.1.0\include"
set GLEW="C:\Users\jsand\Documents\PhysicsEngine\engine\include\glew-2.1.0"
rem set CUDA="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include"
set ENGINE="C:\Users\jsand\Documents\PhysicsEngine\engine\include"
set PROJECT="C:\Users\jsand\Documents\PhysicsEngine\sample_project\Demo\Demo\include"

set CompilerFlags= /MDd -Oi -W4 -wd4201 -wd4189 -wd4100 -wd4530 -wd4996 -wd4127 -wd4211 -wd4512 -wd4458 -Zi /Fo"obj"\ /Fd"obj"\
set Libs=user32.lib gdi32.lib xinput.lib ole32.lib opengl32.lib glew32.lib cudart.lib engine.lib freetype.lib

:: copy engine lib to scene to binary bin folder
copy "C:\Users\jsand\Documents\PhysicsEngine\engine\lib\engine.lib" "C:\Users\jsand\Documents\PhysicsEngine\tools\scene_to_binary\bin"

:: compile project c++ files to obj
cl /c /I%PROJECT% /I%ENGINE% /I%GLEW% %CompilerFlags% ..\..\..\sample_project\Demo\Demo\src\Load.cpp
cl /c /I%PROJECT% /I%ENGINE% /I%GLEW% %CompilerFlags% ..\..\..\sample_project\Demo\Demo\src\LogicSystem.cpp
cl /c /I%PROJECT% /I%ENGINE% /I%GLEW% %CompilerFlags% ..\..\..\sample_project\Demo\Demo\src\PlayerSystem.cpp

:: compile editor c++ files to obj
cl /c /I%PROJECT% /I%ENGINE% /I%GLEW% %CompilerFlags% ..\scene_to_binary_main.cpp

:: create list of .obj files
set OBJ_FILES=
for /r %%v in (*.obj) do call :concat_obj %%v

:: link to create editor exe
set LinkerFlags=/SUBSYSTEM:CONSOLE /OPT:REF /OPT:ICF /TLBID:1 /DYNAMICBASE /NXCOMPAT /MACHINE:X64 /OUT:scene_to_binary.exe /DEBUG

link %LinkerFlags% %OBJ_FILES% %Libs%

goto :eof
:concat_obj
set OBJ_FILES=%OBJ_FILES% %1
goto :eof