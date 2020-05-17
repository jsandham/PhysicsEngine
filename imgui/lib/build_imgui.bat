@echo off

if not defined DevEnvDir (
	call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
)

set GLEW="../include"
set WARN=-W4 
set OPT=/Od
set MODEFLAGS=/MDd -Zi /Fo"debug/obj"\ /Fd"debug/obj"\ 
set MODE=debug
set FLAGS=-nologo

:: run through batch file parameter inputs
for %%x in (%*) do (
	if "%%x"=="/help" (
		echo "help"
	)
	if "%%x"=="/O2" (
		set OPT=/O2
	)
	if "%%x"=="/debug" (
		set MODE=debug
		set MODEFLAGS=/MDd -Zi /Fo"debug/obj"\ /Fd"debug/obj"\ 
	)
	if "%%x"=="/release" (
		set MODE=release
		set MODEFLAGS=/MD /Fo"release/obj"\ /Fd"release/obj"\ 
	)
)

:: print build settings
echo [92mBuild mode: %MODE%[0m
echo [92mOptimization level: %OPT%[0m	

:: compile c++ code

echo [92mCompiling C++ imgui code...[0m
for /R "../src" %%f in (*.cpp) do (
	call cl /c /I%GLEW% %OPT% %WARN% %MODEFLAGS% %FLAGS% %%f
)

:: create list of .obj files
echo [92mCompiled objects...[0m
set OBJ_FILES=
for /r "%MODE%/obj" %%v in (*.obj) do (
	call :concat_obj %%v
	echo %%v
)

:: create static imgui library
echo [92mCreating static imgui library...[0m
lib /out:%MODE%/imgui.lib %OBJ_FILES%

:: delete .obj fles
echo [92mDeleting objects...[0m
set OBJ_FILES=
for /r "%MODE%/obj" %%v in (*.obj) do (
	del /s %%v
)

goto :eof
:concat_obj
set OBJ_FILES=%OBJ_FILES% %1
goto :eof