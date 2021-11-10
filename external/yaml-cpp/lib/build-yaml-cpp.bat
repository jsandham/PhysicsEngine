@echo off

set YAML="../include"
set OPT=/Od
set MODEFLAGS=/FS /MDd -Zi /Fo"debug/obj"\ /Fd"debug/obj"\ 
set MODE=debug
set FLAGS=-nologo /EHsc

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
		set MODEFLAGS=/FS /MDd -Zi /Fo"debug/obj"\ /Fd"debug/obj"\ 
	)
	if "%%x"=="/release" (
		set MODE=release
		set MODEFLAGS=/FS /MD /Fo"release/obj"\ /Fd"release/obj"\ 
	)
)

:: print build settings
echo [92mBuild mode: %MODE%[0m
echo [92mOptimization level: %OPT%[0m	

:: compile c++ code
echo [92mCompiling C++ yaml-cpp code...[0m
for /R "../src/" %%f in (*.cpp) do (
	call cl /c /std:c++17 /I%YAML% %OPT% %MODEFLAGS% %FLAGS% %%f
)

:: create list of .obj files
echo [92mCompiled objects...[0m
set OBJ_FILES=
for /r "%MODE%/obj" %%v in (*.obj) do (
	call :concat_obj %%v
)

:: create static yaml-cpp library
echo [92mCreating static yaml-cpp library...[0m
if %MODE%==debug (
	lib /nologo /out:%MODE%/yaml-cppd.lib %OBJ_FILES%
)
if %MODE%==release (
	lib /nologo /out:%MODE%/yaml-cpp.lib %OBJ_FILES%
)

:: delete .obj fles
::echo [92mDeleting objects...[0m
::set OBJ_FILES=
::for /r "%MODE%/obj" %%v in (*.obj) do (
::	del /s %%v
::)

goto :eof
:concat_obj
set OBJ_FILES=%OBJ_FILES% %1
goto :eof