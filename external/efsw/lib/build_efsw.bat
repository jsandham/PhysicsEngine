@echo off

set SRC_INCL="../src"
SET INCL="../INCLUDE"
set WARN=-W4 -wd4701
set OPT=/Od
set MODEFLAGS=/MDd /EHsc -Zi /Fo"debug/obj"\ /Fd"debug/obj"\ 
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
		set MODEFLAGS=/MDd /EHsc -Zi /Fo"debug/obj"\ /Fd"debug/obj"\ 
	)
	if "%%x"=="/release" (
		set MODE=release
		set MODEFLAGS=/MD /EHsc /Fo"release/obj"\ /Fd"release/obj"\ 
	)
)

:: print build settings
echo [92mBuild mode: %MODE%[0m
echo [92mOptimization level: %OPT%[0m	

:: compile c++ code

echo [92mCompiling C++ efsw code...[0m
for /R "../src/efsw" %%f in (*.cpp) do (
	call cl /c /I%SRC_INCL% /I%INCL% %OPT% %WARN% %MODEFLAGS% %FLAGS% %%f
)

:: create list of .obj files
echo [92mCompiled objects...[0m
set OBJ_FILES=
for /r "%MODE%/obj" %%v in (*.obj) do (
	call :concat_obj %%v
)

:: create static efsw library
echo [92mCreating static efsw library...[0m
lib /nologo /out:%MODE%/efsw.lib %OBJ_FILES%

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