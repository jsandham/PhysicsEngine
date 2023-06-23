@echo off

if not defined DevEnvDir (
	call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
)

:: import libraries (dll's are available by system)
set STANDARD_LIB="opengl32.lib" "user32.lib" "ole32.lib"

:: import libraries
set GLEW_LIB="../../lib"

:: static libraries
set ENGINE_LIB="../../lib"
set GOOGLETEST_LIB="../../../googletest/lib"

set ENGINE="../../include"
set GOOGLETEST="../../../googletest/include"
set GLEW="../../include/glew-2.1.0"
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
echo [92mCompiling C++ test code...[0m
for /R "../src" %%f in (*.cpp) do (
	call cl /c /I%ENGINE% /I%GLEW% /I%GOOGLETEST% %OPT% %WARN% %MODEFLAGS% %FLAGS% %%f
)

:: create list of .obj files
echo [92mCompiled objects...[0m
set OBJ_FILES=
for /r "%MODE%/obj" %%v in (*.obj) do (
	call :concat_obj %%v
	echo %%v
)

:: create test executable
echo [92mCreating test executable...[0m
link %OBJ_FILES% %STANDARD_LIB% %ENGINE_LIB%/%MODE%/engine.lib %GLEW_LIB%/%MODE%/glew32.lib %GOOGLETEST_LIB%/%MODE%/googletest.lib

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