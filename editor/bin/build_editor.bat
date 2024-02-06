@echo off

set GLEW="../../external/glew-2.1.0"
set GLM="../../external/glm"
set YAML="../../external/yaml-cpp/include"
set IMGUI="../../external/imgui/include"
set IMGUIZMO="../../external/imguizmo/include"
set IMGUICOLORTEXTEDIT="../../external/imguicolortextedit/include"
set FILEWATCH="../../external/efsw/include"
set ENGINE="../../engine/include"
set WARN=-W4 -wd4100 -wd4996 -wd4211 -wd4201
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
echo [92mBuild modeflags: %MODEFLAGS%[0m
echo [92mOptimization level: %OPT%[0m	

:: create list of source files
echo [92mSource files...[0m
set SRC_FILES=
for /r "../src" %%v in (*.cpp) do (
	call :concat_src %%v
)

:: create editor executable
echo [92mCreating editor executable...[0m

:: static libraries
set YAML_LIB="../../external/yaml-cpp/lib/%MODE%/yaml-cppd.lib"
if %MODE%==release (
	set YAML_LIB="../../external/yaml-cpp/lib/%MODE%/yaml-cpp.lib"
)
set IMGUI_LIB="../../external/imgui/lib/%MODE%/imgui.lib"
set IMGUIZMO_LIB="../../external/imguizmo/lib/%MODE%/imguizmo.lib"
set IMGUICOLORTEXTEDIT_LIB="../../external/imguicolortextedit/lib/%MODE%/imguicolortextedit.lib"
set FILEWATCH_LIB="../../external/efsw/lib/%MODE%/efsw.lib"
set ENGINE_LIB="../../engine/lib/%MODE%/engine.lib"

::import libraries
set GLEW_LIB="../../external/glew-2.1.0/lib/%MODE%/glew32.lib"
set CUDART_LIB="%CUDA_PATH:"=%\lib\x64\cudart.lib"

echo %CUDART_LIB%

set LIBS=%YAML_LIB% %IMGUI_LIB% %IMGUIZMO_LIB% %IMGUICOLORTEXTEDIT_LIB% %FILEWATCH_LIB% %ENGINE_LIB% %GLEW_LIB% %CUDART_LIB% opengl32.lib comdlg32.lib
set INCLUDES=/I%GLEW% /I%GLM% /I%YAML% /I%IMGUI% /I%IMGUIZMO% /I%IMGUICOLORTEXTEDIT% /I%FILEWATCH% /I%ENGINE%
cl /std:c++17 /Fe"%MODE%/EditorApplication" %OPT% %WARN% %MODEFLAGS% %FLAGS% %INCLUDES% ../EditorApplication.cpp %SRC_FILES% %LIBS%

goto :eof
:concat_src
set SRC_FILES=%SRC_FILES% %1
goto :eof