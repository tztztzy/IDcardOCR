@echo off
chcp 65001
setlocal EnableDelayedExpansion

echo ==========================================
echo  身份证矫正DLL编译脚本
echo ==========================================
echo.

:: 检查是否安装了CMake和Visual Studio
where cmake >nul 2>nul
if %errorlevel% neq 0 (
    echo 错误: 未找到CMake。请先安装CMake并添加到PATH。
    exit /b 1
)

:: 设置OpenCV路径（根据你的实际路径修改）
set "OPENCV_DIR=C:\opencv\build"
if not exist "%OPENCV_DIR%" (
    echo 警告: 默认OpenCV路径 %OPENCV_DIR% 不存在。
    echo 请设置OPENCV_DIR环境变量或在脚本中修改路径。
    
    :: 尝试查找常见安装位置
    if exist "D:\opencv\build" (
        set "OPENCV_DIR=D:\opencv\build"
    ) else if exist "D:\tools\opencv\build" (
        set "OPENCV_DIR=D:\tools\opencv\build"
    ) else if exist "%ProgramFiles%\opencv\build" (
        set "OPENCV_DIR=%ProgramFiles%\opencv\build"
    ) else (
        echo 错误: 无法找到OpenCV安装目录。
        echo 请设置OPENCV_DIR环境变量或修改本脚本中的路径。
        exit /b 1
    )
)

echo 使用OpenCV路径: %OPENCV_DIR%

:: 设置构建目录
set "BUILD_DIR=build"

:: 清理旧构建（可选）
if "%1"=="clean" (
    echo 清理旧构建...
    if exist %BUILD_DIR% rmdir /s /q %BUILD_DIR%
)

:: 创建构建目录
echo 创建构建目录...
if not exist %BUILD_DIR% mkdir %BUILD_DIR%
cd %BUILD_DIR%

:: 运行CMake配置
echo.
echo 配置CMake...
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DOpenCV_DIR="%OPENCV_DIR%" ^
    -DCMAKE_BUILD_TYPE=Release

if %errorlevel% neq 0 (
    echo 错误: CMake配置失败。
    exit /b 1
)

:: 编译
echo.
echo 编译DLL...
cmake --build . --config Release --parallel

if %errorlevel% neq 0 (
    echo 错误: 编译失败。
    exit /b 1
)

echo.
echo ==========================================
echo  编译成功！
echo ==========================================
echo.
echo 输出文件:
echo   - bin\Release\card_correction.dll
echo   - lib\Release\card_correction.lib
echo.

:: 复制DLL到Rust项目（可选）
echo 是否复制DLL到Rust项目目录? (Y/N)
set /p COPY_DLL=
if /I "%COPY_DLL%"=="Y" (
    set "RUST_PROJECT_DIR=..\..\idcard-ocr\src-tauri"
    if not exist !RUST_PROJECT_DIR! (
        set "RUST_PROJECT_DIR=D:\Downloads\IDcardOCR\idcard-ocr\src-tauri"
    )
    
    if exist !RUST_PROJECT_DIR! (
        echo 复制 card_correction.dll 到 !RUST_PROJECT_DIR!...
        copy "bin\Release\card_correction.dll" "!RUST_PROJECT_DIR!\" /Y
        echo 复制完成。
    ) else (
        echo 警告: 未找到Rust项目目录。
    )
)

cd ..
echo.
echo 按任意键退出...
pause >nul
