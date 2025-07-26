@echo off
echo Cleaning up PDF RAG Chatbot project...

cd "C:\Users\user\Desktop\Applications\RAG system"

echo Deleting unnecessary files...

if exist "install_fix.bat" (
    del "install_fix.bat"
    echo ✅ Deleted install_fix.bat
)

if exist "test_installation.py" (
    del "test_installation.py" 
    echo ✅ Deleted test_installation.py
)

if exist "QUICK_START.md" (
    del "QUICK_START.md"
    echo ✅ Deleted QUICK_START.md  
)

if exist "requirements_compatible.txt" (
    del "requirements_compatible.txt"
    echo ✅ Deleted requirements_compatible.txt
)

if exist "RAG_Application" (
    rmdir /s /q "RAG_Application"
    echo ✅ Deleted RAG_Application folder
)

echo.
echo 🎉 Project cleanup complete!
echo.
echo Clean project structure:
dir /b

echo.
echo ✨ Your project is now clean and GitHub ready!
pause