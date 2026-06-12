@echo off
setlocal enabledelayedexpansion
set "SCRIPT=%~dp0run.ps1"
set "ARGS="
:collect
if "%~1"=="" goto run
set "ARG=%~1"
if /I "!ARG!"=="-Debug" set "ARG=-AuditDebug"
if /I "!ARG!"=="/Debug" set "ARG=-AuditDebug"
set "ARGS=!ARGS! ^"!ARG!^""
shift
goto collect
:run
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %ARGS%
