echo off
REM 2021-11-23, Stefan Holzinger
REM
REM helper file for speeding up a video with name "animation"

IF EXIST animation_speedUp.mp4 (
    echo "animation_speedUp.mp4 already exists! rename the file"
) ELSE (
	"C:\Program Files (x86)\FFMPEG\bin\ffmpeg.exe" -i animation.mp4 -r 16 -filter:v "setpts=0.25*PTS" animation_speedUp.mp4
)