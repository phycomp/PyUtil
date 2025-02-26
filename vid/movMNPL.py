ffmpeg -i v0.mp4 -i v1.mp4 -i v2.mp4 -i v3.mp4 -i v4.mp4 -filter_complex \
"[0][1:v]xfade=transition=fade:duration=1:offset=3[vfade1]; \
 [vfade1][2:v]xfade=transition=fade:duration=1:offset=10[vfade2]; \
 [vfade2][3:v]xfade=transition=fade:duration=1:offset=21[vfade3]; \
 [vfade3][4:v]xfade=transition=fade:duration=1:offset=25,format=yuv420p; \
 [0:a][1:a]acrossfade=d=1[afade1]; \
 [afade1][2:a]acrossfade=d=1[afade2]; \
 [afade2][3:a]acrossfade=d=1[afade3]; \
 [afade3][4:a]acrossfade=d=1" \
-movflags +faststart out.mp4

*****************************  Overlay *********************************************
ffmpeg -loop 1 -t 8.54 -i 5.jpg -loop 1 -t 6.10 -i 1.jpg -loop 1 -t 8.86 -i 2.jpg -loop 1 -t 8.66 -i 3.jpg -i 4.jpg -filter_complex "[4]split=2[color][alpha];
[color]crop=iw/2:ih:0:0[color];
[alpha]crop=iw/2:ih:iw/2:0[alpha];
[color][alpha]alphamerge[ovrly];
[0][1]xfade=transition=circleopen:duration=1.00:offset=7.54[v0];
[v0][2]xfade=transition=diagbl:duration=1.00:offset=12.14[v1];
[v1][3]xfade=transition=slideright:duration=1.00:offset=20.00[v2];
[v2][4]xfade=transition=slideright:duration=1.00:offset=27.00[v3];
[v3]concat=n=1:v=1:a=0,format=yuv420p[concatenated_video];
[concatenated_video][ovrly]overlay=0:0" output.mp4
**************************** png2clip **********************************************
  #!/bin/bash
LST=($(ls -1 *.jpg))
TOT=${#LST[*]}
f="${LST[0]}"
INP=("-loop" "1" "-t" "5" "-i" "$f")
echo $f
FLS="[0]scale=-2:720,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[0p]"
PDX="[0p]"
OFS=0
for (( i=1; i<=$(( $TOT -1 )); i++ )); do
  f="${LST[$i]}"
  INP+=("-loop" "1" "-t" "5" "-i" "$f")
  ((OFS += 4))

  PDS="[${i}p]"
  FLS+=";[${i}]scale=-2:720,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1${PDS}"
  FLX+=";${PDX}${PDS}xfade=duration=1:offset=${OFS}"
  PDX="[${i}x]"
  FLX+="${PDX}"

  echo $OFS $f
done
#echo "${INP[@]}" -filter_complex "$FLS $FLX" -map $PDX -c:v h264_nvenc -cq 20 -y /tmp/output.mp4 -hide_banner'

echo "ffmpeg ${INP[@]} -filter_complex \"$FLS $FLX\" -map \"$PDX\" -c:v h264_nvenc -cq 20 -y /tmp/output.mp4 -hide_banner"
