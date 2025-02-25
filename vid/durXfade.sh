ffmpeg \
-loop 1 -t 5 -i 1.jpg \
-loop 1 -t 5 -i 2.jpg \
-loop 1 -t 5 -i 3.jpg \
-loop 1 -t 5 -i 4.jpg \
-loop 1 -t 5 -i 5.jpg \
-filter_complex \
"[0]setdar=16/9[s0]; \
[1]setdar=16/9[s1]; \
[2]setdar=16/9[s2]; \
[3]setdar=16/9[s3]; \
[4]setdar=16/9[s4]; \
[0:v]fade=t=out:st=4:d=1[v0]; \
[1:v]fade=t=in:st=0:d=1,fade=t=out:st=4:d=1[v1]; \
[2:v]fade=t=in:st=0:d=1,fade=t=out:st=4:d=1[v2]; \
[3:v]fade=t=in:st=0:d=1,fade=t=out:st=4:d=1[v3]; \
[4:v]fade=t=in:st=0:d=1,fade=t=out:st=4:d=1[v4]; \

[v0][v1][v2][v3][v4]concat=n=5:v=1:a=0,format=yuv420p[v]" -map "[v]" out.mp4

xfade=transition=fade:duration=0.5:offset=408.84
