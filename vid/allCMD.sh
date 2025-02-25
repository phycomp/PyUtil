ffmpeg -i 新民start.mp4 -c copy -bsf:v h264_mp4toannexb -f mpegts -y 新民start.ts
ffmpeg -i 三分餘地.mp4 -c copy -bsf:v h264_mp4toannexb -f mpegts -y 三分餘地.ts
ffmpeg -i 白水老人.mp4 -c copy -bsf:v h264_mp4toannexb -f mpegts -y 白水老人.ts
ffmpeg -i 知命立命.mp4 -c copy -bsf:v h264_mp4toannexb -f mpegts -y 知命立命.ts
ffmpeg -i 論語1黃講師.mp4 -c copy -bsf:v h264_mp4toannexb -f mpegts -y 論語1黃講師.ts
ffmpeg -i 論語2楊講師.mp4 -c copy -bsf:v h264_mp4toannexb -f mpegts -y 論語2楊講師.ts
ffmpeg -i 論語3.mp4 -c copy -bsf:v h264_mp4toannexb -f mpegts -y 論語3.ts
ffmpeg -i 班務辦道.mp4 -c copy -bsf:v h264_mp4toannexb -f mpegts -y 班務辦道.ts
ffmpeg -i 知心之旅.mp4 -c copy -bsf:v h264_mp4toannexb -f mpegts -y 知心之旅.ts
ffmpeg -i 好吃餐點.mp4 -c copy -bsf:v h264_mp4toannexb -f mpegts -y 好吃餐點.ts
ffmpeg -i "concat:新民start.ts|三分餘地.ts|白水老人.ts|知命立命.ts|論語1黃講師.ts|論語2楊講師.ts|論語3.ts|班務辦道.ts|知心之旅.ts|好吃餐點.ts" -c copy -y stage1.mp4
ffmpeg -i stage1.mp4 -c copy -an -y noAudio.mp4
ffmpeg -i noAudio.mp4 -f concat -i allMP3 -c:v copy -y -shortest stage2.mp4

ffmpeg -r 1/5 -i C:\data-Sam\320.jpg -c:v libx264 -r 30 -pix_fmt yuv420p C:\data-Sam\out.mp4
ffmpeg -f image2 -r 60 -i path/filename%03d.jpg -vcodec libx264 -crf 18  -pix_fmt yuv420p test.mp4

#Fadein &Fadeout effect
ffmpeg -r 1/5 -i %03d.jpg -c:v libx264 -r 30 -y -pix_fmt yuv420p slide.mp4
ffmpeg -i slide.mp4 -y -vf fade=in:0:30 slideFadein.mp4
ffmpeg -i slideFadein.mp4 -y -vf fade=out:120:30 slideFadeout.mp4
