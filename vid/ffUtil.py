#!/usr/bin/env python3
from sys import argv
from os import system
from os.path import splitext

for file in argv[1:]:
    base, ext=splitext(file)
    base=base.replace(' ', '')
    #cmd='ffmpeg -i "%s" -codec:a libmp3lame -qscale:a 2 %s.mp3'%(file, base)
    cmd='ffmpeg -i "%s" -acodec libmp3lame %s.mp3'%(file, base)
    print(cmd)
    system(cmd)

ffmpeg -i [input_file] -vcodec copy -an [output_file]
ffmpeg -ss 00:00:10  -t 5 -i "video.mp4" -ss 0:00:01 -t 5 -i "music.m4a" -map 0:v:0 -map 1:a:0 -y out.mp4
ffmpeg -i *_tile.png -r 10 -o result.mp4
ffmpeg -i sample_video_ffmpeg.mp4 -vf ass=output_subtitle.ass output_ass.mp4
ffmpeg -i ~/Downloads/yt1s.com\ -\ 祈禱\ 翁倩玉.mp4 -vf ass=prayer2.ass -y assPrayer2.mp4
ffmpeg -i 明燈照性門.ts -i new.lightEnlight.mp3 -vf  ass=new明燈照性門.ass -shortest -y lightEnlight.mp4
ffmpeg -i ~/vidMNPL/立身行道.ts -i ~/Downloads/立身行道.mp3 -vf  ass=~/vidMNPL/立身行道2.ass -y 立身行道2.mp4

ffmpeg -i [input_file] -vcodec copy -an [output_file]

ffmpeg -ss 00:00:10  -t 5 -i "video.mp4" -ss 0:00:01 -t 5 -i "music.m4a" -map 0:v:0 -map 1:a:0 -y out.mp4

ffmpeg -i k.mp4 -q:a 0 -map a k.mp3
ffmpeg -i video.mp4 -b:a 192K -vn music.mp3

cclive --exec="ffmpeg -i %i %i.mp3;" http://www.youtube.com/watch?v=WwXVMC_3ccU
ffmpeg -i YPT09.mp3  -acodec libmp3lame -b:a 32k 疏梅弄影2.mp3
$ ffmpeg -i video.mp4 video.avi 類似地 你可以轉換媒體檔案到你選擇的任何格式 例如 為轉換 YouTube flv 格式視訊為 mpeg 格式 執行：
$ ffmpeg -i video.flv video.mpeg
如果你想維持你的源視訊檔案的質量 使用 -qscale 0 引數 $ ffmpeg -i input.webm -qscale 0 output.mp4
為檢查 FFmpeg 的支援格式的列表 執行： ffmpeg -formats
3、轉換視訊檔案到音訊檔案 轉換一個視訊檔案到音訊檔案 只需具體指明輸出格式 像 .mp3 或 .ogg 或其它任意音訊格式 上面的命令將轉換 input.mp4 視訊檔案到 output.mp3 音訊檔案 
$ ffmpeg -i input.mp4 -vn output.mp3
此外 你也可以對輸出檔案使用各種各樣的音訊轉換編碼選項 像下面演示 
$ ffmpeg -i input.mp4 -vn -ar 44100 -ac 2 -ab 320 -f mp3 output.mp3
-vn – 表明我們已經在輸出檔案中禁用視訊錄製 
-ar – 設定輸出檔案的音訊頻率 通常使用的值是22050 Hz、44100 Hz、48000 Hz 
-ac – 設定音訊通道的數目 
-ab – 表明音訊位元率 
-f – 輸出檔案格式 在我們的例項中 它是 mp3 格式 
4、更改視訊檔案的解析度 如果你想設定一個視訊檔案為指定的解析度 你可以使用下面的命令：
$ ffmpeg -i input.mp4 -filter:v scale=1280:720 -c:a copy output.mp4 或 $ ffmpeg -i input.mp4 -s 1280x720 -c:a copy output.mp4
上面的命令將設定所給定視訊檔案的解析度到 1280×720 類似地 為轉換上面的檔案到 640×480 大小 執行：
$ ffmpeg -i input.mp4 -filter:v scale=640:480 -c:a copy output.mp4 或者 ffmpeg -i input.mp4 -s 640x480 -c:a copy output.mp4 這個技巧將幫助你縮放你的視訊檔案到較小的顯示裝置上 例如平板電腦和手機 
5、壓縮視訊檔案 減小媒體檔案的大小到較小來節省硬體的空間總是一個好主意 下面的命令將壓縮並減少輸出檔案的大小 
$ ffmpeg -i input.mp4 -vf scale=1280:-1 -c:v libx264 -preset veryslow -crf 24 output.mp4
請注意 如果你嘗試減小視訊檔案的大小 你將損失視訊質量 如果 24 太有侵略性 你可以降低 -crf 值到或更低值 你也可以通過下面的選項來轉換編碼音訊降低位元率 使其有立體聲感 從而減小大小 -ac 2 -c:a aac -strict -2 -b:a 128k
6、壓縮音訊檔案 正像壓縮視訊檔案一樣 為節省一些磁碟空間 你也可以使用 -ab 標誌壓縮音訊檔案 例如 你有一個 320 kbps 位元率的音訊檔案 你想通過更改位元率到任意較低的值來壓縮它 像下面 
$ ffmpeg -i input.mp3 -ab 128 output.mp3 各種各樣可用的音訊位元率列表是：96kbps 112kbps 128kbps 160kbps 192kbps 256kbps 320kbps
7、從一個視訊檔案移除音訊流 如果你不想要一個視訊檔案中的音訊 使用 -an 標誌 
$ ffmpeg -i input.mp4 -an output.mp4 在這裡 -an 表示沒有音訊錄製 上面的命令會撤銷所有音訊相關的標誌 因為我們不要來自 input.mp4 的音訊 
8、從一個媒體檔案移除視訊流 類似地 如果你不想要視訊流 你可以使用 -vn 標誌從媒體檔案中簡單地移除它 -vn 代表沒有視訊錄製 換句話說 這個命令轉換所給定媒體檔案為音訊檔案 
下面的命令將從所給定媒體檔案中移除視訊 ffmpeg -i input.mp4 -vn output.mp3 你也可以使用 -ab 標誌來指出輸出檔案的位元率 如下面的示例所示 ffmpeg -i input.mp4 -vn -ab 320 output.mp3
9、從視訊中提取影象 FFmpeg 的另一個有用的特色是我們可以從一個視訊檔案中輕鬆地提取影象 如果你想從一個視訊檔案中建立一個相簿 這可能是非常有用的 
為從一個視訊檔案中提取影象 使用下面的命令ffmpeg -i input.mp4 -r 1 -f image2 image-%2d.png
ffmpeg -i 忠義鼎訓文.mp4 -r 1/5 -f image2 Loyal%3d.png
-r – 設定幀速度 即 每秒提取幀到影象的數字 預設值是 25 
-f – 表示輸出格式 即 在我們的例項中是影象 
image-%2d.png – 表明我們如何想命名提取的影象 在這個例項中 命名應該像這樣image-01.png、image-02.png、image-03.png 等等開始 如果你使用 %3d 那麼影象的命名像 image-001.png、image-002.png 等等開始 
10、裁剪視訊
FFMpeg 允許以我們選擇的任何範圍裁剪一個給定的媒體檔案 
裁剪一個視訊檔案的語法如下給定：
ffmpeg -i input.mp4 -filter:v "crop=w:h:x:y" output.mp4
在這裡 
input.mp4 – 源視訊檔案 
-filter:v – 表示視訊過濾器 
crop – 表示裁剪過濾器 
w – 我們想自源視訊中裁剪的矩形的寬度 
h – 矩形的高度 
x – 我們想自源視訊中裁剪的矩形的 x 座標  
y – 矩形的 y 座標 
比如說你想要一個來自視訊的位置 (200,150) 且具有 640 畫素寬度和 480 畫素高度的視訊 命令應該是：
$ ffmpeg -i input.mp4 -filter:v "crop=640:480:200:150" output.mp4
請注意 剪下視訊將影響質量 除非必要 請勿剪下 
11、轉換一個視訊的具體的部分
有時 你可能想僅轉換視訊檔案的一個具體的部分到不同的格式 以示例說明 下面的命令將轉換所給定視訊input.mp4 檔案的開始 10 秒到視訊 .avi 格式 
$ ffmpeg -i input.mp4 -t 10 output.avi
在這裡 我們以秒具體說明時間 此外 以 hh.mm.ss 格式具體說明時間也是可以的 
12、設定視訊的螢幕高寬比
你可以使用 -aspect 標誌設定一個視訊檔案的螢幕高寬比 像下面 
$ ffmpeg -i input.mp4 -aspect 16:9 output.mp4
通常使用的高寬比是 16:9 4:3 16:10 5:4 2:21:1 2:35:1 2:39:1
13、新增海報影象到音訊檔案 你可以新增海報影象到你的檔案 以便影象將在播放音訊檔案時顯示 這對託管在視訊託管主機或共享網站中的音訊檔案是有用的 
$ ffmpeg -loop 1 -i inputimage.jpg -i inputaudio.mp3 -c:v libx264 -c:a aac -strict experimental -b:a 192k -shortest output.mp4
14、使用開始和停止時間剪下一段媒體檔案 可以使用開始和停止時間來剪下一段視訊為小段剪輯 我們可以使用下面的命令 
$ ffmpeg -i input.mp4 -ss 00:00:50 -codec copy -t 50 output.mp4 在這裡 
–ss – 表示視訊剪輯的開始時間 在我們的示例中 開始時間是第 50 秒 
-t – 表示總的持續時間 當你想使用開始和結束時間從一個音訊或視訊檔案剪下一部分時 它是非常有用的 類似地 我們可以像下面剪下音訊 
$ ffmpeg -i audio.mp3 -ss 00:01:54 -to 00:06:53 -c copy output.mp3
15、切分視訊檔案為多個部分 一些網站將僅允許你上傳具體指定大小的視訊 在這樣的情況下 你可以切分大的視訊檔案到多個較小的部分 像下面 
$ ffmpeg -i input.mp4 -t 00:00:30 -c copy part1.mp4 -ss 00:00:30 -codec copy part2.mp4 在這裡 
-t 00:00:30 表示從視訊的開始到視訊的第 30 秒建立一部分視訊 
-ss 00:00:30 為視訊的下一部分顯示開始時間戳 它意味著第 2 部分將從第 30 秒開始 並將持續到原始視訊檔案的結尾 
16、接合或合併多個視訊部分到一個 FFmpeg 也可以接合多個視訊部分 並建立一個單個視訊檔案 建立包含你想接合檔案的準確的路徑的 join.txt 所有的檔案都應該是相同的格式（相同的編碼格式） 所有檔案的路徑應該逐個列出 像下面 
file /home/sk/myvideos/part1.mp4
file /home/sk/myvideos/part2.mp4
file /home/sk/myvideos/part3.mp4
file /home/sk/myvideos/part4.mp4
現在 接合所有檔案 使用命令：
$ ffmpeg -f concat -i join.txt -c copy output.mp4
如果你得到一些像下面的錯誤；
[concat @ 0x555fed174cc0] Unsafe file name '/path/to/mp4'
join.txt: Operation not permitted
新增 -safe 0 :
$ ffmpeg -f concat -safe 0 -i join.txt -c copy output.mp4
上面的命令將接合 part1.mp4、part2.mp4、part3.mp4 和 part4.mp4 檔案到一個稱為 output.mp4 的單個檔案中 
17、新增字幕到一個視訊檔案
我們可以使用 FFmpeg 來新增字幕到視訊檔案 為你的視訊下載正確的字幕 並如下所示新增它到你的視訊 
$ fmpeg -i input.mp4 -i subtitle.srt -map 0 -map 1 -c copy -c:v libx264 -crf 23 -preset veryfast output.mp4
18、預覽或測試視訊或音訊檔案
你可能希望通過預覽來驗證或測試輸出的檔案是否已經被恰當地轉碼編碼 為完成預覽 你可以從你的終端播放它 用命令：
$ ffplay video.mp4
類似地 你可以測試音訊檔案 像下面所示 
$ ffplay audio.mp3
19、增加/減少視訊播放速度
FFmpeg 允許你調整視訊播放速度 
為增加視訊播放速度 執行：
$ ffmpeg -i input.mp4 -vf "setpts=0.5*PTS" output.mp4
該命令將雙倍視訊的速度 
為降低你的視訊速度 你需要使用一個大於 1 的倍數 為減少播放速度 執行：
$ ffmpeg -i input.mp4 -vf "setpts=4.0*PTS" output.mp4
20、建立動畫的 GIF 出於各種目的 我們在幾乎所有的社交和專業網路上使用 GIF 影象 使用 FFmpeg 我們可以簡單地和快速地建立動畫的視訊檔案 下面的指南闡釋瞭如何在類 Unix 系統中使用 FFmpeg 和 ImageMagick 建立一個動畫的 GIF 檔案 在 Linux 中如何建立動畫的 GIF
21、從 PDF 檔案中建立視訊 從 PDF 檔案中建立一個視訊 在一個大螢幕裝置（像一臺電視機或一臺電腦）中觀看它們 從一批 PDF 檔案中製作一個電影
在 Linux 中如何從 PDF 檔案中建立一個視訊
ffmpeg -ss 00:00:00  -t 152.14 -i allMerge.ts -i a.mp3  -map 0:v:0 -map 1:a:0  -y -shortest 培德.mp4
