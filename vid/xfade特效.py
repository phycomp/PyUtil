ffmpeg -loop 1 -t 5 -i 1.png -loop 1 -t 5 -i 2.png -filter_complex "[0][1]xfade=transition=fade:duration=1:offset=4,format=yuv420p" output.mp4
ffmpeg -loop 1 -t 5 -i 1.png -loop 1 -t 5 -i 2.png -filter_complex "[0][1]xfade=transition=fade:duration=1:offset=4,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 output.gif
Gallery Below is gallery of the available effects. fade is the default transition. Names in bold are also available in xfade_opencl.

fade (default)	fadeblack	fadewhite	distance
wipeleft	wiperight	wipeup	wipedown
slideleft	slideright	slideup	slidedown
smoothleft	smoothright	smoothup	smoothdown
rectcrop	circlecrop	circleclose	circleopen
horzclose	horzopen	vertclose	vertopen
diagbl	diagbr	diagtl	diagtr
hlslice	hrslice	vuslice	vdslice
dissolve	pixelize	radial	hblur
wipetl	wipetr	wipebl	wipebr
zoomin transition for xfade
fadegrays	squeezev	squeezeh	zoomin
hlwind	hrwind	vuwind	vdwind
coverleft	coverright	coverup	coverdown
revealleft	revealright	revealup	revealdown

Command used to make each gallery image; as reference for future gallery additions:

ffmpeg -f lavfi -i "color=c=blue:s=180x136:r=15:d=2,format=rgb24,drawtext=text='fadeblack':x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxborderw=4:boxcolor=white:fontfile=/usr/share/fonts/TTF/VeraMono.ttf:fontsize=20" -f lavfi -i "color=c=aqua:s=180x136:r=15:d=2,format=rgb24,drawtext=text='fadeblack':x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxborderw=4:boxcolor=white:fontfile=/usr/share/fonts/TTF/VeraMono.ttf:fontsize=20" -filter_complex "[0][1]xfade=duration=1:offset=1:transition=fadeblack,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" fadeblack.gif

Custom effects You can make your own custom effects using transition=custom and the expr options. See the xfade documentation for more info. xfade_opencl xfade_opencl is the Open CL variant of the xfade filter. This filter supports a subset of the filters available in xfade (see bold names in gallery above) and also supports creation of custom effects. It requires ffmpeg to be configured with --enable-opencl and you must initialize a hardware device in your command. See OpenCL Video Filters for general info.

***************************************
ffmpeg -loop 1 -t 5 -i 1.png -loop 1 -t 5 -i 2.png -filter_complex "[0][1]xfade=transition=fade:duration=1:offset=4,format=yuv420p" output.mp4
ffmpeg -loop 1 -t 5 -i 1.png -loop 1 -t 5 -i 2.png -filter_complex "[0][1]xfade=transition=fade:duration=1:offset=4,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 output.gif
Gallery Below is gallery of the available effects. fade is the default transition. Names in bold are also available in xfade_opencl.

fade (default)	fadeblack	fadewhite	distance
wipeleft	wiperight	wipeup	wipedown
slideleft	slideright	slideup	slidedown
smoothleft	smoothright	smoothup	smoothdown
rectcrop	circlecrop	circleclose	circleopen
horzclose	horzopen	vertclose	vertopen
diagbl	diagbr	diagtl	diagtr
hlslice	hrslice	vuslice	vdslice
dissolve	pixelize	radial	hblur
wipetl	wipetr	wipebl	wipebr
zoomin transition for xfade
fadegrays	squeezev	squeezeh	zoomin
hlwind	hrwind	vuwind	vdwind
coverleft	coverright	coverup	coverdown
revealleft	revealright	revealup	revealdown

Command used to make each gallery image; as reference for future gallery additions:

ffmpeg -f lavfi -i "color=c=blue:s=180x136:r=15:d=2,format=rgb24,drawtext=text='fadeblack':x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxborderw=4:boxcolor=white:fontfile=/usr/share/fonts/TTF/VeraMono.ttf:fontsize=20" -f lavfi -i "color=c=aqua:s=180x136:r=15:d=2,format=rgb24,drawtext=text='fadeblack':x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxborderw=4:boxcolor=white:fontfile=/usr/share/fonts/TTF/VeraMono.ttf:fontsize=20" -filter_complex "[0][1]xfade=duration=1:offset=1:transition=fadeblack,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" fadeblack.gif

Custom effects You can make your own custom effects using transition=custom and the expr options. See the xfade documentation for more info. xfade_opencl xfade_opencl is the Open CL variant of the xfade filter. This filter supports a subset of the filters available in xfade (see bold names in gallery above) and also supports creation of custom effects. It requires ffmpeg to be configured with --enable-opencl and you must initialize a hardware device in your command. See OpenCL Video Filters for general info.
****************************
#ffmpeg -loop 1 -t 5 -i 1.png -loop 1 -t 5 -i 2.png -filter_complex "[0][1]xfade=transition=fade:duration=1:offset=4,format=yuv420p" output.mp4
#ffmpeg -loop 1 -t 5 -i 1.png -loop 1 -t 5 -i 2.png -filter_complex "[0][1]xfade=transition=fade:duration=1:offset=4,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 output.gif
