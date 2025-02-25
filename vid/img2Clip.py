#!/usr/bin/env python
loopArg, scaleArg, cnctArg, xfadeArg='ffmpeg ', '-filter_complex "', '', '"'
from random import choice as rndmChc, choices as rndmChoices
from glob import glob
FILEs=glob('*.jpg')
fLen, trnsDur, vidDur, offset=len(FILEs), 1, 5, 0
EFFECTs=["fade", "wipeleft", "wiperight", "wipeup", "wipedown", "slideleft", "slideright", "slideup", "slidedown", "circlecrop", "rectcrop", "distance", "fadeblack", "fadewhite", "radial", "smoothleft", "smoothright", "smoothup", "smoothdown", "circleopen", "circleclose", "vertopen", "vertclose", "horzopen", "horzclose", "dissolve", "pixelize", "diagtl", "diagtr", "diagbl", "diagbr", "hlslice", "hrslice", "vuslice", "vdslice", "hblur", "fadegrays", "wipetl", "wipetr", "wipebl", "wipebr", "squeezeh", "squeezev", "zoomin"]

XFADEs=['custom', 'fade', 'wipeleft', 'wiperight', 'wipeup', 'wipedown', 'slideleft', 'slideright', 'slideup', 'slidedown', 'circlecrop', 'rectcrop', 'distance', 'fadeblack', 'fadewhite', 'radial', 'smoothleft', 'smoothright', 'smoothup', 'smoothdown', 'circleopen', 'circleclose', 'vertopen', 'vertclose', 'horzopen', 'horzclose', 'dissolve', 'pixelize', 'diagtl', 'diagtr', 'diagbl', 'diagbr', 'hlslice', 'hrslice', 'vuslice', 'vdslice', 'hblur', 'fadegrays', 'wipetl', 'wipetr', 'wipebl', 'wipebr', 'squeezeh', 'squeezev', 'zoomin', 'fadefast', 'fadeslow', 'hlwind', 'hrwind', 'vuwind', 'vdwind', 'coverleft', 'coverright', 'coverup', 'coverdown', 'revealleft', 'revealright', 'revealup', 'revealdown']
xFADE=rndmChoices(XFADEs, k=fLen)
for idx, file in enumerate(FILEs):
    loopArg+=f'-loop 1 -t {vidDur} -i {file} '
    #ffct=rndmChce(EFFECTs)
    #xffct=rndmChce(XFADEs)
    xfd=xFADE[idx]
    offset+=vidDur-trnsDur
    if not idx:
        #fdArg='fade=t=out:st=4:d=1' #effect='out'
        #xfadeArg+=f"[0][1:v]xfade=transition={xfd}:duration={trnsDur}:offset={offset}[vfade1];"
        #xfadeArg+=f"[0p][1p]xfade=duration={trnsDur}:offset={offset}[1x];"
        xfadeArg+=f"""[0][1]xfade=transition={xfd}:duration={trnsDur}:offset={offset}[f1];"""
    #elif idx==fLen-2:
    #   #xfadeArg+=f"[vfade{idx}][{idx+1}:v]xfade=transition={xfd}:duration={trnsDur}:offset={offset},format=yuv420p; "
    #   xfadeArg+=f"[{idx}x][{idx+1}p]xfade=duration=1:offset={offset}[{idx}x]"
    #elif idx==fLen-1:
    #    xfadeArg+=f'''[f{idx}]format=yuv420p[video]" -map "[video]"'''
    else:
        #fdArg='fade=t=in:st=0:d=1,fade=t=out:st=4:d=1' #effect='in'
        #xfadeArg+=f"[vfade{idx}][{idx+1}:v]xfade=transition={xfd}:duration={trnsDur}:offset={offset}[vfade{idx+1}]; "
        #xfadeArg+=f"[{idx}x][{idx+1}p]xfade=duration={trnsDur}:offset={offset}[{idx+1}x];"
        xfadeArg+=f'''[f{idx-1}][{idx}]xfade=transition={xfd}:duration={trnsDur}:offset={offset}[f{idx}];'''
    #scaleArg+=f"[{idx}:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,{fdArg}[v{idx}]; " #fade=t={effect}:st=4:d=1
    #cnctArg+=f'[v{idx}]'
#extraARG=f"""-map "[f{idx}]" -r 25 -pix_fmt yuv420p -vcodec libx264 """
#extraARG=f"-map [{idx}x] -c:v libx264 -crf 17"  # -y OUTPUT.mp4
#extraARG='''-map "[video]"'''
#concatArg=f'{cnctArg}concat=n={len(FILEs)}:v=1:a=0,format=yuv420p[v]"'
#finalArg=f'{loopArg}{scaleArg}{concatArg} -map "[v]" -y out.mp4'
#finalArg=f'{loopArg} "{xfadeArg}" -y out.mp4'  #{extraARG}
xfadeArg+='"'
finalArg=f"""-map "[f{idx}]" -r 25 -pix_fmt yuv420p -vcodec libx264 -y output.mp4"""
cmd=f"{loopArg} {xfadeArg} {finalArg}"
print(cmd)


"""
ffmpeg -y -loop 1 -t 10 -i "img01.jpg" -loop 1 -t 11 -i "img02.jpg" -loop 1 -t 5 -i "img03.jpg" -filter_complex 
[0][1]xfade=transition=diagbr:duration=3:offset=7[f1];
[f1][2]xfade=transition=diagbr:duration=1:offset=20[f2];
[f2]format=yuv420p[video]" -map "[video]" 

ffmpeg -loop 1 -t 5 -i test4.jpg -i test4.jpg -preset ultrafast -vsync vfr -filter_complex "[0:v]fade=t=out:st=4:d=1,scale=w=1280:h=720:force_original_aspect_ratio=1,pad=1280:720:(ow-iw)/2:(oh-ih)/2[v0];[1:v]scale=w=1280:h=720:force_original_aspect_ratio=1,pad=1280:720:(ow-iw)/2:(oh-ih)/2[v1];[v0][v1]concat=n=2:v=1:a=0,format=yuv420p[v]" -map "[v]" rd.mp4

[2x][3p]xfade=duration=1:offset=12[3x]
" -map [3x] -c:v libx264 -crf 17 /tmp/output.mp4
"""

'''
count=0
"[0][1:v]xfade=transition=fade:duration=1:offset=3[vfade1]; \
 [vfade1][2:v]xfade=transition=fade:duration=1:offset=10[vfade2]; \
 [vfade2][3:v]xfade=transition=fade:duration=1:offset=21[vfade3]; \
 [vfade3][4:v]xfade=transition=fade:duration=1:offset=25,format=yuv420p; \
v0.mp4	4	+	0	-	1	3
v1.mp4	8	+	3	-	1	10
v2.mp4	12	+	10	-	1	21
v3.mp4	5	+	21	-	1	25
xfade=transition=dissolve:duration=3:offset=3
ffmpeg \
-loop 1 -t 5 -i 1.jpg \
-loop 1 -t 5 -i 2.jpg \
-loop 1 -t 5 -i 3.jpg \
-loop 1 -t 5 -i 4.jpg \
-loop 1 -t 5 -i 5.jpg \
-filter_complex \
"[0:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fade=t=out:st=4:d=1[v0]; \
 [1:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fade=t=in:st=0:d=1,fade=t=out:st=4:d=1[v1]; \
 [2:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fade=t=in:st=0:d=1,fade=t=out:st=4:d=1[v2]; \
 [3:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fade=t=in:st=0:d=1,fade=t=out:st=4:d=1[v3]; \
 [4:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fade=t=in:st=0:d=1,fade=t=out:st=4:d=1[v4]; \
 [v0][v1][v2][v3][v4]concat=n=5:v=1:a=0,format=yuv420p[v]" -map "[v]" out.mp4
'''
