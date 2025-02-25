from streamlit import info as stInfo, code as stCode, sidebar, text_input, write as stWrite, session_state
('病歷號32567127')

MENUs=['basicUsage', 'miscVid', '檢驗類資料', '病歷文本']  #, 'Annot', 'nerTagger', 'embedding', 'BILUO', 'viterbi', 'Metadata',
menu = sidebar.radio('Output', MENUs, index=0)
if menu==MENUs[0]:
    xfadeUsage='''ffmpeg -i v0.mp4 -i v1.mp4 -i v2.mp4 -i v3.mp4 -i v4.mp4 -filter_complex \
    "[0][1:v]xfade=transition=fade:duration=1:offset=3[vfade1]; \
     [vfade1][2:v]xfade=transition=fade:duration=1:offset=10[vfade2]; \
     [vfade1][2:v]xfade=transition=fade:duration=1:offset=13[vfade2];

     [2:v]xfade=transition=fade:duration=1:offset=13[vfade2];[vfade2]
     [vfade2][3:v]xfade=transition=fade:duration=1:offset=21[vfade3]; \
     [vfade3][4:v]xfade=transition=fade:duration=1:offset=25,format=yuv420p; \
     [0:a][1:a]acrossfade=d=1[afade1]; \
     [afade1][2:a]acrossfade=d=1[afade2]; \
     [afade2][3:a]acrossfade=d=1[afade3]; \
     [afade3][4:a]acrossfade=d=1" \
    -movflags +faststart out.mp4

    '''
    session_state['miscUsage']=xfadeUsage
    stWrite([xfadeUsage])
elif menu==MENUs[1]:
    miscUsage=session_state['miscUsage']    #=xfadeUsage
    stWrite([miscUsage])
    #stWrite([miscUsage]) #session_state['miscUsage']=xfadeUsage
    #ffprobe vid.mp4 2>&1|grep Duration|awk -F, '{print $1}'|awk '{print $2}'
    noVID=text_input('vid')
    vidDur=4
    if noVID:
        noVID=int(noVID)
        visClip=''
        audClip=''
        nxtDur=0
        avClip=''
        for itr in range(noVID):
            nxtItr=itr+1
            if itr==noVID-1: nxtDur+=vidDur-1
            else: nxtDur+=nxtItr*vidDur-1
            if not itr:
                xfadeSyntax=f"[{itr}][{nxtItr}:v]xfade=transition=fade:duration=1:offset={nxtDur}[vfade{nxtItr}];"
                afadeSyntax=f"[{itr}:a][{nxtItr}:a]acrossfade=d=1[afade{nxtItr}];"
            elif itr==noVID-1:
                xfadeSyntax=f"[vfade{itr}][{nxtItr}:v]xfade=transition=fade:duration=1:offset={nxtDur},format=yuv420p;"
                afadeSyntax=f"[afade{itr}][{nxtItr}:a]acrossfade=d=1 "
            else:
                xfadeSyntax=f"[vfade{itr}][{nxtItr}:v]xfade=transition=fade:duration=1:offset={nxtDur}[vfade{nxtItr}];"
                afadeSyntax=f"[afade{itr}][{nxtItr}:a]acrossfade=d=1[afade{nxtItr}];"
            visClip+=xfadeSyntax
            audClip+=afadeSyntax
        avClip+=visClip+audClip+'-movflags +faststart out.mp4'
        #stWrite([visClip])
        stWrite([avClip])
elif menu==MENUs[2]:pass
elif menu==MENUs[3]:pass
