from streamlit import subheader, sidebar, session_state, radio as stRadio, columns as stCLMN, text_area, text_input, multiselect, toggle as stToggle, tabs as stTAB, markdown as stMarkdown, write as stWrite, code as stCode #slider, markdown, dataframe, code as stCode, cache as stCache,
from stUtil import rndrCode
from streamlit import image as stImage
from streamlit import columns as stCLMN, container as stContainer
#from rndrCode import rndrCode
#from os import walk as osWalk
#from os.path import basename
from os.path import dirname, basename
from 遞廻 import rcrsvDIR
from pandas import DataFrame
from mimetypes import guess_type#, read_mime_types    mimetypes.(path_file_to_upload)[
#from streamlit_pdf_viewer import pdf_viewer
from fhndlrUtil import 四欄文件, 三欄文件, pdfVWR, profileDIR, clmn4PIC
from pathlib import Path, PurePath
from streamlit import video as stVideo

MENU, 表單, brwsrType=[], ['遍歷目錄', '檔案管理'], ['PDF', 'PIC', 'VID', 'MD'] #, '卦爻辭', '錯綜複雜', '二十四節氣'
for ndx, Menu in enumerate(表單): MENU.append(f'{ndx}{Menu}')
with sidebar:
  menu=stRadio('表單', MENU, horizontal=True, index=0)

tabPDF, tabPIC, tabVID, tabMD = stTAB(brwsrType)

if menu==len(表單):
  pass
elif menu==MENU[1]:
  #tblName='sutra'
  #sutraCLMN=queryCLMN(tblSchm='public', tblName=tblName, db='sutra')
  ##rndrCode(sutraCLMN)
  #fullQuery=f'''select {','.join(sutraCLMN)} from {tblName} where 章節~'中庸';'''
  #rsltQuery=runQuery(fullQuery, db='sutra')
  #rndrCode([fullQuery, rsltQuery])
  #rsltDF=session_state['rsltDF']=DataFrame(rsltQuery, index=None, columns=sutraCLMN)
  #rsltDF#[rsltDF['章節']=='中庸']
  根, 檔案=session_state['根檔案']
  rndrCode([根, 檔案])
    #if count%3==2: stImage(fullName, caption=fullName)
    #[stImage(f'images/{fname}', caption=f'images/{fname}') for ndx, fname in enumerate(檔案) if ndx%3==2]
elif menu==MENU[0]:
  page, PRUNEPATHS=10, ['pages', '__pycache__']
  prntDIR=['/media/Archive', '/media/vimUsage', '/media/道場']
  with sidebar:
    母=stRadio('表單', prntDIR, horizontal=True, index=0, key='母')
  if 母:
    母層=Path(母) #/archive
    根層=母層.iterdir()  #map(lambda x:母層/x, ['soulSong', '善歌中庸', '簡體善歌', '佛典歌曲', 'ICD10', 'nlpArchve', 'PDFs', 'PICs', '台北區', 'archive', '英文善歌'])    # #archive   PICs  台北區
  #rndrCode(list(母層.iterdir()))
  #rndrCode(Path.__dict__.keys())    #'stat', 'lstat', 'exists', 'is_dir', 'is_file', 'is_mount', 'is_symlink', 'is_junction', 'is_block_device', 'is_char_device', 'is_fifo', 'is_socket', 'samefile', 'open', 'read_bytes', 'read_text', 'write_bytes', 'write_text', 'iterdir', '_scandir', '_make_child_relpath', 'glob', 'rglob', 'walk', '__init__', '__new__', '__enter__', '__exit__', 'cwd', 'home', 'absolute', 'resolve', 'owner', 'group', 'readlink', 'touch', 'mkdir', 'chmod', 'lchmod', 'unlink', 'rmdir', 'rename', 'replace', 'symlink_to', 'hardlink_to', 'expanduser'
  #根層=listdir(母層)
  #rndrCode(listdir(母層))
  #根層=filter(Path.is_dir, map(lambda x:f'{母層}/{x}', listdir(母層))) #[isdir(d) for dir in DIR ]
  #根層=filter(basename, list(根層))
  #rndrCode(['根層', list(根層)])
  #DIR=['PICs', 'images', 'PDFs']
  with sidebar:
    目錄=map(lambda x:PurePath(x).name, 根層)
    #rndrCode(['目錄', list(目錄)])
    目錄=stRadio('目錄', 目錄, horizontal=True, index=0)
  if 目錄:
    目錄=母層/目錄  #f'{母層}/{目錄}'
    for 根, 檔案  in rcrsvDIR(DIR=目錄):
      根=Path(根)
      with sidebar:
        dirPRFL=profileDIR(根, 檔案)
        def parsePRFL(d):
          tmp={}
          for k, v in dirPRFL.items(): tmp.update({k:len(v)})
          return tmp
            #{k:len(v)} 
        rndrCode([根, parsePRFL(dirPRFL)])
        #瀏覽=stRadio('瀏覽', brwsrType, key='browser', horizontal=True, index=0)
    with sidebar:
      分頁=stRadio('分頁', range(page), horizontal=True, index=0, key='分頁')   #
      #rndrCode([根, 檔案])
      pageSize=len(檔案)//page  #len(檔案)//(page*3) if 瀏覽=='PIC' else 
      pageSize+=1
      pgStmp=f'{根}|{page}|{pageSize}'
  with tabMD:
    mdKEY=[]
    for k in dirPRFL:
      #if k and (k.find('video')!=-1 or k.find('audio')!=-1): imgKEY.append(k)
      if k and ('text' in k or 'sh' in k or 'csv' in k):mdKEY.append(k)
    分頁檔案=[]
    for k in mdKEY:
      #dirPRFL['image/png']
      分頁檔案+=dirPRFL[k]
    #if 分頁檔案:=dirPRFL.get('text/markdown'):
    with sidebar:
      neatPDF=map(basename, 分頁檔案)
      #根=Path(dirname(分頁檔案[0]))
      MD=stRadio('', neatPDF, horizontal=True, index=None, key='tabMD')
    if MD:
      with open(根/MD) as fin:
        stWrite(fin.read())
  with tabPDF:
    pdfMODE=stRadio('pdfMODE', ['單PDF', '圖片'], horizontal=True, index=None, key='pdfMODE')
    if 分頁檔案:=dirPRFL.get('application/pdf'):
      with sidebar:
        neatPDF=map(basename, 分頁檔案)
        根=Path(dirname(分頁檔案[0]))
        PDF=stRadio('', neatPDF, horizontal=True, index=None, key='tabPDF')
      if PDF and pdfMODE: pdfVWR(根, PDF, pdfMODE=pdfMODE)

  with tabPIC:
    imgKEY=[]
    for k in dirPRFL:
      if k and 'image' in k: imgKEY.append(k)
    for k in imgKEY:
      #dirPRFL['image/png']
      分頁檔案=dirPRFL[k]
      檔案段=[分頁檔案[ndx*pageSize:(ndx+1)*pageSize-1] for ndx in range(page)]
      #rndrCode(分頁檔案)
      if str(分頁):
        if len(檔案段)==1: 分頁檔案=檔案段
        else: 分頁檔案=檔案段[分頁]#檔案[分頁*pageSize:(分頁+1)*pageSize-1]
      clmn4PIC(分頁檔案)

  with tabVID:
    vidKEY=[]
    for k in dirPRFL:
      #if k and (k.find('video')!=-1 or k.find('audio')!=-1): imgKEY.append(k)
      if k and ('audio' in k or 'video' in k):vidKEY.append(k)
    for k in vidKEY:
      #dirPRFL['image/png']
      分頁檔案=dirPRFL[k]
      with sidebar:
        neatVID=map(basename, 分頁檔案)
        根=Path(dirname(分頁檔案[0]))#.stem    #dirname()
        VID=stRadio('', neatVID, horizontal=True, index=None, key=f'{k}')
        #VID=stRadio('', 分頁檔案, horizontal=True, index=None)
      if VID:   #pathlib物件
      #for VID in neatVID:
        subheader(basename(VID))
        fullVid=根/VID
        stVideo(fullVid.as_posix(), autoplay=True) #st.video(data, format="video/mp4", start_time=0, *, subtitles=None, end_time=None, loop=False, autoplay=False, muted=False)
