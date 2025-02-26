from os import walk as osWalk
from mimetypes import MimeTypes
from mimetypes import guess_type, add_type#, guess_extension#, read as mimeREAD (filename, strict=True)
from streamlit import columns as stCLMN, image as stImage, markdown as stMarkdown
#from rndrCode import rndrCode
from base64 import b64encode
from pathlib import Path
import mimetypes

if not mimetypes.inited: mimetypes.init()
add_type('application/vnd.oasis.opendocument.presentation', '.odp')
add_type('text/markdown', '.md')

def profileDIR(根, 檔案):
  dirPRFL={}
  for fname in 檔案:
    fullName=根/fname
    #ext=fullName.suffix
    #mt=MimeTypes()  #fullName.as_posix()
    #mType=mt.read(fullName.as_posix(), strict=True)
    #mType=mimeREAD(fullName, strict=True)[0]    #
    mType=guess_type(fullName)[0]#guess_extension, strict=True
    if not dirPRFL.get(mType):
      dirPRFL[mType]=[fullName]
    else:
      tmp=dirPRFL[mType]
      tmp.append(fullName)
      dirPRFL[mType]=tmp
  return dirPRFL

def rcrsvDIR(DIR='.'):
  pthDIR=Path(DIR)
  for 根, 目錄, 檔案 in pthDIR.walk():
    if not 目錄:
      yield 根, 檔案
    else:
      rcrsvDIR(DIR=目錄)

def pdfVWR(根, 檔案):
  #newPage=0
  #for fname in 檔案:
  fullPDF=根/檔案
  #pdf_viewer(fullName, key=fullName)
  with open(fullPDF, "rb") as fin:
    base64PDF = b64encode(fin.read()).decode('utf-8')
  pdfPAGE = f'<iframe src="data:application/pdf;base64,{base64PDF}" width=800 height=800 type="application/pdf"></iframe>'
  stMarkdown(pdfPAGE, unsafe_allow_html=True)

def fhndler(根, 檔案):
  for fname in 檔案:
    fullName=f'{根}/{fname}'
    mimeType=guess_type(fullName)[0]
    if mimeType.split('/')[0]=='image':
      stImage(fullName, caption=fullName)
    elif mimeType=='application/pdf':
      pdf_viewer(fullName, key=fullName)

def 四欄文件(根, 檔案):
  #rndrCode(檔案)
  leftPane, midLeft, midRight, rightPane=stCLMN(4)   #Container(3)
  with leftPane:
    for ndx, fname in enumerate(檔案):
      fullName=f'{根}/{fname}'
      #rndrCode(fullName)    #[根, 檔案]
      mimeType=guess_type(fullName)[0]
      mimeType=mimeType
      if mimeType.split('/')[0]=='image': #/png
        if not ndx%4:
          stImage(fullName, caption=fullName, use_column_width=True) #f'images/{fname}'
      #elif mimeType=='application/pdf':
      #  pdf_viewer(fullName, key=fullName)
  with midLeft:
    for ndx, fname in enumerate(檔案):
      fullName=f'{根}/{fname}'
      mimeType=guess_type(fullName)[0]
      mimeType=mimeType.split('/')[0]
      if mimeType.split('/')[0]=='image': #/png
        if ndx%4==1:
          stImage(fullName, caption=fullName, use_column_width=True)  #f'images/{fname}'
    #[stImage(f'images/{fname}', caption=f'images/{fname}') for ndx, fname in enumerate(檔案) if ndx%3==1]
  with midRight:
    for ndx, fname in enumerate(檔案):
      fullName=f'{根}/{fname}'
      mimeType=guess_type(fullName)[0]
      mimeType=mimeType.split('/')[0]
      if mimeType.split('/')[0]=='image': #/png
        if ndx%4==2:
          stImage(fullName, caption=fullName, use_column_width=True)  #f'images/{fname}'
    #[stImage(f'images/{fname}', caption=f'images/{fname}') for ndx, fname in enumerate(檔案) if ndx%3==1]
  with rightPane:
    for ndx, fname in enumerate(檔案):
      fullName=f'{根}/{fname}'
      mimeType=guess_type(fullName)[0]
      mimeType=mimeType.split('/')[0]
      if mimeType=='image':
        if ndx%4==3:
          stImage(fullName, caption=fullName, use_column_width=True)

def 三欄文件(根, 檔案): #dsply文件
  leftPane, midPane, rightPane=stCLMN(3)   #Container(3)
  with leftPane:
    for ndx, fname in enumerate(檔案):
      fullName=f'{根}/{fname}'
      #rndrCode(fullName)    #[根, 檔案]
      mimeType=guess_type(fullName)[0]
      mimeType=mimeType
      if mimeType.split('/')[0]=='image': #/png
        if not ndx%3:
          stImage(fullName, caption=fullName, use_column_width=True) #f'images/{fname}'
      #elif mimeType=='application/pdf':
      #  pdf_viewer(fullName, key=fullName)
  with midPane:
    for ndx, fname in enumerate(檔案):
      fullName=f'{根}/{fname}'
      mimeType=guess_type(fullName)[0]
      mimeType=mimeType.split('/')[0]
      if mimeType.split('/')[0]=='image': #/png
        if ndx%3==1:
          stImage(fullName, caption=fullName, use_column_width=True)  #f'images/{fname}'
    #[stImage(f'images/{fname}', caption=f'images/{fname}') for ndx, fname in enumerate(檔案) if ndx%3==1]
  with rightPane:
    for ndx, fname in enumerate(檔案):
      fullName=f'{根}/{fname}'
      mimeType=guess_type(fullName)[0]
      mimeType=mimeType.split('/')[0]
      if mimeType=='image':
        if ndx%3==2:
          stImage(fullName, caption=fullName, use_column_width=True)
