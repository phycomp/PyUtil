from os import walk as osWalk
def rcrsvDIR(DIR='.'):
  for 根, 目錄, 檔案 in osWalk(DIR):
    if not 目錄:
      yield 根, 檔案
    else:
      rcrsvDIR(DIR=目錄)
