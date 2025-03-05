from os.path import splitext, basename
from glob import glob
from csv import reader as csvReader
from pandas import ExcelWriter, read_csv, read_excel, concat
#from openpyxl import Workbook # from https://pythonhosted.org/openpyxl/ or PyPI (e.g. via pip)
from stUtil import rndrCode

def singleCSV(cat, CSV):
  csvDF=[]
  for csvfile in CSV:
    try:
      csvdf = read_csv(csvfile)
      csvDF.append(csvdf)
    except:
      rndrCode(['csvdf', csvfile])
  cmbndDF=concat(csvDF, ignore_index=True)
  cmbndDF.to_csv(f'single{cat}.csv', index=False)
  return cmbndDF
  #wb.save('singleCSV.csv')
def csv2XLS(CSV):
  wb = Workbook()
  for csvfile in CSV:
    ws = wb.active
    with open(csvfile) as fin:    #, 'rb'
      reader = csvReader(fin)
      for r, row in enumerate(reader, start=1):
        for c, val in enumerate(row, start=1):
          ws.cell(row=r, column=c).value = val
  wb.save('allMerge.xlsx')

def df2XLS(cat, CSV):
  xclWriter = ExcelWriter(f'csv2XLS/{cat}Merge.xlsx', engine='xlsxwriter')
  for csvfname in CSV:
    try:
      csvDF = read_csv(csvfname)
      csvSheet = splitext(csvfname)[0].split('/')[-1]  #basename()
      shtPos=csvSheet.find('-')+1
      csvSheet=csvSheet[shtPos:].replace('-', '')
      #rndrCode(['csvSheet', csvfname, csvSheet])
      csvDF.to_excel(xclWriter, sheet_name=csvSheet[:31], index=False)
    except: rndrCode(['csvfname read_csv', csvfname])
  xclWriter.close()

def xls2XLS(cat, XLS):
  xclWriter = ExcelWriter(f'xls2XLS/{cat}Merge.xlsx', engine='xlsxwriter')
  for csvfname in XLS:
    try:
      csvDF = read_excel(csvfname)
      csvSheet = splitext(csvfname)[0].split('/')[-1]  #basename()
      shtPos=csvSheet.find('-')+1
      csvSheet=csvSheet[shtPos:].replace('-', '')
      #rndrCode(['csvSheet', csvfname, csvSheet])
      csvDF.to_excel(xclWriter, sheet_name=csvSheet[:31], index=False)   #
    except: rndrCode(['csvfname read_csv', csvfname])
  xclWriter.close()
