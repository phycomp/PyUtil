import sys
import getopt
from Core.Interfaces.IReader import IReader
from pdfminer.pdfparser import PDFDocument, PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter, process_pdf
from pdfminer.pdfdevice import PDFDevice, TagExtractor
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.cmapdb import CMapDB
from pdfminer.layout import LAParams
from cStringIO import StringIO

class PdfReader(object):
    def __init__(self): pass
    def readText(self,path, outtype='text', opts={}):
        outfile = path[:-3] + outtype
        outdir = '/'.join(path.split('/')[:-1])
        # debug option
        debug = 0
        # input option
        password = ''
        pagenos = set()
        maxpages = 0
        # output option
        # ?outfile = None
        # ?outtype = None
        outdir = None
        #layoutmode = 'normal'
        codec = 'utf-8'
        pageno = 1
        scale = 1
        showpageno = True
        laparams = LAParams()
        for (k, v) in opts:
            if k == '-d': debug += 1
            elif k == '-p': pagenos.update( int(x)-1 for x in v.split(',') )
            elif k == '-m': maxpages = int(v)
            elif k == '-P': password = v
            elif k == '-o': outfile = v
            elif k == '-n': laparams = None
            elif k == '-A': laparams.all_texts = True
            elif k == '-V': laparams.detect_vertical = True
            elif k == '-M': laparams.char_margin = float(v)
            elif k == '-L': laparams.line_margin = float(v)
            elif k == '-W': laparams.word_margin = float(v)
            elif k == '-F': laparams.boxes_flow = float(v)
            elif k == '-Y': layoutmode = v
            elif k == '-O': outdir = v
            elif k == '-t': outtype = v
            elif k == '-c': codec = v
            elif k == '-s': scale = float(v)
    
        print laparams
        #
        #PDFDocument.debug = debug
        #PDFParser.debug = debug
        CMapDB.debug = debug
        PDFResourceManager.debug = debug
        PDFPageInterpreter.debug = debug
        PDFDevice.debug = debug
        #
        rsrcmgr = PDFResourceManager()
    
        #outtype = 'text'
    
        outfp = StringIO()
    
        device = HTMLConverter(rsrcmgr, outfp, codec=codec, laparams=laparams)
    
    
        fp = file(path, 'rb')
        process_pdf(rsrcmgr, device, fp, pagenos, maxpages=maxpages, password=password,
                        check_extractable=True)
        fp.close()
        device.close()
        print outfp.getvalue()
        outfp.close()
    
        return



reader = PdfReader()
opt = map(None,['-W','-L','-t'],[0.5,0.4,'html'])
reader.readText("/test_data/test.pdf","html",opt)
