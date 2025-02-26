from stUtil import rndrCode
from __future__ import print_function
from docutils import core
from docutils.writers.html4css1 import Writer, HTMLTranslator
import sys, os

class HTMLFragmentTranslator(HTMLTranslator):
    def __init__( self, document ):
        HTMLTranslator.__init__( self, document )
        self.head_prefix = ['','','','','']
        self.body_prefix = []
        self.body_suffix = []
        self.stylesheet = []
    def astext(self):
        return ''.join(self.body)

html_fragment_writer = Writer()
html_fragment_writer.translator_class = HTMLFragmentTranslator

def reST_to_html( s ):
    return core.publish_string(s, writer=html_fragment_writer)

if __name__ == '__main__':
    if len(sys.argv)>1:
        if sys.argv[1] != "":
            rstfile = open(sys.argv[1])
            text = rstfile.read()
            rstfile.close()
            if len(sys.argv)>2:
                if sys.argv[2] != "":
                    htmlfile = sys.argv[2]
            else:
                htmlfile = os.path.splitext(os.path.basename(sys.argv[1]))[0]+".html"
            result = reST_to_html(text)
            print(result)
            output = open(htmlfile, "wb")
            output.write(result)
            output.close()  
    else:
        rndrCode("Usage:\ncompile_rst.py docname.rst\nwhich results in => docname.html\ncompile_rst.py docname.rst desiredname.html\nwhich results in => desiredname.html")
