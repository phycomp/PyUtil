from olefile import OleFileIO
from re import findall, compile

def asciiABLE(inputStrm):
    asciiREGEX = compile('[^\x20-\x7E]')
    cleaned = asciiREGEX.sub('', inputStrm)
    return cleaned

def parsePPT(fname):
  with OleFileIO(fname) as ole:
    byteStrm = ole.openstream("PowerPoint Document").read()

  text_data = byteStrm.decode('utf-8', errors='replace')
  all_text = findall('\x00\x00[a-zA-Z0-9].*?\x00\x00', text_data)
  all_text = [x.replace('\x00\x00', '') for x in all_text if x != '\x00\x00\x00\x00']
  allXtrcted = [x for x in all_text if len(x) <= len(asciiABLE(x))]
  return ''.join(allXtrcted)
