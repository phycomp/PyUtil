#!/usr/bin/env python
from sys import argv
from os import system

def restoreSys():
  cmd='cp /home/josh/.gcin/XiangJingSheng.cin.ori /home/josh/.gcin/XiangJingSheng.cin'
  print(cmd)
  system(cmd)

fname, Ori=argv[1], '%s.ori'%argv[1]
cmd='cp %s %s'%(fname, Ori)
print(cmd)
system(cmd)
fin=open(Ori, 'rb')
data=fin.read()
#Data=data.encode('utf-8')
begin, end, carriage=b'%chardef begin\n', b'%chardef end\n', b'\n'
pos=data.find(begin)
pos+=len(begin)
end_pos=fin.seek(-len(end), 2)
fin.seek(pos)
with open('%s'%fname, 'wb') as fout:
  fout.write(data[:pos])
  Temp, temp={}, []
  for line in fin:
    line=line.split()
    if not line:
      temp=sorted(temp)
      if fin.tell()==end_pos:
        for Chn in temp:
          #Chn=Chn.decode('utf-8')
          Eng, len_chn=b'', len(Chn)
          if len_chn>12: Eng=Temp[Chn]#.encode('utf-8')]
          else:
            #print(len_chn, Chn.decode('utf-8'))
            for chn in Chn.decode('utf-8'):
              try: eng=Temp[chn.encode('utf-8')]
              except:
                restoreSys()
                chnChar=chn.encode('utf-8')
                print(f"chnChar={chnChar.decode('utf8')}")
              #if chn=='臸':print('臸', eng)
              #eng=eng.decode('utf-8')
              #print(eng, chn, Chn.decode('utf-8'))
              #print(eng, chn)
              if len_chn>6: Eng+=eng.decode('utf-8')[0].encode('utf-8')#eng[0].decode('utf-8')
              else: Eng+=eng[:2]#.decode('utf-8')
            #print(Eng)
          space=b' '*(5-len(Eng))
          phrase=Eng+space+Chn+carriage
          #print(phrase.decode('utf-8'))
          fout.write(phrase)
      else:
          for chn in temp:
            eng=Temp[chn]
            space=b' '*(5-len(eng))
            phrase=eng+space+chn+carriage
            fout.write(phrase)
      fout.write(carriage)
      temp=[]
    else:
        #print('line=', line)
        try:
          eng, chn=line
          if not temp.count(chn):
             temp.append(chn)
             Temp[chn]=eng
          #temp.add(chn)
        except:
          print('line=', line)
        #print(eng, chn)
  fout.write(end)
  #fout.close()
  cmd='gcin2tab %s'%fname
  print(cmd)
  system(cmd)
