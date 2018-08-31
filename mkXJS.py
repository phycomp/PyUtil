#!/usr/bin/env python
from sys import argv
from os import system

fname, Ori=argv[1], '%s.ori'%argv[1]
cmd='cp %s %s'%(fname, Ori)
print(cmd)
system(cmd)
fin, fout=open(Ori, 'rb'), open(fname, 'wb')
data=fin.read()
#Data=data.encode('utf-8')
begin, end, carriage=b'%chardef begin\n', b'%chardef end\n', b'\n'
pos=data.find(begin)
pos+=len(begin)
end_pos=fin.seek(-len(end), 2)
fin.seek(pos)
fout.write(data[:pos])
Temp, temp={}, []
for line in fin:
    line=line[:-1].split()
    if not line:
        temp=sorted(temp)
        if fin.tell()==end_pos:
            for Chn in temp:
                Eng, len_chn=b'', len(Chn)
                if len_chn>=5:
                    Eng=Temp[Chn]
                else:
                    for chn in Chn:
                        eng=Temp[chn]
                        if len_chn>2: Eng+=eng[0]
                        else: Eng+=eng[:2]
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
        eng, chn=line
        if not temp.count(chn):
           temp.append(chn)
           Temp[chn]=eng
        #temp.add(chn)
fout.write(end)
fout.close()
cmd='gcin2tab %s'%fname
print(cmd)
system(cmd)
