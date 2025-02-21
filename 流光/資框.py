from pandas import read_csv
#df['distance'] = df.apply(lambda row: haversine(40.671, -73.985, row['latitude'], row['longitude']), axis=1)
#len(set(feeym[feeym==2016].apply(str)+fid.ARNHIST))
fid=read_csv('CD_10k.csv')#, sep='\x06')#, iterator=True)
count, fout=0, open('/tmp/rinseCD.tsv', 'w')
feeym=fid.FEE_YM[fid.FEE_YM.apply(lambda v:round(v/100)==2016)]
#feeTup=feeym.itertuple()
#for idx, FID in fid.itertuples():
#       if round(FID.FEE_YM/100)==2016: count+=1
#       if count>=10000:break
#       #fout.write(fid.__next__().to_csv())
#       fid.to_csv('/tmp/rinseCD.tsv', sep='\x06', index=False)
fid['feeym']=fid.FEE_YM.apply(lambda v:1 if int(v/100)==2016 else 0)                           
for idx, value in fid.itertuples():
        if not fid.feeym:fid.to_csv('/tmp/rinseCD.tsv', sep='\x06')

import pandas as pd
df = pd.read_csv("movies.csv")
# the len() method is not available to query, so pre-calculate
title_len = df["title"].str.len()
# build the data frame and send to csv file, title_len is a local variable
df.query('views >= 1000 and starring > 3 and @title_len <= 10').to_csv(...
df.loc[df.categories.apply(lambda cat: 'Food' in cat or 'Restaurants' in cat)]
In [15]: df.categories = df.categories.apply(",".join)

snippet dframe "dframe" b
from dbUtil import runQuery, queryCLMN
sutraCLMN=['章節', '內容']
rsltQuery=runQuery(f'''select {','.join(sutraCLMN)} from $0;''', db='$1')
rsltDF=session_state['rsltDF']=DataFrame(rsltQuery, columns=sutraCLMN, index=None)
sutraDF=session_state['rsltDF']=DataFrame([['', '']], columns=sutraCLMN, index=[0])
背骨=普欄['Path'][普欄['Path']==bbone]
DF['找到']=DF['Path'].where(DF['Type(s)'].str.match('BackboneElement|null'))
DF[DF.fiveClass.str.contains('', case=False)]
SRS1=df[clmn].str.extract('(?P<坪>\d+)坪\s+[xX]\s+((?P<all>全部|公同共有全部)|(公同共有\s?)?(?P<母子>(?P<分母>\d+)分之(?P<分子>\d+)))')
rndrCode([any(SRS1[clmn].notna()) for clmn in newCLMN])
if any([any(SRS1[clmn].notna()) for clmn in newCLMN]): session_state[f'keyCLMN{house}']=clmn
df.assign(profit=df['profit'].where(df['profit_flag'])).groupby('name', as_index=False)[['sales', 'profit']].sum(min_count=1)
endsnippet

snippet dfvec "dfvec" b
from pandas import concat as pndsConcat
from pandas import DataFrame, read_excel, read_csv
DF[DF.$1.str.contains('$2', case=False)]
ccf = ccf.query('Class==1')
DF[DF.$1.str.contains('$2', case=False)]
rsltDF[rsltDF['章節']=='中庸']
dfX=read_csv(Pth/'train.csv', index_col='id')
dfX=pndsConcat([dfX, ccf], axis=0)
SRS1=df[clmn].str.extract('(?P<坪>\d+)坪\s+[xX]\s+((?P<all>全部|公同共有全部)|(公同共有\s?)?(?P<母子>(?P<分母>\d+)分之(?P<分子>\d+)))')
#rndrCode([any(SRS1[clmn].notna()) for clmn in newCLMN])  DF.isnull()
if any([any(SRS1[clmn].notna()) for clmn in newCLMN]):
endsnippet

snippet runQuery "runQuery" b
from dbUtil import runQuery, queryCLMN
sutraCLMN=['章節', '內容']
rsltQuery=runQuery(f'''select {','.join(sutraCLMN)} from $1;''', db='$2')
rsltDF=session_state['rsltDF']=DataFrame(rsltQuery, columns=sutraCLMN, index=None)
sutraDF=session_state['rsltDF']=DataFrame([['', '']], columns=sutraCLMN, index=[0])
sutraCLMN=queryCLMN(tblSchm='public', tblName=tblName, db='sutra')
fullQuery=f'''select {','.join(sutraCLMN)} from {tblName} where 章節~'中庸';'''
rsltQuery=runQuery(fullQuery, db='sutra')
rsltDF=session_state['rsltDF']=DataFrame(rsltQuery, index=None, columns=sutraCLMN)
endsnippet
