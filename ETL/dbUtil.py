from psycopg2 import connect as pgCnnct
from streamlit import secrets, cache_data as stCache
from re import search

def zip2pgConn(db):
  return ' '.join(['='.join(v) for v in list(db)])

@stCache(hash_funcs={"_thread.RLock":lambda _:None, "builtins.weakref":lambda _:None})
#@stCache(hash_funcs={"_thread.RLock":lambda _:None})
def runQuery(query, db='postgres', commitType='select'):     #, mode='scrt'
  if isinstance(db, zip):
    pgConn=zip2pgConn(db) #zip(dbKey, dbValue)
    conn=pgCnnct(pgConn)
  else: conn=pgCnnct(**secrets[db])
  #initCnnction()#db="postgres", **secrets[])
  with conn.cursor() as cur:
    if commitType=='select':
      cur.execute(query)
      return cur.fetchall()
    else:
      cur.execute(query)
      conn.commit()

def dbCLMN(tblSchm='public', tblName=None, db=None): #, mode=None
  clmnQuery=f'''select column_name from information_schema.columns WHERE table_schema = '{tblSchm}' AND table_name = '{tblName}';'''
  CLMN=runQuery(clmnQuery, db=db)
  CLMN=map(lambda x:x[0], CLMN)#[v[0] for v in CLMN]
  #res = any(ele.isupper() for ele in test_str)
  newCLMN=[]
  for clmn in CLMN:
    if any(e.isupper() for e in clmn): newCLMN.append(f'"{clmn}"')
    #if search('[A-Z]', clmn):newCLMN.append(f'"{clmn}"')
    else: newCLMN.append(clmn)
  return newCLMN

def clmnDF(CLMN):
  return [clmn.replace('"', '')for clmn in CLMN]
