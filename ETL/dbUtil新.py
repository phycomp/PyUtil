#import asyncio
#from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg import AsyncConnection as pgCnnct
from streamlit import secrets, cache_data as stCache
from stUtil import rndrCode

async def connect_to_db(): # 異步連接 Postgres 數據庫
    conn = await pgCnnct.connect("dbname=test user=postgres password=yourpassword host=localhost")
    return conn

async def extract_data(conn, query): # 提取數據（Extract）
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(query)
        data = await cur.fetchall()
    return data

async def asyncExtractData(query, db=None, commitType='select'):
    conn = await pgCnnct.connect(**secrets[db])
    #"dbname=test user=postgres password=yourpassword host=localhost"
    async with conn.cursor(row_factory=dict_row) as cur:
      await cur.execute(query)
      #data = 
      #await conn.close()
      return await cur.fetchall()#data

@stCache(hash_funcs={"_thread.RLock":lambda _:None, "builtins.weakref":lambda _:None})
#@stCache(hash_funcs={"_thread.RLock":lambda _:None})
async def runQuery(query, db='postgres', commitType='select'):     #, mode='scrt'
  if isinstance(db, zip):
    pgConn=zip2pgConn(db) #zip(dbKey, dbValue)
    conn=pgCnnct(pgConn)
  else: conn=pgCnnct(**secrets[db])
  async with conn.cursor() as cur:
    if commitType=='select':
      async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(query)
        #data = await cur.fetchall()
        return await cur.fetchall()
    else:
      cur.execute(query)
      conn.commit()

async def dbCLMN(tblSchm='public', tblName=None, db=None): #, mode=None
  clmnQuery=f'''select column_name from information_schema.columns WHERE table_schema = '{tblSchm}' AND table_name = '{tblName}';'''
  CLMN=await asyncExtractData(clmnQuery, db=db)
  #rndrCode(['CLMN=', CLMN])
  CLMN=map(lambda x:x['column_name'], CLMN)#[v[0] for v in CLMN]
  #res = any(ele.isupper() for ele in test_str)
  newCLMN=[]
  for clmn in CLMN:
    if any(e.isupper() for e in clmn): newCLMN.append(f'"{clmn}"')
    #if search('[A-Z]', clmn):newCLMN.append(f'"{clmn}"')
    else: newCLMN.append(clmn)
  return newCLMN

def clmnDF(CLMN):
  return [clmn.replace('"', '')for clmn in CLMN]
