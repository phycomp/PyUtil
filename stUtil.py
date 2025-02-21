from grdOpt import mkGrid
from st_aggrid import AgGrid, GridUpdateMode    #, DataReturnMode, GridOptionsBuilder, JsCode

def dsetGrid(dfDset, CLMN):
  grdOpt=mkGrid(dfDset)
  agValue=AgGrid(dfDset, gridOptions=grdOpt, allow_unsafe_jscode=True, theme="balham", reload_data=False, update_mode=GridUpdateMode.SELECTION_CHANGED)
#if any(agValue.selected_rows)==True:
#if hasattr(agValue.selected_rows, '主旨') : #any()==True
  if agValue.selected_rows is not None:   #hasattr(, CLMN) : #any()==True__hash__
    rowNDX=agValue.selected_rows#['_selectedRowNodeInfo']['nodeRowIndex']#[agValue.selected_rows[0]['outcome']][0]
    dsetDF=agValue.selected_rows[CLMN]  #=session_state['rowNDX']['主旨', '內容']
    return dsetDF
from streamlit_extras.stylable_container import stylable_container
from streamlit import code as stCode

def rndrCode(cntxtMsg):
  with stylable_container("codeblock", """code{white-space: pre-wrap !important;}"""): stCode(cntxtMsg) #cntnr.write([dsetDF])
