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

from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder, JsCode
def mkGrid(dsetDF):
  gb = GridOptionsBuilder.from_dataframe(dsetDF)
  #gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=3)
  gb.configure_pagination(enabled=True, paginationAutoPageSize=True, paginationPageSize=20) #
  #toolTip = JsCode(""" function(params) { return '<span title="' + params.value + '">'+params.value+'</span>';  }; """) # if using with cellRenderer
  gb.configure_columns(['annttInfo', 'outcome'], editable=True) #, cellRenderer=toolTip
  #gb.configure_pagination(enabled=True, paginationPageSize=10)  #paginationAutoPageSize=True, pagination=true, 

  #gridOptions = GridOptionsBuilder.from_dataframe(df)
  #gridOptions.configure_column('Name', editable=False, cellRenderer=tooltipjs)
  js = JsCode("""function(e) { let api = e.api;
    let rowIndex = e.rowIndex;
    let col = e.column.colId;
    let rowNode = api.getDisplayedRowAtIndex(rowIndex);
    console.log("column index: " + col + ", row index: " + rowIndex); };""")
  gb.configure_grid_options(onCellClicked=js)
  gb.configure_selection(selection_mode ='single')
  grdOpt = gb.build()
  return grdOpt
