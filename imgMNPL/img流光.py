from streamlit import session_state, columns, select_slider, selectbox, write, image as stImage, checkbox, text_input
from os import listdir
from math import ceil
from pandas import DataFrame
from stUtil import rndrCode

directory = 'images'
files = listdir(directory)
rndrCode(files)

def initialize():    
    df = DataFrame({'file':files, 'incorrect':[False]*len(files), 'label':['']*len(files)})
    df.set_index('file', inplace=True)
    return df

if 'df' not in session_state:
    df = initialize()
    session_state.df = df
else:
    df = session_state.df 


leftPane, midPane, rightPane = columns(3)
with leftPane:
    batch_size = select_slider("Batch size:",range(10,110,10))
with midPane:
    row_size = select_slider("Row size:", range(1,6), value = 5)
num_batches = ceil(len(files)/batch_size)
with rightPane:
    page = selectbox("Page", range(1,num_batches+1))

def update (image, col): 
    df.at[image,col] = session_state[f'{col}_{image}']
    if session_state[f'incorrect_{image}'] == False:
       session_state[f'label_{image}'] = ''
       df.at[image,'label'] = ''

batch = files[(page-1)*batch_size : page*batch_size]

grid = columns(row_size)
col = 0
for image in batch:
    with grid[col]:
        stImage(f'{directory}/{image}', caption='bike')
        checkbox("Incorrect", key=f'incorrect_{image}', value = df.at[image,'incorrect'], on_change=update, args=(image,'incorrect'))
        if df.at[image,'incorrect']:
            text_input('New label:', key=f'label_{image}', value = df.at[image,'label'], on_change=update, args=(image,'label'))
        else:
            write('##')
            write('##')
            write('###')
    col = (col + 1) % row_size
def output():
    rndrCode('## Corrections')
    df[df['incorrect']==True]
