"py3 vim.vars['python3_host_prog']=vim.env.VIRTUAL_ENV+"/bin/python"
"py3 vimVars=vim.funcs.environ()
"py3 vim.vars['python3_host_prog']=vimVars['VIRTUAL_ENV']+'/bin/python'
let g:python3_host_prog=getenv('VIRTUAL_ENV').'/bin/python'
"py3 vim.exec_lua("")
lua require('PAQs')
"lua require('autocmd2')
"lua require('paqKEY')
lua require('settings')
"lua require('mouse')
lua require('status')
"lua require('ftype')
"lua require('tabline')
"lua require('autocmd')
"lua require('zoom')
lua require('nvimAutopair')
"lua require('nthmFtype')
lua require('initIron')
if exists("g:neovide")
  lua require('neovide')
endif
lua require('replUtil')
"py3<<EOF
"  'vim.g.python3_host_prog = vim.env.VIRTUAL_ENV.."/bin/python"'
"  --require'nvimAutopair'
"  --require'nvtreeSetup'
")
"EOF

"py3 vim.g['chadtree_ignore']='node_modules'
py3file ~/.config/nvim/py/ridPttrn.py
py3file ~/.config/nvim/py/b2g.py
py3file ~/.config/nvim/py/window.py
py3file ~/.config/nvim/py/paqKEY.py
py3file ~/.config/nvim/py/fexp.py
py3file ~/.config/nvim/py/bufUtil.py
py3file ~/.config/nvim/py/KEYs.py
"py3file ~/.config/nvim/py/whichKey.py
py3file ~/.config/nvim/py/cmdline.py
py3file ~/.config/nvim/py/merge.py
"py3file ~/.config/nvim/python3/autocmd.py
py3file ~/.config/nvim/py/srchPttrn.py
py3file ~/.config/nvim/py/zoom.py
"py3file ~/.config/nvim/py/jupyter.py
py3file ~/.config/nvim/py/regMark.py
py3file ~/.config/nvim/py/setColor.py
py3file ~/.config/nvim/py/tabLine.py
py3file ~/.config/nvim/py/commentUtil.py
py3file ~/.config/nvim/py/fontUtil.py
py3file ~/.config/nvim/py/mdUtil.py
py3file ~/.config/nvim/py/tabUtil.py
py3file ~/.config/nvim/py/clipUtil.py
py3file ~/.config/nvim/py/ftUtil.py
"py3file ~/.config/nvim/py/schmUtil.py
py3file ~/.config/nvim/py/pnctUtil.py
py3file ~/.config/nvim/py/rcUtil.py
py3file ~/.config/nvim/py/shUtil.py
py3file ~/.config/nvim/py/sssnUtil.py
py3file ~/.config/nvim/py/vwrHelp.py
py3file ~/.config/nvim/py/webUtil.py

"source ~/.config/nvim/vim/miscAuto.vim
source ~/.config/nvim/vim/bufSel.vim
source ~/.config/nvim/vim/mdUtil.vim
source ~/.config/nvim/vim/WCL.vim
source ~/.config/nvim/vim/ndntProg.vim
"source ~/.config/nvim/vim/tabUtil2.vim
"source ~/.config/nvim/vim/colorschm.vim
"source ~/.config/nvim/vim/tabcolor2.vim
