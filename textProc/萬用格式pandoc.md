\newpage
Pandoc ( 萬用的文件轉換器 ) 由於最近經常用 Markdown 撰寫文件，而我部落格文章現在也開始改用 Markdown 撰寫，寫習慣了之後，發現真的回不去了。現在的我不用再被難用的 Office Word 奴役，因為它功能太多了，多到有點困擾，每次都會被樣式、段落、章節、自動校正搞得很毛。不過，很多專案再交付文件的時候，客戶都會要求用 Word 或 PDF 交付，所以最後還是要轉成 non-Markdown 的格式。我最近發現的 Pandoc 真的很厲害，可以轉換數十種不同的文件格式，甚至於產生 EPUB 電子書都可以，功能非常強大！

# 安裝 Pandoc
完整的安裝文件可以參考 Installing pandoc 說明，這套工具是跨平台的，所以 Windows, macOS, Linux 都可以安裝。

在 Windows 底下，只要透過 Chocolatey 用一行指令就可以安裝完成：

 ```choco install pandoc -y```
如果要輸出 PDF 的話，可能還要另外安裝 MiKTeX 才能用。不過這套 MiKTeX 預設並沒有內建中文字型，PDF 中無法直接輸出中文字，需要額外設定才能用。所以我不建議使用，如果要輸出 PDF 的話，可以先輸出成 docx 檔案，再用 Word 2016 內建的 PDF 輸出功能產生 PDF 檔案。


## 開始使用 Pandoc
這套工具只能在命令列模式下執行，所以你可以先用 pandoc --version 查詢安裝版本。也可以用 pandoc -h 查詢參數說明。完整的說明文件可以參考 Pandoc User’s Guide 使用手冊。預設 Pandoc 在使用的時候，會自動判斷副檔名來決定轉換格式，所以使用上非常簡單。基本用法如下：

`pandoc -o output.html input.md`
所以如果你有一個檔名為 README.md 的 Markdown 文件，想轉換成 Word ( *.docx ) 格式，就可以用以下命令轉換：

 **pandoc -o README.docx README.md**
由於 Pandoc 也支援 Pipe 的方式輸入，所以你也可以這樣執行：

type README.md | pandoc -o README.docx
cat README.md | pandoc -o README.html
預設 Pandoc 都會把所有文件輸出為 部份文件 (document fragment)，意思也就是說，如果你將 Markdown 輸出為 html 格式，它就只會產生 HTML 的 <body> 內的部份內容而已，不會是完整的一份 HTML 文件。如果你想輸出一份獨立的文件，可以加上 -s 參數，如下範例：

pandoc -o README.docx -s README.md
## 指定輸入輸出格式

Pandoc 真正強大的地方，在於他可以自由的轉換格式，還可以對輸入文件與輸出文件進行微調，產生你想要的文件格式。

如果你想要指定輸入與輸出格式，可以用 -f FORMAT 與 -t FORMAT 來設定。例如你想要將 Markdown 輸出成 EPUB v2 格式的電子書文件，就可以用以下命令完成文件格式轉換：

pandoc -f markdown -t epub2 -o README.epub README.md
這種用法，有一個非常強大的能力，就是可以把任意網頁轉換成任意格式的文件！

例如以下命令，就可以將將任意線上的 Markdown 文件轉換成一份 Word (docx) 檔案：

pandoc -f markdown -t docx -o CHANGELOG.docx --request-header User-Agent:"Mozilla/5.0" https://raw.githubusercontent.com/angular/angular/master/CHANGELOG.md
當然，你想要轉換成什麼格式都可以！

更多支援的格式 你可以用 pandoc --list-input-formats 查詢所有支援的 輸入格式，如下清單：

commonmark (CommonMark Markdown)
creole (Creole 1.0)
docbook (DocBook)
docx (Word docx)
epub (EPUB)
fb2 (FictionBook2 e-book)
gfm (GitHub-Flavored Markdown), or the deprecated and less accurate markdown_github; use markdown_github only if you need extensions not supported in gfm.
haddock (Haddock markup)
html (HTML)
jats (JATS XML)
json (JSON version of native AST)
latex (LaTeX)
markdown (Pandoc's Markdown)
markdown_mmd (MultiMarkdown)
markdown_phpextra (PHP Markdown Extra)
markdown_strict (original unextended Markdown)
mediawiki (MediaWiki markup)
muse (Muse)
native (native Haskell)
odt (ODT)
opml (OPML)
org (Emacs Org mode)
rst (reStructuredText)
t2t (txt2tags)
textile (Textile)
tikiwiki (TikiWiki markup)
twiki (TWiki markup)
vimwiki (Vimwiki)
Extensions can be individually enabled or disabled by appending +EXTENSION or -EXTENSION to the format name. See Extensions below, for a list of extensions and their names. See --list-input-formats and --list-extensions, below.

也可以透過 pandoc --list-output-formats 查詢所有支援的 輸出格式，如下清單：

asciidoc (AsciiDoc)
beamer (LaTeX beamer slide show)
commonmark (CommonMark Markdown)
context (ConTeXt)
docbook or docbook4 (DocBook 4)
docbook5 (DocBook 5)
docx (Word docx)
dokuwiki (DokuWiki markup)
epub or epub3 (EPUB v3 book)
epub2 (EPUB v2)
fb2 (FictionBook2 e-book)
gfm (GitHub-Flavored Markdown), or the deprecated and less accurate markdown_github; use markdown_github only if you need extensions not supported in gfm.
haddock (Haddock markup)
html or html5 (HTML, i.e. HTML5/XHTML polyglot markup)
html4 (XHTML 1.0 Transitional)
icml (InDesign ICML)
jats (JATS XML)
json (JSON version of native AST)
latex (LaTeX)
man (groff man)
markdown (Pandoc's Markdown)
markdown_mmd (MultiMarkdown)
markdown_phpextra (PHP Markdown Extra)
markdown_strict (original unextended Markdown)
mediawiki (MediaWiki markup)
ms (groff ms)
muse (Muse),
native (native Haskell),
odt (OpenOffice text document)
opml (OPML)
opendocument (OpenDocument)
org (Emacs Org mode)
plain (plain text),
pptx (PowerPoint slide show)
rst (reStructuredText)
rtf (Rich Text Format)
texinfo (GNU Texinfo)
textile (Textile)
slideous (Slideous HTML and JavaScript slide show)
slidy (Slidy HTML and JavaScript slide show)
dzslides (DZSlides HTML5 + JavaScript slide show),
revealjs (reveal.js HTML5 + JavaScript slide show)
s5 (S5 HTML and JavaScript slide show)
tei (TEI Simple)
zimwiki (ZimWiki markup)
the path of a custom lua writer, see Custom writers below
※ 請注意：odt、docx 與 epub 這三種文件類型，必須使用 -o 參數指定輸出檔案名稱，其他的文件類型如果沒加上 -o 參數，預設會從 stdout 輸出。

無論是輸入格式或輸出格式，都可以在格式名稱後方加上 +EXTENSION 或 -EXTENSION 來啟用特定 reader 或 writer 的擴充屬性設定，調整輸入或輸出的細節！
