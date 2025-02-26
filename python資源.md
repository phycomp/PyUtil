Python資料整理 weixin_33845477

Python基本安裝：

    *  http://www.python.org/ 官方標準Python開發包和支持環境，同時也是Python的官方網站；
    *  http://www.activestate.com/ 集成多個有用插件的強大非官方版本，特別是針對Windows環境有不少改進；

Python文檔：

    *  http://www.python.org/doc/current/lib/lib.html Python庫參考手冊。
    *  http://www.byteofpython.info/ 可以代替Tutorial使用，有中文譯版的入門書籍。
    *  http://diveintopython.org/ 一本比較全面易懂的入門書，中文版翻譯最近進步爲很及時的5.4了。
    *  http://www.python.org/peps/pep-0008.html 建議采用的Python編碼風格。
    *  http://doc.zoomquiet.org/ 包括Python內容的一個挺全面的文檔集。

常用插件：

    * h ttp://www.pfdubois.com/numpy/ Python的數學運算庫，有時候一些別的庫也會調用裏面的一些功能，比如數組什麼的；
    *  http://www.pythonware.com/products/pil/ Python下著名的圖像處理庫Pil；
    *  http://simpy.sourceforge.net/ 利用Python進行仿真、模藕的解決方案；
    * Matplotlib 據說是一個用來繪制二維圖形的Python模塊，它克隆了許多Matlab中的函數， 用以幫助Python用戶輕松獲得高質量(達到出版水平)的二維圖形；
    *  http://www.amk.ca/python/code/crypto python的加解密擴展模塊；
    * http://cjkpython.i18n.org/ 提供與python有關的CJK語言支持功能：轉碼、顯示之類。
    * Psyco、Pyrex：兩個用于提高Python代碼運行效率的解決方案；
    * Pyflakes、PyChecker、PyLint：都是用來做Python代碼語法檢查的工具。
    *  http://wxpython.sourceforge.net/ 基于wxWindows的易用且強大的圖形界面開發包wxPython；
    * http://www.pygame.org/ 用Python幫助開發遊戲的庫，也可以用這個來播放視頻或者音頻什麼的，大概依靠的是SDL；
    *  http://starship.python.net/crew/theller/py2exe/ win下將Python程序編譯爲可執行程序的工具，是一個讓程序脫離Python運行環境的辦法，也可以生成Windows服務或者COM組件。其他能完成Python腳本到可執行文件這個工作的還有Gordon McMillan's Installer、Linux專用的freeze以及py2app、setuptools等。不過此類工具難免與一些模塊有一些兼容性的問題，需要現用現測一下。
    * 嵌入式數據庫：BerkeleyDB的Python版，當然還有其他的好多。
    * PEAK提供一些關于超輕量線程框架等基礎性重要類庫實現。

部分常用工具：

    *  http://www.scons.org/ Java有Ant這個巨火的構建工具，Python的特性允許我們構建更新類型的構建工具，就是scons了。
    * Python Sidebar for Mozilla FireFox的一個插件，提供一個用來查看Python文檔、函數庫的側邊欄。
    * IPython 很好用的Python Shell。wxPython發行版還自帶了PyCrust、PyShell、PyAlaCarte和PyAlaMode等幾個工具，分別是圖形界面Shell和代碼編輯器等，分別具有不同特點可以根據自己的需要選用。
    * Easy Install 快速安裝Python模塊的易用性解決方案。

推薦資源：

    * Parnassus山的拱頂 巨大的Python代碼庫，包羅萬象。既可以從上面下載代碼參考學習，同時也是與Python有關程序的大列表。
    * Python號星際旅行船 著名Python社區，代碼、文檔、高人這裏都有。
    * faqts.com的Python程序設計知識數據庫 Python程序設計知識庫，都是與Python有關的程序設計問題及解決方法。
    * 啄木鳥 Pythonic 開源社區 著名的（也可以說是最好的）國內Python開源社區。

代碼示例：

    *  http://newedit.tigris.org/technical.html Limodou的NewEdit編輯器的技術手冊，討論了一些關于插件接口實現、i18實現、wxPython使用有關的問題，值得參考。

其他東西：

    *  http://www.forum.nokia.com/main/0,,034-821,00.html Nokia居然發布了在Series 60系統上運行Python程序（圖形界面用wxPython）的庫，還有一個Wiki頁是關于這個的：h ttp://www.postneo.com/postwiki/moin.cgi/PythonForSeries60 。Python4Symbian這個頁面是記錄的我的使用經驗。
    * pyre：使用Python完成高性能計算需求的包，真的可以做到麼？還沒研究。
    * Parallel Python：純Python的并行計算解決方案。相關中文參考頁面
    * Pexpect：用Python作爲外殼控制其他命令行程序的工具（比如Linux下標準的ftp、telnet程序什麼的），還沒有測試可用程度如何。
    * pyjamas：Google GWT的Python克隆，還處在早期版本階段。
    * Durus：Python的對象數據庫。

有意思的東西：

    * Howie：用Python實現的MSN對話機器人。
    * Cankiri：用一個Python腳本實現的屏幕錄像機。

參考資料

    * ZDNET文章：學習Python語言必備的資源
    * Pythonic Web 應用平台對比
    * 在wxPython下進行圖像處理的經驗 （其實，僅使用wxPython也可以完成很多比較基礎的圖像處理工作，具體可以參照《wxPython in Action》一書的第12節）
    * 通過win32擴展接口使用Python獲得系統進程列表的方法
    * 如何獲得Python腳本所在的目錄位置
    * Python的縮進問題
    * py2exe使用中遇到的問題
    * idle的中文支持問題
    * 序列化存儲 Python 對象

Python IDE

我的IDE選擇經驗

    * [url]http://www.xored.com Trustudio[/url] 一個基于Eclipse的、同時支持Python和PHP的插件，曾經是我最喜歡的Python IDE環境，功能相當全了，不過有些細節不完善以致不大好用。
    *  http://pydev.sourceforge.net/ 另一個基于Eclipse的，非常棒的Python環境，改進速度非常快，現在是我最喜歡的IDE。
    *  http://www-900.ibm.com/developerWorks/cn/opensource/os-ecant/index.shtml 用 Eclipse 和 Ant 進行 Python 開發
    *  http://www.die-offenbachs.de/detlev/eric3.html ERIC3 基于QT實現的不錯的PYTHON IDE,支持調試，支持自動補全，甚至也支持重構，曾經在Debian下用它，但圖形界面開發主要輔助qt，我傾向wxpython，所以最後還是放棄了這個。
    *  http://www.scintilla.org/ 同時支持Win和Linux的源代碼編輯器，似乎支持Python文件的編輯。
    * h ttp://boa-constructor.sourceforge.net/ 著名的基于WxPython的GUI快速生成用的Python IDE，但是開發進度實在太差了……
    *  http://pype.sourceforge.net/ 成熟的Python代碼編輯器，號稱功能介于EMACS和IDLE之間的編輯器。
    *  http://www.stani.be/python/spe SPE：號稱是一個Full Featured編輯器，集成WxGlade支持GUI設計。
