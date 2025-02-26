python】爬蟲系列之lxml庫介紹和xpath提取網頁數據 本文介紹了Python中lxml庫的使用，包括如何下載安裝及解析HTML字符串和文件。接著詳細講解了XPath語言，用于選取XML或HTML文檔中的節點，包括選取、謂語、選取未知節點和選取若干路徑等。并通過實例展示了XPath在實際爬蟲項目中的應用，如選取元素、屬性、特定條件篩選等。
摘要由CSDN通過智能技術生成
展開
目錄
一、爬蟲提取網頁數據的流程
二、 lxml庫介紹
  2.1 下載安裝lxml庫
  2.2 解析HTML網頁
三、xpath介紹
  1. 選取節點
  2. 謂語
  3. 選取未知節點
  4. 選取若干路徑
活動地址：CSDN21天學習挑戰賽
學習日記 Day16
一、爬蟲提取網頁數據的流程
獲取到html字符串，經過lxml庫解析，變成HTML頁面，再通過xpath的提取可以獲取到需要的數據。
 二、 lxml庫介紹
lxml看是xml和html的解析器，主要功能是解析xml和html中的數據；lxml是一款高性能的python html、xml解析器，也可以利用xpath語法，來定位特定的元素即節點信息。
2.1 下載安裝lxml庫
使用命令 【pip install lxml】下載安裝lxml庫。
  2.2 解析HTML網頁
解析HTML網頁用到了lxml庫中的etree類。
示例1：解析HTML字符串
from lxml import etree
text = '''
<html><body>
    <div class="key">
        <div class="Website">CSDN</div>
        <div class="Lang">python</div>
        <div class="Content">lxml解析HTML</div>
    </div>
</body></html>
'''
# 初始化，傳入一個html形式的字符串
html = etree.HTML(text)
print(html)
print(type(html))
# 將字符串序列化爲html字符串
result = etree.tostring(html).decode('utf-8')
print(result)
print(type(result))
輸出結果：
<Element html at 0x1c9ba29e880>
<class 'lxml.etree._Element'>
<html><body>
    <div class="key">
        <div class="Websit">CSDN</div>
        <div class="Lang">python</div>
        <div class="Content">lxml&#35299;&#26512;HTML</div>
    </div>
</body></html>
<class 'str'>
示例2：讀取并解析HTML文件
from lxml import etree
# 初始化，傳入一個html形式的字符串
html = etree.parse('test.html')
# 將字符串序列化爲html字符串
result = etree.tostring(html).decode('utf-8')
print(result)
print(type(result))
html=etree.HTML(result)
print(html)
print(type)
顯示結果：
<html><body>
    <div class="key">
        <div class="Website">CSDN</div>
        <div class="Lang">python</div>
        <div class="Content">lxml&#35299;&#26512;HTML</div>
    </div>
</body></html>
<class 'str'>
<Element html at 0x19f271bf4c0>
<class 'type'>
總結：
如果有一個html的字符串，可以使用etree.HTML(html_str)函數初始化爲html，然後使用etree.tostring(html).decode('utf-8')序列化爲html字符串；
如果html內容被存儲到文件中，那麼可以使用etree.parse(filename)解析文件，然後使用etree.tostring(html).decode('utf-8')序列化爲html字符串。
三、xpath介紹
xpath是在xml文檔中查找信息的語言，可用來在xml文檔中對元素和屬性進行遍歷。
1. 選取節點
常用的路徑表達式有：
表達式	說明
nodename	選取此節點的所有子節點
/	從根節點選取
//	從匹配選擇的當前節點選擇文檔中的節點，而不考慮它們的位置
.	選取當前節點
..	選取當前節點的父節點
@	選取屬性
示例：
表達式	說明
bookstore	選取bookstore元素的所有子節點
/bookstore	選取根元素bookstore。注釋：假如路徑起始于正斜杠/，此路徑始終代表到某元素的絕對路徑！
bookstore/book	選取屬于bookstore的子元素的所有book元素
//book	選取所有book子元素，不管它們在文檔中的位置
bookstore//book	選擇屬于bookstore元素的後代的所有book元素，不管它們位于bookstore之下的什麼位置
//@lang	選取名爲lang的所有屬性
2. 謂語
謂語用來查找某個特定的節點或者包含某個指定的值的節點，被嵌在方括號中。
路徑表達式	說明
/bookstore/bok[1]	選取屬于bookstore子元素的第一個book元素
/bookstore/book[last()]	選取屬于bookstore子元素的最後一個book元素
/bookstore/book[last()-1]	選取屬于bookstore子元素的倒數第二個book元素
/bookstore/book[position()<3]	選取屬于bookstore子元素的最前面的兩個book元素
//title[@lang]	選取所有擁有名爲lang的屬性的title元素
//title[@lang='eng']	選取所有title元素，且這些元素擁有值爲eng的lang屬性
/bookstore/book[price>35.00]	選取bookstore元素的所有book元素，且其中的price元素的值必須大于35.00
/bookstore/book[price>35.00]/title	選取bookstore元素中的book元素的所有title元素，且其中的price元素的值須大于35.00
3. 選取未知節點
xpath通配符可以用來選取未知的xml元素。
通配符	說明
*	匹配任何元素節點
@*	匹配任何屬性節點
node()	匹配任何類型的節點
示例：
路徑表達式	說明
/bookstore/*	讀取bookstore元素的所有子元素
//*	選取文檔中的所有元素
html/node()/meta/@*	選取html下面任意節點下的meta節點的所有屬性
//title[@*]	選取所有帶有屬性的title元素
4. 選取若干路徑
通過在路徑表達式中使用|運算符，可以選取若干個路徑。
示例：
路徑表達式	結果
//book/title | //book/price	選取book元素的所有title和price元素
//title | //price	選取文檔中的所有title和price元素
/bookstore/book/title | //price	選取屬于bookstore元素的book元素的所有title元素，以及文檔中所有的price元素
xpath使用示例：
新建test.html文件：
<!-- hello.html -->
<div>
    <ul>
         <li class="item-0"><a href="link1.html">first item</a></li>
         <li class="item-1"><a href="link2.html">second item</a></li>
         <li class="item-inactive"><a href="link3.html"><span class="bold">third item</span></a></li>
         <li class="item-1"><a href="link4.html">fourth item</a></li>
         <li class="item-0"><a href="link5.html">fifth item</a></li>
     </ul>
 </div>
⚪ 獲取所有的<li>標簽
# 獲取所有的<li>標簽
from lxml import etree
html = etree.parse('test.html')
print(type(html))
result = html.xpath('//li')
print(result)
print(len(result))
print(type(result))
print(type(result[0]))
顯示結果：
<class 'lxml.etree._ElementTree'>
[<Element li at 0x21eca60ad80>, <Element li at 0x21eca60ae40>, <Element li at 0x21eca60ae80>, <Element li at 0x21eca60aec0>, <Element li at 0x21eca60af00>]
5
<class 'list'>
<class 'lxml.etree._Element'>
⚪ 獲取<li>標簽的所有class屬性
# 獲取<li>標簽的所有class屬性
from lxml import etree
html = etree.parse('test.html')
result = html.xpath('//li/@class')
print(result)
print(len(result))
print(type(result))
print(type(result[0]))
顯示結果：
['item-0', 'item-1', 'item-inactive', 'item-1', 'item-0']
5
<class 'list'>
<class 'lxml.etree._ElementUnicodeResult'>
⚪ 獲取<li>標簽下href爲link1.html的<a>標簽
# 獲取<li>標簽的所有class屬性
from lxml import etree
html = etree.parse('test.html')
result = html.xpath('//li/a[@href="link1.html"]')
print(result)
print(len(result))
print(type(result))
print(type(result[0]))
顯示結果
[<Element a at 0x141b529f380>]
1
<class 'list'>
<class 'lxml.etree._Element'>
⚪ 獲取<li>標簽下的所有<span>標簽
# 獲取<li>標簽的所有class屬性
from lxml import etree
html = etree.parse('test.html')
result = html.xpath('//li//span')
print(result)
顯示結果：
[<Element span at 0x1da409cf480>]
1
⚪ 獲取<li>標簽下的<a>標簽裏的所有class
# 獲取<li>標簽的所有class屬性
from lxml import etree
html = etree.parse('test.html')
result = html.xpath('//li/a//@class')
print(result)
print(len(result))
顯示結果：
['bold']
1
⚪ 獲取最後一個<li>的<a>的href
# 獲取<li>標簽的所有class屬性
from lxml import etree
html = etree.parse('test.html')
result = html.xpath('//li[last()]/a/@href')
print(result)
print(len(result))
顯示結果：
['link5.html']
1
⚪ 獲取倒數第二個元素的內容
# 獲取<li>標簽的所有class屬性
from lxml import etree
html = etree.parse('test.html')
result = html.xpath('//li[last()-1]/a')
print(result)
print(result[0].text)
print(len(result))
顯示結果：
[<Element a at 0x19937dbf300>]
fourth item
1
⚪ 獲取class值爲bold的標簽名
# 獲取<li>標簽的所有class屬性
from lxml import etree
html = etree.parse('test.html')
result = html.xpath('//*[@class="bold"]')
print(result)
print(result[0].tag)
print(len(result))
顯示結果：
[<Element span at 0x1ecc377f2c0>]
span
