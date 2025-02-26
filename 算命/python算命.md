用Python給你算一命 AI研習社 開發者（KaiFaX）
面向全棧工程師的開發者 專注于前端、Java/Python/Go/PHP的技術社區 來源：blog.csdn.net/qq_35488769?type=blog
背景 一直以來，中式占卜都是基于算命先生手工實現，程序繁瑣（往往需要沐浴、計算天時、靜心等等流程）。準備工作復雜（通常需要銅錢等道具），計算方法復雜，需要純手工計算二進制并轉換爲最終的卦象，爲了解決這個問題，筆者基于python實現了一套科學算命工具，用于快速進行占卜。 本文的算命方式采用八卦 + 周易+ 梅花易數實現，腳本基于python3.9.0開發。本人對于周易五行研究較淺，如有疏漏請見諒 最終效果如圖，在運行程序之後，會根據當前的運勢自動獲取你心中所想之事的卦象（本卦、互卦、變卦）
前置知識
基礎原理 首先我們需要了解一些最基本的占卜知識，目前我國幾種比較主流的占卜方式基本都是基于易演化而來。總體而言都是根據某些現象，得到不同的卦象，而不同的卦象最終會代表所占卜事情的開端，發展和結果。 太極生兩儀，兩儀生四象，四象生八卦 這句話相信大家在很多影視作品中都聽說過，但很少有人知道它的真正含義，這句話其實概括了卦象產生的流程。 太極：代表一個絕對混沌的狀態，是一個哲學觀念，非要套用我們的客觀世界，可以理解爲是大爆炸之前宇宙的狀態，所有的物理法則都不生效，當我們還未起卦時就處于這個狀態。
兩儀：同樣是一個哲學觀念，代表一個事物的兩個對立狀態，套用到客觀世界可以是“生-死”、“黑-白”、“清-濁”，在占卜的過程中，我們通常會有“陰-陽”兩個狀態，爲了方便記錄，古人發明了兩個符號代表這兩個狀態，在占卜的時候，一個這樣的狀態我們稱之爲一爻（yao）
四象：當我們將陰陽兩兩組合時，就可以得到四種不同的組合，古人稱之爲四象，注意，這裏的四象同樣是哲學層面的狀態，它可以是“青龍白虎朱雀玄武”，也可以是東南西北四個方位，在占卜的時候，我們通常會用“太陰”、“少陰”、“太陽”、“少陽”來稱呼這四象
八卦：當給我們在四象中增加一爻，也就是三個陰陽組合在一起的時候，我們可以得到八個組合，古人認爲這八個組合可以代表自然界中的八類事物（八中狀態），即是爲八卦
當然，八個狀態用來代表事情的發展方向還是不夠用，于是古人又將八卦（單獨的八卦稱之爲經卦）兩兩組合，從而得到了64個不同的別卦，易經中的六十四卦就是這麼產生的
目前，國內的主流占卜基本都是通過不同的取數方式得到不同的別卦，最終判斷事情的走向。其實對于程序員來說，可以吧兩儀當做一個一位二進制數，有0、1兩個狀態。四象就是兩位二進制數，有00,01,10,11四個狀態。八卦則是三位二進制數，有000,、001、010、011、100、101、110、111四個狀態
如何產生卦象 現在我們知道了卦象是如何演變的，但是我們還沒有能夠得到卦象的方式，其實在占卜的過程中，不同的占卜方式往往最大的區別就是起卦方式不同，我們這裏采用梅花易數的方式起卦。
梅花易數起卦法（這裏只截取兩種起卦方式）：
1、年月日時起卦 即以農歷之年月日總和除以八，以余數爲卦數求上卦;以年月日時總和除以八，以余數爲卦數求下卦，再以年月日時總和除以六，以余數爲動爻。 例：農歷壬申年四月十一日巳時起卦：申年9數，巳時6數。
上卦爲：（年+月+日）÷8，取余數。即：（9+4+11）÷8，此處無余數。
下卦爲：（年+月+日+時）÷8，取余數。即：（9+4+11+6）÷8，余數爲6爲坎卦。
動爻數爲：（年+月+日+時）÷6，取余數。即：（9+4+11+6）除以6，此處無余數。
此卦爲：上卦爲坤，下卦爲坎，動爻爲上爻。
2、直接以數起卦
這是一種簡便而準确率極高的起卦方法。當有人求測某事時，可以讓來人隨意說出兩個數，第一個數取爲上卦，第二個數取爲下卦，兩數之和除以6，余數爲動爻，或者可以隨便借用其他能得到兩數的辦法起卦，如翻書、日歷等等。
開發
我們將梅花易數的起卦方式流程用程序員的話總結一下，流程如下
獲取一個隨機數（我們這裏用當前的時間戳）對8取模，當做上挂（一個三位二進制數）
再獲取一個隨機數，對八取模，當做下挂（一個三位二進制數）
將上述兩個隨機數進行組合，得到一個六位二進制數
六位二進制數轉化爲十進制數并查表，得到本卦
取一個隨機數，對6取模，將上述六位二進制數對應位數的0變爲1,1變爲0，然後轉化爲十進制數并查表，得到變卦
根據本卦和變卦查表，得到占卜結果
import json
import random
import time

#別挂配置數據
gua_data_path = "data.json"

#別卦數據
gua_data_map = { }
fake_delay = 10

#讀取別卦數據
def init_gua_data(json_path):
 with open(gua_data_path,'r',encoding='utf8')as fp:
  global gua_data_map
  gua_data_map = json.load(fp)
#爻圖標映射
yao_icon_map = { 0:"- -", 1:"---" }

base_gua_name_map = { 0:"坤",1:"震",2:"坎",3:"兌",4:"艮",5:"離",6:"巽",7:"乾" } #經卦名

def base_gua_to_yao(gua, yao_length=3): #數字轉化爲二進制數組
 result = []
 while gua >= 1:
  level = 0 if gua % 2 == 0 else 1
  gua //= 2
  result.append(level)
 while len(result) < yao_length:
  result.append(0)
 return result

#二進制數組轉化爲數字
def base_yao_to_gua(array):
 array = array[:]
 while len(array) > 0 and array[-1] == 0:
  array.pop()
 result = 0
 for i in range(len(array)):
  if array[i] == 0:
   continue
  result += pow(2, i)

 return result

def print_gua(gua): #打印一個挂
 yao_list = base_gua_to_yao(gua, 6)
 up_yao_list = yao_list[0:3]
 up = base_yao_to_gua(up_yao_list)

 print(yao_icon_map[up_yao_list[2]])
 print(yao_icon_map[up_yao_list[1]] + " " + base_gua_name_map[up])
 print(yao_icon_map[up_yao_list[0]])

 print("")

 down_yao_list = yao_list[3:6]
 down = base_yao_to_gua(down_yao_list)
 print(yao_icon_map[down_yao_list[2]])
 print(yao_icon_map[down_yao_list[1]] + " " + base_gua_name_map[down])
 print(yao_icon_map[down_yao_list[0]])

def calculate_with_plum_flower(): #使用梅花易數 #起上卦
 print("使用梅花易數♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️♣️")
 print_a_wait_animation("卜上卦：", fake_delay)
 up_base_gua = int(round(time.time() * 1000)) % 8
 up_yao_array = base_gua_to_yao(up_base_gua)
 print("上卦獲取成功,上卦爲:", base_gua_name_map[up_base_gua])
 #起下卦
 print_a_wait_animation("正在獲取下卦：", fake_delay)
 down_base_gua = random.randint(0, 999999999999) % 8
 down_yao_array = base_gua_to_yao(down_base_gua)
 print("上卦獲取成功,下卦爲:", base_gua_name_map[down_base_gua])
 #組成卦象
 print_a_wait_animation("正在組成本卦：", fake_delay)
 print("------------------------------------------------本卦------------------------------------------------")
 yao_list = up_yao_array + down_yao_array
 gua = base_yao_to_gua(yao_list)
 print_gua(gua)
 #讀取本卦象信息
 gua_code = str(base_gua_name_map[up_base_gua]) + str(base_gua_name_map[down_base_gua])
 gua_data = gua_data_map[gua_code]
 print("本卦爲:", gua_data['name'])
 print("辭:", gua_data['words'],"譯:",gua_data['white_words'])
 print("象:", gua_data['picture'],"譯:",gua_data['white_picture'])
 print_a_wait_animation("正在組成互卦：", fake_delay)
 print("------------------------------------------------互卦------------------------------------------------")
 #讀取互卦象信息
 up_hu_yao_list = [yao_list[4],yao_list[5],yao_list[0]]
 up_hu_gua = base_yao_to_gua(up_hu_yao_list)
 down_hu_yao_list =[yao_list[5],yao_list[0],yao_list[1]]
 down_hu_gua = base_yao_to_gua(down_hu_yao_list)
 hu_yao_list = up_hu_yao_list + down_hu_yao_list
 hu_gua = base_yao_to_gua(hu_yao_list)
 hu_gua_code = str(base_gua_name_map[up_hu_gua]) + str(base_gua_name_map[down_hu_gua])
 hu_gua_data = gua_data_map[hu_gua_code]
 print_gua(hu_gua)
 print("互卦爲:", hu_gua_data['name'])
 print("辭:", hu_gua_data['words'],"譯:",hu_gua_data['white_words'])
 print("象:", hu_gua_data['picture'],"譯:",hu_gua_data['white_picture'])
 print_a_wait_animation("正在組成變卦：", fake_delay)
 print("------------------------------------------------變卦------------------------------------------------")
 change_index = int(round(time.time() * 1000)) % 6
 change_yao_list = yao_list[:]
 change_yao_list[change_index] = 0 if change_yao_list[change_index] == 1 else 1
 up_change_yao_list = change_yao_list[0:3]
 up_change_gua = base_yao_to_gua(up_change_yao_list)
 down_change_yao_list =change_yao_list[3:5]
 down_change_gua = base_yao_to_gua(down_change_yao_list)

 change_gua = base_yao_to_gua(change_yao_list)
 print_gua(change_gua)
 change_gua_code = str(base_gua_name_map[up_change_gua]) + str(base_gua_name_map[down_change_gua])
 change_gua_data = gua_data_map[change_gua_code]
 print("變卦爲:", change_gua_data['name'])
 print("辭:", change_gua_data['words'],"譯:",change_gua_data['white_words'])
 print("象:", change_gua_data['picture'],"譯:",change_gua_data['white_picture'])

def print_a_wait_animation(tips,times):
 animation = "|/-\\"
 idx = 0
 for i in range(times):
  print(tips + animation[idx % len(animation)],animation[idx % len(animation)],animation[idx % len(animation)],animation[idx % len(animation)],animation[idx % len(animation)], end="\r"),
  idx += 1
  time.sleep(0.1)

init_gua_data(gua_data_path)
calculate_with_plum_flower()
源代碼
源碼在公衆號開發者後台回復占卜獲取~
1. 回復“m”可以查看歷史記錄；
2. 回復“h”或者“幫助”，查看幫助；
   開發者已開通多個技術群交流學習，請加若飛微信：1321113940  (暗號k）進開發群學習交流
  說明：我們都是開發者。視頻或文章來源于網絡，如涉及版權或有誤，請您與若飛（1321113940）聯系，將在第一時間刪除或者修改，謝謝！
65492feae487f89910f75d4fdf5ab6dd.png
開 發 者 ： KaiFaX
面向全棧工程師的開發者
專注于前端、Java/Python/Go/PHP的技術社區
