import flaski.database
import flaski.dbmodels
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np


classes =  ["アゴハゼ","アカエイ","アカエソ","アカヒトデ","アカクラゲ","アカササノハベラ","アケボノチョウチョウウオ","アケウス",
           "アマオブネガイ","アメフラシ","アナアオサ","アンドンクラゲ","アオヤガラ","バテイラ","ベッコウガサ","ベニカエルアンコウ",
           "ベニツケギンポ","ベリルイソギンチャク","ボラ","コモチイソギンチャク","ダイダイイソカイメン", "ダイナンギンポ", "ダンゴウオ",
           "ドングリガヤ", "ドロメ", "エビスガイ", "エゾカサネカンザシゴカイ", "ガンガゼ", "ガザミ", "ギンカクラゲ", "ギンユゴイ", 
          "ゴマフニナ",
           "ゴマモンガラ", "ゴンズイ", "ハブクラゲ","ハコフグ" ,"ハナアナゴ", "ハナミノカサゴ","ハナオコゼ", "ハナタツ","ハネウミヒドラ",
          "ハオコゼ",
           "ヒイラギ",
           "ヒメイソギンチャク", "ヒメジ", "ヒメケハダヒザラガイ","ヒトエガイ", "ヒザラガイ", "ホンベラ", "ホンソメワケベラ", 
          "ホシノハゼ",
           "ホシササノハベラ","ホウボウ", "フクイカムリ", "フウライチョウチョウウオ", "フウセンウミウシ", "ヒョウモンダコ",
          "イバラカンザシゴカイ",
           "イボイワオウギガニ", "イボヤギ", "イラモ", "イロカエルアンコウ", "イシダイ", "イシダタミ", "イシガキダイ", "イソバナ",
           "イソガニ", "イソハゼ", "イソカニダマシ", "イソスズメダイ", "イトマキヒトデ", "イワガニ", "ジャノメアメフラシ", 
          "カエルアンコウ",
           "カゴカキダイ", "カイウミヒドラ", "カミソリウオ", "カモガイ", "カンムリベラ","カサゴ", "カツオノエボシ", 
           "カツオノカンムリ", "カワハギ", "ケハダヒザラガイ", "ケムシヒザラガイ", "ケヤリムシ", "キバナトサカ", "キイロイボウミウシ",
           "キンメモドキ", "キンセンイシモチ", "キンチャクダイ", "キヌバリ", "キタマクラ", "コバンヒメジ","コノハガニ", "コロダイ", 
           "コショウダイ", "コスジイシモチ", "クボガイ", "クマノコガイ" , "クマノミ", "クモハゼ", "クロダイ", "クロガヤ", 
          "クロホシイシモチ",
           "クロフジツボ", "クロイシモチ", "クロイソカイメン", "クロメジナ", "クロスジウミウシ", "クサフグ", "キュウセン", "マダイ",
          "マヒトデ",
           "マメダコ", "マメコブシガニ", "マナマコ", "マトウダイ", "マツバガイ", "マツカサウオ", "メイチダイ", "メジナ", "ミドリヒモムシ",
           "ミドリイソギンチャク", "ミギマキ", "ミナミハコフグ", "ミノヒラムシ", "ミノカサゴ", "ミサキヒモムシ", "ミスガイ", 
           "ミツボシクロスズメダイ", "ミズクラゲ", "モンガラカワハギ", "ムカデミノウミウシ", "ムラサキカイメン", "ナベカ",
          "ナミイソカイメン",
           "ナミノハナ", "ネンブツダイ", "ニシキベラ", "オハグロベラ", "オジサン", "オカメブンプク","オキナメジナ", "オニカサゴ",
          "オオタマウミヒドラ", 
           "オオトゲトサカ", "オオツノヒラムシ", "オトヒメゴカイ", "オヤビッチャ", "ロクセンスズメダイ", "サンハチウロコムシ",
          "サザナミヤッコ",
           "シロガヤ", "シマメノウフネガイ", "ソラスズメダイ", "スベスベマンジュウガニ","スイ", "スジホシムシモドキ", 
           "スカシカシパン","スナジャワン", "スズメダイ",
           "ショウジンガニ", "タカベ", "タカノハダイ", "タコベラ", "タカノマクラ", "タナバタウオ", "タテジマイソギンチャク",
           "タテジマキンチャクダイ", "タツノイトコ", "タツノオトシゴ", "トゲチョウチョウウオ", "トコブシ","トラギス", "ツバメコノシロ",
           "ツメタガイ", "ツノヒラムシ", "ツユベラ","チョウハン", "チョウチョウウオ", "ウマヅラハギ", "ウメボシイソギンチャク",
           "ウミヒノキ", "ウミニナ", "ウミテング", "ウノアシ", "ウスヒラムシ", "ウスヒザラガイ", "ウツボ", "ウズイチモンジ", 
          "ヤッコカンザシゴカイ",
           "ヨメガカサ", "ヨメヒメジ", "ヨロイイソギンチャク", "ヨツアナカシパン", "ヨウジウオ"]

num_classes = len(classes)
image_size = 224

#モデルの指定
MODEL_FILE_PATH = 'iso2_cnn.h5'


#uploadされた画像を判定する関数、名前と確率を表示
def judge(file_path):

    model=load_model(MODEL_FILE_PATH)
    model.load_weights('iso2_cnn_weights.h5')
    image = Image.open(file_path)
    image = image.convert("RGB")
    width, height = image.size
    if  width == height:#縦横が同じ長さだったらそのまま使う
        image = image
    elif width > height:
            result = Image.new(image.mode, (width, width), (0,0,0))#縦で足りない長さを黒で埋めた正方形の台紙を用意
            result.paste(image, (0, (width - height) // 2))#縦幅の中央を決めて画像を貼り付け
            image =  result
    else:
            result = Image.new(image.mode, (height, height), (0,0,0))#縦が長い場合、横を黒で埋める
            result.paste(image, ((height - width) // 2, 0))#縦の中央を決めて画像を配置
            image = result

    image = image.resize((image_size,image_size))
    data = np.asarray(image) / 255.0
    X = []
    X.append(data)
    X = np.array(X)
 
    results = []
    result = model.predict([X])[0]#リストXの確率を得る
    predicted = result.argsort()[::-1]#値(確率)がソートされた際に元の配列のインデックス番号を得る

    up_thirds=predicted[0:3]#上から3つ得る

    for up_third in up_thirds:
    
        classify = classes[up_third]#一つずつ取り出したインデックス番号と対応するカタカナ名を得る
    
        per = int(result[up_third] * 100)

        results.append(get_fish_data(classify,per))
    return results




#情報をDBから取り出し辞書型で返す
def get_fish_data(fishname,per):
    ses = flaski.database.db_session()
    fish_master = flaski.dbmodels.FishMaster
    fish_data = ses.query(fish_master).filter(fish_master.fish_name==fishname).first()#予測結果fishnameと一致するものをfishmasterから選ぶ

    fish_data_dict ={}

    if not fish_data is None:
        fish_data_dict['fish_name']=fish_data.fish_name
        if fish_data.poison ==1:
            fish_data_dict['poison']='毒あり'
        else:
            fish_data_dict['poison']=''

        fish_data_dict['poison_part']=fish_data.poison_part
        fish_data_dict['wiki_url']=fish_data.wiki_url
        fish_data_dict['per']=per
    
    return fish_data_dict                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 