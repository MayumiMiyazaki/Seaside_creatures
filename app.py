import os
from flask import Flask,render_template,request,redirect,url_for,flash
from werkzeug.utils import secure_filename
import pykakasi
from models import judge
from flask import send_from_directory
from flask_wtf.csrf import CSRFProtect
from flask_basicauth import BasicAuth

UPLOAD_FOLDER = './static/images/upload'
#アップロードを許可する拡張子
ALLOWED_EXTENSIONS = set(['png','jpg','gif','jpeg'])

#appというFlaskオブジェクトをインスタンス化
app = Flask(__name__)
app.config['SECRET_KEY']=os.urandom(24)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH']=16*1024*1024#16MBまで

csrf = CSRFProtect(app)
app.config.update(SESSION_COOKIE_SECURE=True,SESSION_COOKIE_HTTPONLY=True,SESSION_COOKIE_SAMESITE='Lax')

#Basic認証
#ここで環境変数を取得し、Herokuにあるユーザーネームとパスワードを取りに行く
app.config['BASIC_AUTH_USERNAME'] = os.environ['BASIC_AUTH_USERNAME']
app.config['BASIC_AUTH_PASSWORD'] = os.environ['BASIC_AUTH_PASSWORD']
app.config['BASIC_AUTH_FORCE'] = True
basic_auth = BasicAuth(app)

#全てのリクエストの後に
#Content-TypeをHTMLと認識してもJavascriptは実行しない
#xfo="DENY"外部、オリジンともにページ埋め込みを拒否
@app.after_request
def set_header(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    return response

#漢字かなまじりの文をローマ字とする
class Kakashi:
    kakashi = pykakasi.kakasi()
    kakashi.setMode('H','a')
    kakashi.setMode('K','a')
    kakashi.setMode('J','a')
    conv = kakashi.getConverter()

    @classmethod
    def japanese_to_ascii(cls,japanese):
        return cls.conv.do(japanese)

#画像ファイル名を右から1要素だけ分割、検証、lower()で小文字に変換
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

#トップページの挙動
@app.route('/',methods=['GET','POST'])
@basic_auth.required#Basic認証の要求
def index():
    if request.method == 'POST':
        
        if 'file' not in request.files:
            flash('ファイルを選択してください')
            return render_template('index.html')
        
        file = request.files['file']
        
        if file.filename == '' :
            flash('画像がありません')
            return render_template('index.html')

        if not allowed_file(file.filename):
            flash('拡張子は png, jpg, jpeg, gif のみ使用可能です')
            return render_template('index.html')
        else:
            ascii_filename = Kakashi.japanese_to_ascii(file.filename)
            
            filename = secure_filename(ascii_filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)

            file.save(file_path)
            return render_template('display_img.html',file_path=file_path)
                #file_pathにあるアップロードした画像を表示
    else:
        return render_template('index.html')



#予測の挙動
@app.route('/classify',methods=['GET','POST'])
def classify_img():
    if request.method == 'POST':
        file_path = request.form['image']#imageフォームのvalueであるfile_pathを得る
        data = judge(file_path)

        return render_template('classify_img.html',fish_data=data,file_path=file_path)

    else:
        return render_template('index.html')

#メニューバーの動き
@app.route('/privacy')
def privacy():
    return render_template('privacy.html')
#メニューバーの動き
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

            #実行関数
if __name__=='__main__':
    app.run()
