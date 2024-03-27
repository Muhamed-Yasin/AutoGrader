from flask import Flask, render_template,request,flash,redirect,url_for,session
from werkzeug.utils import secure_filename
import os
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'Flask_Upload/static'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'Akshay'
app.config['MYSQL_PASSWORD'] = 'secret'
app.config['MYSQL_DB'] = 'pythonlogin'

mysql = MySQL(app)

@app.route('/home', methods=['GET',"POST"])
def upload():
    if request.method=='POST':
        if request.files and 'Answers' in request.files and 'Key' in request.files :
            if request.files['Key'].filename.split('.')[1] == 'txt' and request.files['Answers'].filename.split('.')[1] == 'txt':
                file1 = request.files['Key']
                file1.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file1.filename)))
                file2=request.files['Answers']
                file2.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file2.filename)))
                return calc()
            else:
                flash("Please upload '.txt' files only")
                render_template('index1.html') 
        else:
            flash("Please upload both files")
            render_template('index1.html') 
    return render_template('index1.html')


@app.route('/', methods=['GET',"POST"])
@app.route('/pythonlogin/', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        global jjj
        jjj=username
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password,))
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            return redirect('/home/enter')
        else:
            msg = 'Incorrect username/password!'
    return render_template('index.html', msg=msg)


@app.route('/pythonlogin/logout')
def logout():
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   return redirect(url_for('login'))

@app.route('/pythonlogin/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, password, email,))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        msg = 'Please fill out the form!'
    return render_template('register.html', msg=msg)


@app.route('/pythonlogin/profile')
def profile():
    if 'loggedin' in session:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        return render_template('profile.html', account=account)
    return redirect(url_for('login'))


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import seaborn as sns
from sklearn.metrics import pairwise
import tensorflow_hub as hub
import tensorflow_text as text
from gingerit.gingerit import GingerIt
from sentence_transformers import SentenceTransformer
import difflib
import sys


@app.route('/home/enter', methods=['GET',"POST"])
def enter():
    return render_template('enter.html')
    

@app.route('/home/profile', methods=['GET',"POST"])
def prof():
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM accounts WHERE username = %s', (jjj,))
    account = cursor.fetchone()

    username = account['username']
    password = account['password']
    email = account['email']
    return render_template('prof.html',username=username,password=password,email=email)


'''@app.route('/home/profile', methods=['GET',"POST"])
def profile():
    return render_template('profile.html')'''



def calc():
    print("we are in calc")
    f1 = open('Flask_Upload/static/'+request.files['Answers'].filename, "r")
    x=f1.read()
    z1=x.split("\n")
    x=[]
    for i in z1:
        if len(i)!=0 and i!=" ":
            x.append(i)
    x1=list(x)

    f2 = open('./Flask_Upload/static/'+request.files['Key'].filename, "r")
    y=f2.read()
    z2=y.split("\n")
    y=[]
    for j in z2:
        if len(j)!=0 and j!=" ":
            y.append(j)
    y1=list(y)

    print("CALC")
    leng=len(x)


    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=5000,stop_words='english')
    from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=3000)


    b = np.zeros(len(x), dtype = int)
    marks = np.ones(len(x), dtype = int)

    parser=GingerIt()

    for i in range(len(x)):
        z=x[i]
        ct=parser.parse(z)
        
        f=len(ct['corrections'])
        b[i]=f

    a=nltk.sent_tokenize(y[0])
    for i in range(len(a)):
        words = nltk.word_tokenize(a[i])
        words = [ps.stem(word) for word in words if word not in set(stopwords.words('english'))]
        a[i] = ' '.join(words) 
    key=a

    answers=[]
    for i in range(len(x)):
        a=nltk.sent_tokenize(x[i])
        for i in range(len(a)):
            words = nltk.word_tokenize(a[i])
            words = [ps.stem(word) for word in words if word not in set(stopwords.words('english'))]
            a[i] = ' '.join(words)  
        answers.append(a)

    df = pd.DataFrame(answers, columns=['answers'])


    k_cv = cv.fit_transform(key)
    a_cv = cv.transform(df['answers']).toarray()


    k_tfidf = tfidf.fit_transform(key)
    a_tfidf = tfidf.transform(df['answers']).toarray()


    sim_cv = cosine_similarity(k_cv,a_cv)
    sim_tfidf = cosine_similarity(k_tfidf,a_tfidf)


    sim_gon=sim_cv*sim_tfidf
    sim_gon=sim_gon[0]

    for i in range(len(df)):
        if list(sim_gon)[i]==0:
            marks[i]=0


    model = SentenceTransformer('all-MiniLM-L6-v2')

    k_sb1 = model.encode(key)
    a_sb1 = model.encode(df['answers'])

    sim_sb1=cosine_similarity(k_sb1,a_sb1)
    sim_sb1=sim_sb1[0]


    m1=SentenceTransformer('paraphrase-MiniLM-L6-v2')
    k_sb2 = model.encode(key)
    a_sb2 = model.encode(df['answers'])

    sim_sb2=cosine_similarity(k_sb2,a_sb2)
    sim_sb2=sim_sb2[0]

    m2=SentenceTransformer('paraphrase-distilroberta-base-v1')
    k_sb3 = model.encode(key)
    a_sb3 = model.encode(df['answers'])

    sim_sb3=cosine_similarity(k_sb3,a_sb3)
    sim_sb3=sim_sb3[0]



    BERT_MODEL = "https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/4" 
    PREPROCESS_MODEL = "https://tfhub.dev/tensorflow/bert_zh_preprocess/3"

    preprocess = hub.load(PREPROCESS_MODEL)
    bert = hub.load(BERT_MODEL)
    inputs = preprocess(df['answers'])
    outputs = bert(inputs)
    a_b1=outputs["pooled_output"]
    inputs = preprocess(key)
    outputs = bert(inputs)
    k_b1=outputs["pooled_output"]

    sim_b1=cosine_similarity(k_b1,a_b1)
    sim_b1=sim_b1[0]


    BERT_MODEL = "https://tfhub.dev/google/experts/bert/wiki_books/2" 
    PREPROCESS_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    preprocess = hub.load(PREPROCESS_MODEL)
    bert = hub.load(BERT_MODEL)
    inputs = preprocess(df['answers'])
    outputs = bert(inputs)
    a_b2=outputs["pooled_output"]
    inputs = preprocess(key)
    outputs = bert(inputs)
    k_b2=outputs["pooled_output"]

    sim_b2=cosine_similarity(k_b2,a_b2)
    sim_b2=sim_b2[0]


    sim_sb=(sim_sb1+sim_sb2+sim_sb3)/3
    sim_b=(sim_b1+sim_b2)/2
    sim=sim_b*sim_sb

    global length
    global fd
    fin=sim*10-b*0.5
    fin=fin*marks
    final=list(fin)
    final=[round(x,2) for x in final]
    fd=[]
    for i in range(len(x1)):
        n=[x1[i],final[i]]
        fd.append(n)
    print(fd)
    length=len(fd)
    return redirect('/result')

@app.route('/result')
def result():
    return render_template('basic.html',data=fd,length=length)

'''if __name__ == '__main__':
    app.run(debug=True)'''








@app.route('/home/second', methods=['GET',"POST"])
def upload1():
    if request.method=='POST':
        if request.files and 'Answers' in request.files and 'Key' in request.files :
            if request.files['Key'].filename.split('.')[1] == 'txt' and request.files['Answers'].filename.split('.')[1] == 'txt':
                file1 = request.files['Key']
                file1.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file1.filename)))
                file2=request.files['Answers']
                file2.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file2.filename)))
                return calc1()
            else:
                flash("Please upload '.txt' files only")
        else:
            flash("Please upload both files")
    return render_template('index2.html')



def calc1():
    
    print("we are in calc1")
    f1 = open('Flask_Upload/static/'+request.files['Answers'].filename, "r")
    x=f1.read()
    z1=x.split("\n")
    x=[]
    for i in z1:
        if len(i)!=0 and i!=" ":
            x.append(i)
    x1=list(x)

    f2 = open('./Flask_Upload/static/'+request.files['Key'].filename, "r")
    y=f2.read()
    z2=y.split("\n")
    y=[]
    for j in z2:
        if len(j)!=0 and j!=" ":
            y.append(j)
    y1=list(y)
    print("CALC1111")
    leng=len(x)
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=5000,stop_words='english')
    from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=3000)


    b = np.zeros(len(x), dtype = int)
    marks = np.ones(len(x), dtype = int)

    parser=GingerIt()
    for i in range(len(x)):
        z=x[i]
        ct=parser.parse(z)
        
        f=len(ct['corrections'])
        b[i]=f


    keys=[]
    for i in range(len(y)):
        a=nltk.sent_tokenize(y[i])
        for i in range(len(a)):
            words = nltk.word_tokenize(a[i])
            words = [ps.stem(word) for word in words if word not in set(stopwords.words('english'))]
            a[i] = ' '.join(words)  
        keys.append(a)

    df1 = pd.DataFrame(keys, columns=['keys'])

    answers=[]
    for i in range(len(x)):
        a=nltk.sent_tokenize(x[i])
        for i in range(len(a)):
            words = nltk.word_tokenize(a[i])
            words = [ps.stem(word) for word in words if word not in set(stopwords.words('english'))]
            a[i] = ' '.join(words)  
        answers.append(a)

    df2 = pd.DataFrame(answers, columns=['answers'])


    def calcut(a,b):
        ss=cosine_similarity(a,b)
        ss=np.diag(ss)
        return ss

    k_cv = cv.fit_transform(df1['keys']).toarray()
    a_cv = cv.transform(df2['answers']).toarray()


    k_tfidf = tfidf.fit_transform(df1['keys']).toarray()
    a_tfidf = tfidf.transform(df2['answers']).toarray()

    sim_cv=calcut(k_cv,a_cv)
    sim_tfidf=calcut(k_tfidf,a_tfidf)

    sim_gon=sim_cv*sim_tfidf
    for i in range(len(answers)):
        if list(sim_gon)[i]==0:
            marks[i]=0


    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    k_sb1 = model.encode(df1['keys'])
    a_sb1 = model.encode(df2['answers'])

    sim_sb1=calcut(k_sb1,a_sb1)


    m1=SentenceTransformer('paraphrase-MiniLM-L6-v2')
    k_sb2 = model.encode(df1['keys'])
    a_sb2 = model.encode(df2['answers'])

    sim_sb2=calcut(k_sb2,a_sb2)


    m2=SentenceTransformer('paraphrase-distilroberta-base-v1')
    k_sb3 = model.encode(df1['keys'])
    a_sb3 = model.encode(df2['answers'])

    sim_sb3=calcut(k_sb3,a_sb3)


    BERT_MODEL = "https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/4" 
    PREPROCESS_MODEL = "https://tfhub.dev/tensorflow/bert_zh_preprocess/3"

    preprocess = hub.load(PREPROCESS_MODEL)
    bert = hub.load(BERT_MODEL)

    inputs = preprocess(df2['answers'])
    outputs = bert(inputs)
    a_b1=outputs["pooled_output"]
    inputs = preprocess(df1['keys'])
    outputs = bert(inputs)
    k_b1=outputs["pooled_output"]

    sim_b1=calcut(k_b1,a_b1)



    BERT_MODEL = "https://tfhub.dev/google/experts/bert/wiki_books/2" 
    PREPROCESS_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

    preprocess = hub.load(PREPROCESS_MODEL)
    bert = hub.load(BERT_MODEL)

    inputs = preprocess(df2['answers'])
    outputs = bert(inputs)
    a_b2=outputs["pooled_output"]
    inputs = preprocess(df1['keys'])
    outputs = bert(inputs)
    k_b2=outputs["pooled_output"]

    sim_b2=calcut(k_b2,a_b2)



    def scor_fun(a,b,c,d,e,f,g):
        sim_sb=(a+b+c)/3
        sim_b=(d+e)/2
        sim=sim_b*sim_sb
        fin=sim*10-f*0.5
        fin=fin*g
        return list(fin)

    global fd
    global length
    final=scor_fun(sim_sb1,sim_sb2,sim_sb3,sim_b1,sim_b2,b,marks)
    final=[round(x,2) for x in final]
    global s
    s=round(sum(final),2)
    global tot_m
    tot_m=10*len(x1)
    fd=[]
    for i in range(len(x1)):
        n=[x1[i],final[i]]
        fd.append(n)
    print(fd)
    length=len(fd)
    return redirect('/result1')


@app.route('/result1')
def result1():
    return render_template('basicex.html',data=fd,length=length,total=s,tot_m=tot_m)

if __name__ == '__main__':
    app.run(debug=True)



















