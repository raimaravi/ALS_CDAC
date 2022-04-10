from urllib.robotparser import RequestRate
from flask import Flask, redirect, render_template, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.utils import secure_filename
import os.path
from RNN import RNN

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI']="sqlite:///ALS.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
app.config['UPLOAD_FOLDER']='./input/audio_test/'
app.config['MAX_CONTENT_LENGTH']=1024*1024
db=SQLAlchemy(app)

class ALS(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    name=db.Column(db.String(50), nullable=False)
    mob=db.Column(db.String(50), nullable=False)
    file_name=db.Column(db.String(50))
    result=db.Column(db.String(50))
    date_created=db.Column(db.DateTime, default=datetime.now())
    
    def __repr__(self) -> str:
        return f"{self.sno} - {self.name} - {self.mob} - {self.date_created}"

# this is the home page
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/home")
def home2():
    return render_template("home.html")

# new record in creted from this page
@app.route("/check")
def check():
    return render_template("check.html")

# this is the about page
@app.route("/about")
def about():
    return render_template("about.html")

# thsi will contain all the data
@app.route("/alldata", methods=['GET', 'POST'])
def alldata():
    if request.method=="POST":
        name=request.form['name']
        mob=request.form['mob']
        f=request.files['afile']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        als=ALS(name=name, mob=mob, file_name=f.filename)
        db.session.add(als)
        db.session.commit()
    allals = ALS.query.all()
    return render_template("alldata.html", allals=allals)    

# this is used to delete record
@app.route("/delete/<int:sno>")
def delete(sno):
    dals=ALS.query.filter_by(sno=sno).first()
    print(dals.file_name)
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], dals.file_name))
    db.session.delete(dals)
    db.session.commit()
    return redirect("/alldata")

# This page shows the customer record
@app.route("/View/<int:sno>")
def view(sno):
    dals=ALS.query.filter_by(sno=sno).first()
    rnn=RNN()
    rnn.extract_mfcc(dals.file_name)
    rnn.save_graph(dals.file_name)
    res = rnn.predict()
    if res==0:
        dals.result="Don't worry you are Not suffering from ALS"
    else:
        dals.result="You are detected with ALS. Please visit doctor and start your treament."
    return render_template("report.html", dals=dals)


if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)