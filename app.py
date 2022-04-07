from flask import Flask, redirect, render_template, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.utils import secure_filename
import os.path
import RNN

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI']="sqlite:///ALS.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
app.config['UPLOAD_FOLDER']='./input/audio_test'
app.config['MAX_CONTENT_LENGTH']=1024*1024
db=SQLAlchemy(app)

class ALS(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    fname=db.Column(db.String(50), nullable=False)
    lname=db.Column(db.String(50), nullable=False)
    date_created=db.Column(db.DateTime, default=datetime.utcnow())
    def __repr__(self) -> str:
        return f"{self.sno} - {self.fname} - {self.lname} - {self.date_created}"

# this is the home page
@app.route("/")
def home():
    return render_template("main.html")

# new record in creted from this page
@app.route("/new")
def new():
    return render_template("new.html")

# this is the about page
@app.route("/about")
def about():
    return render_template("about.html")

# thsi will contain all the data
@app.route("/alldata", methods=['GET', 'POST'])
def alldata():
    if request.method=="POST":
        fname=request.form['fname']
        lname=request.form['lname']
        f=request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(str(datetime.utcnow()))))
        als=ALS(fname=fname, lname=lname)
        db.session.add(als)
        db.session.commit()
        # print(request.form['inputGroupFile04'])
    allals = ALS.query.all()
    return render_template("alldata.html", allals=allals)

# This page is used to update record
@app.route("/update/<int:sno>", methods=['GET', 'POST'])
def update(sno):
    if request.method=="POST":
        fname=request.form['fname']
        lname=request.form['lname']
        f=request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(str(datetime.utcnow()))))
        als=ALS.query.filter_by(sno=sno).first()
        als.fname=fname
        als.lname=lname
        db.session.add(als)
        db.session.commit()
        return redirect("/alldata")
    dals=ALS.query.filter_by(sno=sno).first()
    return render_template("update.html", dals=dals)

# this is used to delete record
@app.route("/delete/<int:sno>")
def delete(sno):
    dals=ALS.query.filter_by(sno=sno).first()
    db.session.delete(dals)
    db.session.commit()
    return redirect("/alldata")

# This page shows the customer record
@app.route("/View/<int:sno>")
def view(sno):
    dals=ALS.query.filter_by(sno=sno).first()
    file_name = str(dals).split("-")[-1]
    print(file_name)
    return render_template("view.html")


if __name__=="__main__":
    app.run(debug=True)