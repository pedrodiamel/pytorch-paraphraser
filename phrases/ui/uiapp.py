
from flask import Flask, url_for, request,render_template, jsonify, send_file
from ..webapp import app, net


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=['GET', 'POST'])
def analyze():
    phrase=''
    temp=0.75
    if request.method == 'POST':
       
        #analyzes and generate
        #read text
        sentence = request.form['txtsentence']
        # TODO January 27, 2019: paraphrase( sentence )
        # code here
        phrase = sentence
        if 'btngreedy' in request.form:
            phrase, score = net( phrase ) 
            phrase = ' '.join(phrase) 

        elif 'btnsample' in request.form:
            phrase += ' --> sample '
        
    return render_template("index.html", phrase=phrase, temp=temp)

# API FOR PHRASES GENERATE
@app.route('/api/generate/<string:sentence>',methods=['GET'])
def api_tokens(sentence):
	# Analysis
    # TODO January 27, 2019: paraphrase( sentence )
    # code here
    
    phrase = sentence 
    phrase, score = net( phrase )
    phrase = ' '.join(phrase)     
    
    return phrase