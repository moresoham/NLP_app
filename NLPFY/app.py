from flask import Flask, url_for, request, render_template,jsonify,send_file
from flask_bootstrap import Bootstrap

import json
import spacy
nlp = spacy.load('en')
from textblob import TextBlob

# app initialize
app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/analyse', methods=['GET','POST'])
def analyse():
	if request.method=='POST':
		textip = request.form['rawtext']
		rawtext = textip.rstrip()
		# analysis
		docx = nlp(rawtext)

		# tokenization
		custom_tokens = [token.text for token in docx]

		# wordinfo
		custom_word_info = [(token.text, token.lemma_,token.shape_,token.is_alpha,token.is_stop) for token in docx]

		# pos and Entities
		custom_pos = [(token.text, token.tag_,token.pos_, token.dep_) for token in docx]
		custom_ent = [(entity.text, entity.label_) for entity in docx.ents]

		# sentiment 
		blob = TextBlob(rawtext)
		blob_sentiment = blob.sentiment.polarity
		blob_subjectivity = blob.sentiment.subjectivity

		result_json = json.dumps(custom_word_info,sort_keys = False, indent=2)


	return render_template('index.html', custom_tokens = custom_tokens,
							 custom_word_info = custom_word_info,
							 custom_pos = custom_pos,
							 custom_ent = custom_ent,
							 blob_sentiment = blob_sentiment,
							 blob_subjectivity = blob_subjectivity,
							 result_json=result_json,
							 rawtext = rawtext)


if __name__ == '__main__':
	app.run(debug=True)