
# musical recommender
# http://flask.pocoo.org/docs/quickstart/

import pickle
import numpy as np
from nltk.tokenize import word_tokenize

with open("musical_df2.pkl", "rb") as f:
	musical_df2 = pickle.load(f)

import gensim
gen_docs = [[w for w in word_tokenize(cell)] for cell in musical_df2.song_lyrics_pro]
dictionary = gensim.corpora.Dictionary(gen_docs)
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
tf_idf = gensim.models.TfidfModel(corpus)
sims = gensim.similarities.Similarity('/Users/noramay/desktop',tf_idf[corpus], num_features=len(dictionary))

from flask import Flask, render_template, request
app = Flask(__name__)
app.debug = True


@app.route('/', methods=['GET'])
def dropdown():
    musicals = musical_df2.song_titles
    return render_template('scrabble.html', colours=musicals)

@app.route('/api')
def song_rec():

	song_title = request.args.get('value')
	song_idx = np.where(musical_df2.song_titles == song_title)
	song_idx = song_idx[0][0]
	query_doc = [w for w in word_tokenize(musical_df2.song_lyrics_pro.loc[song_idx])]
	query_doc_bow = dictionary.doc2bow(query_doc)
	query_doc_tf_idf = tf_idf[query_doc_bow]
    
	musical = musical_df2.titles.loc[song_idx]
	idxs = np.where(musical_df2.titles == musical)

	mask1 = np.zeros(sims[query_doc_tf_idf].size, dtype=bool)
	mask1[idxs] = True

	km_cat = musical_df2.km_labels.loc[song_idx]
	idxs2 = np.where(musical_df2.km_labels != km_cat)
    
	mask2 = np.zeros(sims[query_doc_tf_idf].size, dtype=bool)
	mask2[idxs2] = True

	a = np.ma.array(sims[query_doc_tf_idf], mask=(mask1 | mask2))

	maxi = np.max(a)

	i = list(sims[query_doc_tf_idf]).index(maxi)
	my_ans = 'I recommend ' + str(musical_df2.song_titles.iloc[i]) + ' from ' +  str(musical_df2.titles.iloc[i])
	return my_ans

if __name__ == "__main__":
    app.run()

app.run(port=5000, debug=True)