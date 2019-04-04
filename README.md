## NLP of musical theater songs

I wanted to see what further insights I could gather using NLP and hopefully find some overarching or changing topics that could feed into a recommendation system for musical theater songs. 

I [scraped](/scrape_musical_lyrics.ipynb) my data from a basic html lyric site, hand selecting muscials that were tony winners and/or ran for more than three years on broadway.

Using each song as a document, I used TF-IDF and NMF to [topic model](/topic_modeling_musicals.ipynb). I then used vader to peform [sentiment analysis](/sentiment_analysis_musicals.ipynb) and eventually built a recommedation system for songs, with additional music data from the spotify API ([beta flask recommendation app](/musical_recommender.py)).

See key findings in my [presetation.](/Musical Presentation.pdf)

