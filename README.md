## NLP of musical theater songs

I wanted to see what further insights I could gather using NLP and hopefully find some overarching or changing topics that could feed into a recommendation system for musical theater songs. 

I scraped by data from a basic html lyric site, hand selecting muscials that were tony winners and/or ran for more than three years on broadway.

Using each song as a document I used TF-IDF and NMF to topic model. I then used vader to peform sentiment analysis. And eventually built a recommedation system for songs, wiht additional music data from the spotify API (asscessible through a flask app)

