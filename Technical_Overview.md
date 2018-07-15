# Videogame Recommendation System
## A content based recommender system for playstation 4 games
### by Max Reinisch

This project builds a content-filter recommendation system using game metadata retrieved from [Giant Bomb's api](https://www.giantbomb.com/api/).  My original plan was build a collaborative filter by scraping review data from [Metacritic](http://www.metacritic.com/game), however the data was not robust enough to develop a reasonably successful model.  I then discovered Giant Bomb's api, and decided to use their user review data.  Unfortunately, the data I retrieved from them was not particularly robust either, however I did discover that they had a lot of game metadata that would be handy for a content filter.  

My final product is a content based recommender system that suggests similar games from a set of over 1,000 titles.  A link to the functioning flask application will be provided [here](#) once it is deployed. 

## Outline
- [Inspiration](#inspiration)
- [Overview](#overview)
- [Tools](#tools)
- [Dataset](#data)
- [Feature Extraction/Engineering](#features)
- [Building](#build)
- [Conclusion](#conclusion)


<a id='inspiration'></a>
## Inspiration:
Games have been a part of my life as long as I can remember.  Some of my earliest experiences on a computer were playing point and click adventures like Spy Fox.  In middle school, I probably spent more time playing Halo and Call of Duty than doing homework.  In college, I became infactuated with indie games, playing as many as I could afford on my ramen budget, looking forward to the future when I could afford any game on my wishlist.  But now as an adult, I have found that I can't afford the time to experience the worlds I yearn to explore.  There will never be a time when I can play every game that I want, which makes the experience of playing a bad game all the more infuriating.  But there is a solution to this problem: Recommendation Systems!

<a id='intro'></a>
## Overview:
### Problem statement:
My goal is to create a web app with a simple UI.  Given one or more games, the app will return a list of 10 "similar" games.  Since I have not played every PS4 game, there is no way for me to know if all 10 games are similar, so I my goal is that for every game that I enter, at least 1 game on the list will be a similar game that I have played.  

### Summary:
This system was trained using game data requested from Giantbomb.com's api.  The system uses features such as game concepts, developers, publishers, themes, and genres to create a binary vector for each game.  Similar games would have more overlap in their binary vectors:

#### Two Similar Games:
![alt text](figures/venn_TombRaider_Uncharted4.png "Two Similar Games")

#### Two Different Games:
![alt text](figures/venn_StreetFighterV_RocketLeague.png "Two Different Games")


The cosine distance is calculated between every game and is saved in a pairwise table.  I then wrote a python class that interfaces with this table, allowing users to easily search one or more games at a time to get an interesting mix of results.  Finally, I wrote a web application which users can easily use.  


<a id='tools'></a>
## Tools:

* Python
* Jupyter Notebooks
* NumPy
* Pandas
* SciKit-Learn
* MatPlotLib and Seaborn
* Flask
* HTML and Bootstrap
* [Docker image on an AWS EC2 instance](https://aws.amazon.com/) 
* [GitHub.com](https://github.com/)
* [Giant Bomb's API](https://www.giantbomb.com/api/)
* [Heroku](https://www.heroku.com/)

<a id='data'></a>
## Data:
You can find the notebook that requested the data [here](GB_API_Scrape/01_Getting_Data.ipynb).
You can find the documentation for accessing Giant Bomb's API [here](https://www.giantbomb.com/api/documentation#toc-0-16).

All I had to do to use the API was to get an API key.  Once I had that, I first scraped the `name` and `guid` fields for all Playstation games.  When requesting from the `games` endpoint, each response had up to 100 games, and most of the meta-data was not included. I iteratively saved each response to a file, `games_list.json`, which I would use to request each game's reviews and metadata individually. 

There are a total of 2800 ps4 games, and just under 30,000 user_reviews in GB's database.  I ran into my first obstacle when I wanted to filter my search to games that had more than 5 user-reviews. Turns out that there was a bug in the API that returned 0 user_reviews for all games, so my filter returned 0 results. So with no way of knowing how many games would be useful, I requested all reviews for all games, only to discover that there were only about 100 users who had reviewed 5 or more games.  

![alt text](data:image/png;base64,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%0A "Not enough user reviews")


Realizing this was not enough to build a collaborative filter, I resorted to building a content filter using the game's metadata.  

I made one request for each of the 2800 games to get the metadata over a 14 hour period.  I iteratively saved each game's metadata a separate json file, identified by each game's unique `guid`.

<a id='features'></a>
## Features:
All my feature engineering happened in the EDA notebook [here](/02.2_Metadata_EDA.ipynb).

Having planned to do NLP on reviews for the collaborative filter, I was still hoping that I could incorperate NLP in the game descriptions.  However, once I retrieved the data, I discovered that the API response contained a field called `concepts` that contained the exact data that I wanted to extract using NLP.  I combined these `concepts` with the `genres`, `themes`, `developers`, and `publishers` features.  Each of these fields were lists of features, so I used `CountVectorizor` on each list to extract all features (so in the end I did sort of use NLP).  

At this point, I had a pandas dataframe with over 5000 features, and 1700 games.  I decided to drop a number of concepts manually, as I did not deem them relevant.  Such concepts included:
* `PAX` or `E3`, because they only state that the game had a presence at a trade show
* `Digital Distribution`, meaning that the game was downloadable
* `Trophies` and `Achievements`, because they don't relate to the actual gameplay
* `WASD Movement` and anything with `Steam`, as these features relate to computer gameplay, not playstation

I then dropped every game with less than 6 features, leaving me with just over 1000 games.  

Next, I ran 2 rounds of algorithmic feature reduction. First I ran a `VarianceThreshold` of .005, leaving me with about 1200 features.
![alt text](figures/num_feats.png "Number of features per game")

Then I ran `TruncatedSVD` to reduce my 1200 features into 300 components.  These 300 components preserved 90% of the variance from the original features.  

![alt text](figures/total_variance.png "Cumulative Sum of Variance per Component")

The components are interesting to look at, as they are loaded with covariant features.  For any component, you can take a look at the high variance features to get an idea for what the component represents.  

For example, one component had:
* concepts_boss fight
* concepts_bosses
* concepts_health
* concepts_death
* concepts_melee

This component was very prominent in a game like Dark Souls.    

<a id='build'></a>
## Build:
I built the model in [this notebook](/03_Build.ipynb)

After running all of the transformations, the final step was to compute a pairwise cosine distance matrix.  The cosine distance between two vectors reflects how similar they are to each other. The cosine distance between 2 vectors in the same direction is 0.  The larger the cosine distance, the more different the vectors are. Therefore, for any given game, if we order the cosine distances from that game to every other game from smalles to largest, we will have a list of games ordered from most to least similar.  

#### Two Similar Games have a distance of 0.64:
![alt text](figures/venn_TombRaider_Uncharted4.png "Two Similar Games")

#### Two Different Games have a distance of 0.92: 
![alt text](figures/venn_StreetFighterV_RocketLeague.png "Two Different Games")

After having done this, I decided that searching for 1 game wasn't good enough. I created a class that will take in a list of games, union their feature vector, transform the features into components, and then find the cosine distance to every other game.  The nice thing is that all you have to do is call `CustomSearch([game1, game2,...]).searchSimilarGames()`.

<a id='conclusion'></a>
## Conclusion:

In conclusion, I created a matrix of cosine simularities that can be used to search for the 10 most similar games.  I then created a class to simplify the process of searching, and used html and flask to create a web app.  Through all my tests, I have found that all searches result in a game that I know to be pretty similar. 