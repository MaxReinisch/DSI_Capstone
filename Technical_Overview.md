# Videogame Recommendation System
@author: Max Reinisch


## 1. Summary:
### 1.1 Inspiration:
Games have been a part of my life as long as I can remember.  Some of my earliest experiences on a computer were playing point and click adventures like Spy Fox.  In middle school, I probably spent more time playing Halo and Call of Duty than doing homework.  In college, I became infactuated with indie games, playing as many as I could afford on my ramen budget, looking forward to the future when I could afford any game on my wishlist.  But now as an adult, I have found that I can't afford the time to experience the worlds I yearn to explore.  There will never be a time when I can play every game that I want, which makes the experience of playing a bad game all the more infuriating.  But there is a solution to this problem: Recommendation Systems!

### 1.2 Approach:
This system was trained using game data requested from Giantbomb.com's api.  The system uses features such as game concepts, developers, publishers, themes, and genres to create a binary vector for each game.  The cosine distance is calculated between every game and is saved in a pairwise table.  With this table, one simply has to search a game to find the cosine distance from every other game.  Smaller distances mean more similar games.  