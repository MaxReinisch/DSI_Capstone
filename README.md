# Videogame Recommendation System
@author: Max Reinisch

## 0. Contents:
1. Summary
2. Outline
3. Data
4. Recommendation System


## 1. Summary:
### 1.1 Inspiration:
Games have been a part of my life as long as I can remember.  Some of my earliest experiences on a computer were playing point and click adventures like Spy Fox.  In middle school, I probably spent more time playing Halo and Call of Duty than doing homework.  In college, I became infactuated with indie games, playing as many as I could afford on my ramen budget, looking forward to the future when I could afford any game on my wishlist.  But now as an adult, I have found that I can't afford the time to experience the worlds I yearn to explore.  There will never be a time when I can play every game that I want, which makes the experience of playing a bad game all the more infuriating.  But there is a solution to this problem: a Machine Learning based Recommendation System!

### 1.2 Approach:
This system is to be trained using reviews accessed from Giantbomb.com's wonderful api.  The system takes two concurrent approaches: It evaluates what games are similar to each other, and which users have similar taste. Then when new users want a recommendation, the system will prompt them for a couple of games that they have played before.  Half of the system figures out which other users had similar opinions and the other half figures out which games are similar to initial reviews.  Then the system can recommend similar games that similar people liked, or a maybe recommend a game that other people liked that is explicitly different than the games that have already been played. 
