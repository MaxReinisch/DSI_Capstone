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

### 1.3 Milestone 3 Report:
#### prompt:
#### Content and things to discuss are up to you, but we expect (at a minimum):
- Do you have data fully in hand and if not, what blockers are you facing?
- Have you done a full EDA on all of your data?
- Have you begun the modeling process? How accurate are your predictions so far?
- What blockers are you facing, including processing power, data acquisition, modeling difficulties, data cleaning, etc.? How can we help you overcome those challenges?
- Have you changed topics since your lightning talk? Since you submitted your Problem Statement and EDA? If so, do you have the necessary data in hand (and the requisite EDA completed) to continue moving forward?
- What is your timeline for the next week and a half? What do you have to get done versus what would you like to get done?

#### Answer:
1. I have all the data I need to complete the project.  Currently, I am limiting my project to just ps4 games, but in the future, I'd like to expand the project to other consoles.  My scraping notebook should be easy enough to modify to get games from other consoles, so I am not worried about obtaining more data in the future.

2. I have done plenty of EDA on the games' features/metadata.  I still plan to do some EDA on the NLP for the games' metadata.  I still have a lot of EDA to do for the reviews, because I realized I did not include the game that each review was for in my initial scrape, which I have since fixed.  However, I did manage to do some initial EDA to get a sense for how many relevant reviews exist (reviewers that have 5+ reviews, but I don't know how many games have 5+ reviews)

3. I have built a matrix of cosine similarities for games using only their "concepts" feature as a prototype.  I am poised to include the rest of the features (minus the actual NLP) in the pipeline, but at the current time I have not included them. I also need to get this done for the reviews to build the collaborative filter.

4. I don't see any blocks at the moment.  

5. I have not changed my topic since the problem statement and EDA.  

6. My timeline goes as follows:
- Finish the pipeline for the content filter on all features by July 4th (soon, but also an easy goal)
- I should finish review EDA and begin the prototype collaborative filter by thursday, and finish the pipeline by friday
- Weekend: NLP time.  The whole weekend I want to use to play with the NLP preprocessing to see how it affects the similarities.  Instead of just using numeric scores, now the individual people will have a higher similarity with each other based on how the describe the games they review.  I likely will hold off on doing this for the actual game metadata, because I believe that that will lead to games being rated as similar just because the authors who wrote the descriptions are the same.  
- Early next week: Monday should be used to get test data from my Metacritic scrape and use it to evaluate the model.  I can then use Tuesday to play with the hyperparameters and see how the test results pan out.  I honestly have no idea how much insight this will give me on evaluating the model, but the process doesn't sound too hard so its probably worth a shot.
- Rest of the week: Home stretch. At this point I should have finalized my model, and now I am spending wednesday-friday setting up a flask application to host my project.  