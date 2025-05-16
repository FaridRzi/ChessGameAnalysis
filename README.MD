# Chess Game Analysis

This project uses the Chess.com public API to analyze historical games for a given player. It provides game summaries, opening performance, and statistical hypothesis testing — useful for both chess improvement and data analysis practice.

## Features
- Opening-level performance analysis
- Color-based win/loss stats
- Hypothesis testing with chi-square, z-tests, and Fisher's exact test

# How to run the scrip?
Initialize an instance of the class with your Chess.com username.

pgo = PlayerGamesOverview("YOUR_USERNAME")

Using the year and month, fetch your data from Chess.com

df = pgo.get_data(2025, 5)