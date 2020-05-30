# Data Exploration on Kaggle's House Prices Dataset
## by Houssem Menhour


## Dataset

> Provide basic information about your dataset in this section. If you selected your own dataset, make sure you note the source of your data and summarize any data wrangling steps that you performed before you started your exploration.

This is a popular dataset from Kaggle with historical sale prices of houses and their features or characteristics. It is meant for practicing regression techniques and predicting sale prices, a task that requires EDA first.

This dataset comes in 1460 entries and 81 columns, 38 of which are numerical while 43 are categorical, and the latter can be split further to ordinal and nominal values.
Some of the categorical columns represent ordinal values, this mainly applies to variables of a quality related value, while others are nominal.

The full data and its description is available [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).


The first thing I had to do was some wrangling in the following order:

* Remap `MSSubClass` values to be categorical instead of numerical.

* Remove columns with too many missing values.

* Fill the rest of missing values with mean/mode values for numerical/categorical variables.

* Remap `object` columns in the DataFrame into `category` columns, taking into consideration ordinal values.

* Remove columns with sever imbalance in values (95% or more similar value).

* Log transform the rest of skewed numerical values.


## Summary of Findings

> Summarize all of your findings from your exploration here, whether you plan on bringing them into your explanatory presentation or not.

My main interest is finding out what features of a house decide its pricing. After a long process of exploration I came to the conclusion that the biggest factors contributing to the price can be summarized in 3 categories:

* **Age Features**: namely the year it was built, sold, and the difference between them.

* **Area Features**: such as lot area, and first floor area and garage area.

* **Quality Feature**: quality ratings for each part of the home (kitchen, exterior, basement ...) as well as their combined score.

That's why I focused on these same aspects in the presentation.


## Key Insights for Presentation

> Select one or two main threads from your exploration to polish up for your presentation. Note any changes in design from your exploration step here.

First, I started the presentation by showcasing the two main issues with the data: missing values, and skewness.

Then, I drew a heatmap of the correlations between the numerical values to better show the relation between each pair of them.

Next, I used box plots with 4 of the main Quality related features to show how much they contribute to the price. A scatter plot with the overall quality score agrees with the result as well. I did the similar thing with numerical data, this time signifying Area and Age information, and using scatter plots with regression lines.

Finally, I created one graph to summarize most of the findings by including all of the Age, Quality and Area information at once.

