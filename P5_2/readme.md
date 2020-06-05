# Data Exploration on Data Expo 2009: Airline on time data
## by Houssem Menhour


## Dataset

This dataset, Data Expo 2009: Airline on time data contains information on domestic flights in USA and their delays as tracked by The U.S. Department of Transportation's (DOT) Bureau of Transportation Statistics (BTS). Exploring this data can give us some insight on the patterns of such delays and their reasons.

The sample I used from the data contains 29 variables and 16,922,186 entries for a total of 3.6GB

The full data can be downloaded from [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HG7NV7). And more information can be found [here](https://www.transtats.bts.gov/Fields.asp?Table_ID=236).


The first thing I had to do before EDA was some wrangling in the following order:

* Load data as predefined data types.

* Remove columns with `NULL` values or unneeded information.

* Parse time as `datetime` and then extract the hour part of it.

* Remove cancelled and Diverted flights from the dataframe to be analysed later.

* Get new dataframe with some aggregated variables.

* Sin transform periodic time data for when needed.


## Summary of Findings


My main interest is finding out what patterns departure delay times `DepTime` follow and how do they relate to other variables.

After a long process of exploration I can summarize my findings with these bullet points:

* A total of 3.1% of all flights has been either cancelled or diverted.

* Delays seem to be equally distributed over time but do improve from year to year.

* While the correlation wasn't clear at first, the first days of each week or month are arguably the best for flying with less delays.

* A tragic event such as 9/11 has long lasting effects on the flight sector.

I focused on these same aspects in the presentation.


## Key Insights for Presentation


First, I gave a short overview of the data.

Then, I displayed the ratio of cancelled or diverted flights in the whole dataset, before jumping into other aspects of the data, starting by plotting the number of flights performed by each carrier.

Next, I used box plots and histograms with 4 of the features in the datasets: `DepDelay`, `ArrDelay`, `Distance`, `AirTime`. I used a logarithmic scale for the first two to make them easier to read. Otherwise, outliers will overshadow the distribution.

Following that, I showed the lack of clear correlation between these variables and others in the dataset, but continued to explore the idea further. In other words, by plotting delay distribution for each year, day of month, and day of week. which gave us better insight on the matter. Just out of curiosity, I also showed how `AirTime` increases with the `Distance`.

After that, I plotted the number of flights and the average delay over time. The first showed an interesting change in the data due to 9/11, while the second highlighted how delays increase significantly during holidays. Accompanying this is a figure showing how each carrier improved those delays over the years.

Finally, The last graph shows more details from the month of 9/11th 2001, to highlight the effect that traggic incident had on the whole industry.

