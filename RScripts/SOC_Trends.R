# This is a simple code demo for visualising scientific interest in key terms
# (adapted by Andy Cunliffe from https://twitter.com/EShekhova/status/1772925867603165413?s=09, who adapted it from elsewhere...)
# This can be really helpful for developing quantitative summaries to evidence 
# statements like "there is rising interest in topic X"
# the europepmc package calls on the Europe PMC database, an archive of > 44 Million
# scientific outputs. While this is not complete, it’s much larger than PubMed is 
# has enough for this application. 

# Step 1 - Load packages ------------------------------------------------

# install.packages("europepmc")
# install.packages("tidyverse")
# install.packages("viridis")

library(europepmc)  # This package is published by https://ropensci.org/, which is really high quality and has a thorough review process.
library(tidyverse)
library(viridis)

# Step 2 - Specify keywords ------------------------------------------------
# (e.g., categories of diet, land management practices, data sources, etc.)

keyword1 <- '"soil organic carbon" OR 
"soil carbon" OR
"soil organic matter" OR
"soil carbon stocks" OR
'
#keyword2 <- '("soil organic carbon" OR "soil carbon") AND "South Africa"'
#keyword3 <- '(("soil organic carbon" OR "soil carbon") AND ("Random Forest" OR 
#                                         "Machine Learning" OR 
#                                         "Deep Learning" OR
#                                         "Remote Sensing" OR
#                                         "Digital" OR
#                                         "Neural" OR
#                                         "Regression"))'
# keyword4 <- '("soil organic carbon" OR "soil carbon") AND ("Climate Change" OR 
#                                       "Land Use")'


# Step 3 - Fetch metrics ------------------------------------------------
# Use the europepmc package to query the Europe PubMed Central RESTful Web Service to 
# fetch the number of publications with specific keywords for each year. 
# Note that the search arguments can be much more nuanced through targeting particular 
# parts of source documents (e.g., titles, abstracts (via "abstract: my-keywords"),
# funders) and/or use of Boolean operators (e.g., and, or, etc.). 
# For more information see https://docs.ropensci.org/europepmc/.

trend_1 <- europepmc::epmc_hits_trend(query = keyword1,
                                      period = 2000:2023, synonym = TRUE)
trend_2 <- europepmc::epmc_hits_trend(query = keyword2,
                                      period = 2000:2023, synonym = TRUE)
trend_3 <- europepmc::epmc_hits_trend(query = keyword3,
                                      period = 2000:2023, synonym = TRUE)
trend_4 <- europepmc::epmc_hits_trend(query = keyword4,
                                      period = 2000:2023, synonym = TRUE)


# Step 4 - Combine data ------------------------------------------------

combined_data <- rbind(
  mutate(trend_1, keywords = keyword1),
  mutate(trend_2, keywords = keyword2),
  mutate(trend_3, keywords = keyword3),
  mutate(trend_4, keywords = keyword4)
)

# Step 5 - Visualise data ------------------------------------------------

combined_plot <- ggplot(combined_data, aes(x = factor(year), y = query_hits, fill = keywords)) +
  geom_col(width = 0.6, alpha = 0.9) +
  theme_minimal() +
  labs(x = "Year", y = "Number of Published Papers") +
  ggtitle("Scientific Interest in Studying SOC") +
  scale_fill_viridis_d() +
  facet_wrap(~keywords, ncol = 2) +
  theme(legend.position = "none",
        axis.text.x = element_text(size = 7, angle = 60, hjust = 1),
        plot.title = element_text(hjust = 0.5),
        legend.text = element_text(size = 15),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())

combined_plot # Display output