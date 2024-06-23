library(dplyr)
library(ggplot2)
library(tidyr)

# Load SOC field samples excel into dataframe
# Remove duplicates
# Plot no. of samples per year
# Plot Carbon distribution
# Plot Bulk density distribution

soc_data <- read.csv('C:\\Mewara-SOC-modelling-in-South-Africa\\DataProcessing\\soc_gdf.csv')
soc_data <- soc_data %>% distinct()
soc_data$C[soc_data$C == 'NA'] <- NA
soc_data$BD[soc_data$BD == 'NA'] <- NA
soc_data <- soc_data %>% filter(soc_data$C <= 100)

soc_data <- soc_data %>%
            mutate(C_range = cut(C, breaks = c(-Inf, 0.5, 1, 2, 3, 4, Inf), 
                       labels = c("<0.5", "0.5-1", "1-2", "2-3", "3-4", ">4")))

missing_C_count <- sum(is.na(soc_data$C))
missing_C_count
missing_BD_count <- sum(is.na(soc_data$BD))
missing_BD_count

soc_data_yearly <- soc_data %>% 
                  group_by(Year) %>%
                  summarise(count = n())

yearly_samples <- ggplot(soc_data_yearly, aes(x=Year, y = count)) +
                  geom_bar(stat = 'identity', fill = 'skyblue') +
                  labs(title = 'Number of samples per year',
                       x = 'Year',
                       y = 'No. of Samples') +
                  scale_x_continuous(breaks = soc_data_yearly$Year) +
                  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

yearly_samples

# Plot for Soil Organic Carbon (SOC)
carbon_dist <- ggplot(soc_data, aes(x = C)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 100) +
  labs(
    title = "Distribution of Soil Organic Carbon (SOC)",
    x = "Soil Organic Carbon (% by mass)",
    y = "Frequency"
  ) +
  scale_x_continuous(
    breaks = seq(0, 20, by = 0.5),
    limits = c(0, 20) 
  ) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

carbon_dist

soc_data$BD <- as.numeric(as.character(soc_data$BD))

# Plot for Soil Organic Carbon (SOC)
bd_dist <- ggplot(soc_data, aes(x = BD)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 50) +
  labs(
    title = "Distribution of Soil Bulk Density",
    x = "Bulk Density (g/cm3)",
    y = "Frequency"
  ) +
  scale_x_continuous(
    breaks = seq(0, 5, by = 0.2),
    limits = c(0, 5) 
  ) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

bd_dist




