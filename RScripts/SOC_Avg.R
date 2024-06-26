library(ggplot2)
library(maps)
library(mapdata)
library(ggplot2)
library(rnaturalearth)
library(sf)

# Extract the year from the date
soc_data$year <- format(soc_data$date, "%Y")

# Aggregate data to get the average SOC by year
average_soc_bd_by_year <- soc_data %>%
  group_by(year) %>%
  summarize(mean_C = mean(C, na.rm = TRUE), mean_BD = mean(BD, na.rm = TRUE))

# Convert the year column back to numeric for plotting
average_soc_bd_by_year$year <- as.numeric(average_soc_bd_by_year$year)

# Plot the average SOC and BD by year with x-axis labels displayed vertically
ggplot(average_soc_bd_by_year, aes(x = year)) +
  geom_line(aes(y = mean_C, color = "Mean C")) +  # Line plot for SOC
  geom_point(aes(y = mean_C, color = "Mean C")) +  # Points on the line for SOC
  geom_line(aes(y = mean_BD, color = "Mean BD"), linetype = "dashed") +  # Line plot for BD
  geom_point(aes(y = mean_BD, color = "Mean BD"), shape = 17) +  # Points on the line for BD
  scale_y_continuous(
    name = "Mean SOC (C in % by mass)",
    sec.axis = sec_axis(~ ., name = "Mean Bulk Density (BD in g/cm3)")
  ) +
  scale_color_manual(values = c("Mean C" = "blue", "Mean BD" = "red")) +
  scale_x_continuous(breaks = average_soc_bd_by_year$year) +  # Ensure each year is shown
  theme_minimal() +
  labs(title = "Average SOC and Bulk Density by Year",
       x = "Year",
       color = "Legend") +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.title.x = element_text(margin = margin(t = 10)),
    axis.title.y = element_text(margin = margin(r = 10)),
    axis.title.y.right = element_text(margin = margin(l = 10)),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)
  )
