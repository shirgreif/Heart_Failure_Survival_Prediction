install.packages("shiny")
install.packages("plotly") # interactive
install.packages("corrplot")
install.packages("tidyverse")
install.packages("pROC")
install.packages("caret")
install.packages("effects")
install.packages("interplot")

library(ggthemes)
library(ggplot2)
library(leaflet)
library(sqldf)
library(moments)
library(plot3D)
library(scatterplot3d)
library(shiny)
library(plotly)
library(dplyr)
library(corrplot)
library(viridis)
library(hrbrthemes)
library(tidyr)
# new
library(tidyverse)
library(pROC)
library(caret)
library(effects)

set.seed(42)

# Import data from a file
df <- read.csv("prepared_data.csv",header=T)
df_train <- read.csv("df_train.csv", header = TRUE)
df_test <- read.csv("df_test.csv", header = TRUE)

# Null model
null_model<-glm(DEATH_EVENT~1,data=df_train,family='binomial'(link = "logit"))
summary(null_model) 
AIC(null_model)
BIC(null_model)

##
# Load the interplot package
library(interplot)

# Fit the full logistic regression model
df$anaemia <- factor(df$anaemia)
df$diabetes <- factor(df$diabetes)
df$high_blood_pressure <- factor(df$high_blood_pressure)
df$smoking <- factor(df$smoking)
df$sex <- factor(df$sex)

full_model <- glm(DEATH_EVENT ~ ejection_fraction*anaemia + creatinine_phosphokinase + serum_creatinine*diabetes + ejection_fraction*high_blood_pressure + platelets +  
                    sex + smoking + time,
                  data = df,
                  family = binomial(link = "logit"))

# Create interaction plot - anaemia and ejection_fraction

interplot(full_model, var1 = "ejection_fraction", var2 = "anaemia",facet_labs = c("1 = Anemia"), esize = 1, plot = TRUE) +
  xlab("Anaemia") +
  ylab("Estimated Coefficient for ejection_fraction") +
  ggtitle("Interaction between anaemia and ejection_fraction") +
  geom_point(size = 2, color = "red") +
  theme(plot.title = element_text(face = "bold"))

# Create interaction plot - diabetes and serum_creatinine

interplot(full_model, var1 = "serum_creatinine", var2 = "diabetes",facet_labs = c("1 = Diabetes"), esize = 1, plot = TRUE) +
  xlab("Diabetes") +
  ylab("Estimated Coefficient for serum_creatinine") +
  ggtitle("Interaction between diabetes and serum_creatinine") +
  geom_point(size = 2, color = "red") +
  theme(plot.title = element_text(face = "bold"))

# Create interaction plot - high_blood_pressure and ejection_fraction

interplot(full_model, var1 = "ejection_fraction", var2 = "high_blood_pressure",facet_labs = c("1 = high blood pressure"), esize = 1, plot = TRUE) +
  xlab("High blood pressure") +
  ylab("Estimated Coefficient for ejection_fraction") +
  ggtitle("Interaction between high_blood_pressure and ejection_fraction") +
  geom_point(size = 2, color = "red") +
  theme(plot.title = element_text(face = "bold"))

##
# Full model
full_model <- glm(DEATH_EVENT ~ ejection_fraction*anaemia + creatinine_phosphokinase + serum_creatinine*diabetes + ejection_fraction*high_blood_pressure + platelets +  
                    sex + smoking + time,
                  data = df_train,
                  family = binomial(link = "logit"))
formula(full_model)
summary(full_model)
AIC(full_model)
BIC(full_model)

coefficients <- coef(full_model)
# Construct the formula string with coefficient estimates
formula_string <- paste("DEATH_EVENT ~", paste(names(coefficients), "*", round(coefficients, digits = 3), collapse = " + "))
# Print the formula
print(formula_string)

# Step methods to select the best model

# Step wise model
step_model <- step(null_model, 
                   scope = list(lower = null_model,
                                upper = full_model),
                   direction = "both",
                   data = df_train)
summary(step_model)
AIC(step_model)
BIC(step_model)
formula(step_model)

# Foward model
foward_model <- step(null_model, 
                     scope = list(lower = null_model,
                                  upper = full_model),
                     direction = "forward",
                     data = df_train)
summary(foward_model)
AIC(foward_model)
BIC(foward_model)
formula(foward_model)

# Backwards model
back_model <- step(full_model, 
                   scope = ~1,
                   direction = "backward",
                   data = df_train)
summary(back_model)
AIC(back_model)
BIC(back_model)
formula(back_model)

# Final models
formula(step_model)
formula(foward_model)
formula(back_model)

# Cosen model - foward_model
chosen <-  foward_model
summary(chosen)

# Obtain the model equetions 

#  coefficients of the model
coefficients <- coef(chosen)

# Equation of the probability (phi) as a function of the explanatory variables
phi_equation <- paste("Phi =", round(exp(coefficients[1]), 4), "* exp(", round(coefficients[2], 4), "* ejection_fraction +", round(coefficients[3], 4), "* serum_creatinine +", round(coefficients[4], 4), "* time) / (1 +", round(exp(coefficients[1]), 4), "* exp(", round(coefficients[2], 4), "* ejection_fraction +", round(coefficients[3], 4), "* serum_creatinine +", round(coefficients[4], 4), "* time))")
# Print the equation
print(phi_equation)

# Equation of the log(odds)
# Construct the log odds equation
log_odds_equation <- paste("Log(Odds) =", 
                           round(coefficients[1], 4), "+", 
                           round(coefficients[2], 4), "* time +", 
                           round(coefficients[3], 4), "* serum_creatinine +", 
                           round(coefficients[4], 4), "* ejection_fraction")

# Print the log odds equation
print(log_odds_equation)
#######
# Assuming you have already trained your logistic regression model 'full_model'

# Step 1: Generate predicted probabilities on the test dataset
predicted_probs <- predict(chosen, newdata = df_test, type = "response")

# Roc data 
roc_data <- data.frame(
  p0.4 = ifelse(predicted_probs > 0.4, 1, 0),
  p0.5 = ifelse(predicted_probs > 0.5, 1, 0),
  p0.6 = ifelse(predicted_probs > 0.6, 1, 0),
  p0.65 = ifelse(predicted_probs > 0.65, 1, 0),  
  p0.7 = ifelse(predicted_probs > 0.7, 1, 0),
  p0.75 = ifelse(predicted_probs > 0.75, 1, 0),
  p0.85 = ifelse(predicted_probs > 0.85, 1, 0),
  p0.9 = ifelse(predicted_probs > 0.9, 1, 0))
# True values for test
actual <- df_test$DEATH_EVENT

# true positive (hit) rate
tpr <- function(pred, actual) {
  res <- data.frame(pred, actual)
  sum(res$actual == 1 & res$pred == 1) / sum(actual == 1)
}
# false positive rate
fpr <- function(pred, actual) {
  res <- data.frame(pred, actual)
  sum(res$actual == 0 & res$pred == 1) / sum(actual == 0)
}
# Get FPR and TPR for each threshold
roc_data <- roc_data %>% 
  gather(key = 'Threshold', value = 'pred') %>% 
  group_by(Threshold) %>%
  summarize(TPR = tpr(pred, actual = actual),  # Change 'predicted_probs' to 'pred'
            FPR = fpr(pred, actual = actual))  # Change 'predicted_probs' to 'pred'

# Calculate Euclidean distance
roc_data <- roc_data %>%
  mutate(Distance = sqrt((1 - TPR)^2 + FPR^2))

# Find threshold corresponding to maximum distance
optimal_threshold <- roc_data$Threshold[which.min(roc_data$Distance)]

# Create the plot with plot_ly - Threshold & TPR & FPR
fig2 <- plot_ly(data = roc_data, x = ~Threshold) %>%
  add_trace(y = ~FPR, mode = 'lines', name = 'False Positive Rate', type = 'scatter') %>%
  add_trace(y = ~TPR, mode = 'lines', name = 'True Positive Rate', type = 'scatter') %>%
  layout(title = list(text = "<b> TPR and FPR at every threshold", y = 0.975, x = 0.45))
fig2 <- fig2 %>% layout(legend = list(title = list(text = '<b> Rate </b>')))
fig2 <- fig2 %>% layout(yaxis = list(title = '<b> Rate </b>'))
fig2 <- fig2 %>% layout(xaxis = list(title = '<b> Threshold </b>'))
fig2

#
# Print optimal threshold
print(paste("Optimal Threshold:", optimal_threshold))
