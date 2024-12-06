library(tidyverse)
library(naniar)
library(gridExtra)
library(caret)
library(corrplot)
library(rpart.plot)
library(randomForest)

library(pROC)
library(GGally)

# Read the data
data <- read_csv("C:/Users/jimbo/Desktop/Computer Science Louisville/Fall2024/Data Mining/Project 2/Datasets/train.csv")

# Save the original Diabetes column
original_diabetes <- data$Diabetes

# Replacing Missing Values
missing_summary_pre <- miss_var_summary(data)
vis_miss(data) + ggtitle("Missing Data Pattern (Before Imputation)")

get_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

data <- as.data.frame(lapply(data, function(x) ifelse(is.na(x), get_mode(x), x)))
missing_summary_post <- any(is.na(data)) # Should be FALSE
vis_miss(data) + ggtitle("Missing Data Pattern (After Imputation)")

# Removing Outliers
check_categorical_outliers <- function(x, threshold = 0.01) {
  freq <- table(x) / length(x)
  rare_categories <- names(freq[freq < threshold])
  if (length(rare_categories) > 0) {
    cat("Rare categories found:", paste(rare_categories, collapse = ", "), "\n")
  } else {
    cat("No rare categories found.\n")
  }
}

categorical_cols <- sapply(data, is.factor) | sapply(data, is.character)
for (col in names(data)[categorical_cols]) {
  cat("Checking", col, ":\n")
  check_categorical_outliers(data[[col]])
  cat("\n")
}

detect_outliers_iqr <- function(x) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  return(x < lower_bound | x > upper_bound)
}

numerical_cols <- sapply(data, is.numeric)
for (col in names(data)[numerical_cols]) {
  outliers <- detect_outliers_iqr(data[[col]])
  if (any(outliers, na.rm = TRUE)) {
    cat("Outliers detected in", col, "\n")
    
    p1 <- ggplot(data, aes(x = .data[[col]])) +
      geom_histogram(bins = 30, fill = "lightblue", color = "black") +
      ggtitle(paste("Distribution of", col, "\nBefore replacing outliers")) +
      theme_minimal()
    
    col_mean <- mean(data[[col]], na.rm = TRUE)
    
    data[[col]][outliers] <- col_mean
    cat("Outliers in", col, "have been replaced with mean:", col_mean, "\n\n")
    
    p2 <- ggplot(data, aes(x = .data[[col]])) +
      geom_histogram(bins = 30, fill = "lightgreen", color = "black") +
      ggtitle(paste("Distribution of", col, "\nAfter replacing outliers")) +
      theme_minimal()
    
    grid.arrange(p1, p2, ncol = 2)
  } else {
    cat("No outliers detected in", col, "\n\n")
  }
}

# Correlation matrix before encoding (including categorical variables)
data_for_corr <- data %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.factor), as.numeric))

cor_matrix_before <- cor(data_for_corr, use = "pairwise.complete.obs")

# Function to create a more readable correlation plot
plot_correlation <- function(cor_matrix, title) {
  corrplot(cor_matrix, method = "color", type = "upper", 
           col = colorRampPalette(c("blue", "white", "red"))(200),
           tl.col = "black", tl.srt = 45, tl.cex = 0.7,
           title = title, mar = c(0,0,2,0))
}

plot_correlation(cor_matrix_before, "Correlation Heatmap Before Encoding")

# Encoding categorical values
id_col <- "id"
data_no_id <- data[, !names(data) %in% id_col]

categorical_cols <- names(data_no_id)[sapply(data_no_id, is.character) | sapply(data_no_id, is.factor)]
dummy_model <- dummyVars(~ ., data = data_no_id, levelsOnly = TRUE)
data_encoded <- as.data.frame(predict(dummy_model, newdata = data_no_id))
data_final <- cbind(data[, id_col, drop = FALSE], data_encoded)

cor_matrix_after_encoding <- cor(data_final[sapply(data_final, is.numeric)])
plot_correlation(cor_matrix_after_encoding, "Correlation Heatmap After Encoding")


# Normalization
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Function to plot distributions before and after normalization
plot_distributions <- function(data_before, data_after, feature) {
  p1 <- ggplot(data_before, aes(x = .data[[feature]])) +
    geom_density(fill = "lightblue", alpha = 0.7) +
    theme_minimal() +
    ggtitle(paste("Distribution of", feature, "Before Normalization"))
  
  p2 <- ggplot(data_after, aes(x = .data[[feature]])) +
    geom_density(fill = "lightgreen", alpha = 0.7) +
    theme_minimal() +
    ggtitle(paste("Distribution of", feature, "After Normalization"))
  
  grid.arrange(p1, p2, ncol = 2)
}

# Select a few representative numeric features for visualization
numeric_features <- names(data_final)[sapply(data_final, is.numeric)]
selected_features <- sample(numeric_features, min(5, length(numeric_features)))

# Apply normalization and plot distributions
data_normalized <- data_final %>%
  mutate(across(where(is.numeric), normalize))

for (feature in selected_features) {
  plot_distributions(data_final, data_normalized, feature)
}

# Add back the original Diabetes column
data_normalized$Diabetes <- original_diabetes
data <- data_normalized[-c(61:62)]

# Final correlation matrix
cor_matrix_final <- cor(data_normalized[sapply(data_normalized, is.numeric)])

# Plot final correlation matrix
plot_correlation(cor_matrix_final, "Final Correlation Heatmap")


#Dimensionality Reduction
features <- data_normalized %>% select(-Diabetes, -id)
target <- data_normalized$Diabetes

# 1. Principal Component Analysis (PCA)
pca_result <- prcomp(features, center = TRUE, scale. = TRUE)

# Determine number of components to retain 95% of variance
cumulative_var <- cumsum(pca_result$sdev^2 / sum(pca_result$sdev^2))
n_components <- which(cumulative_var >= 0.95)[1]

# Plot variance explained
var_explained <- data.frame(
  PC = 1:length(cumulative_var),
  CumulativeVariance = cumulative_var
)

ggplot(var_explained, aes(x = PC, y = CumulativeVariance)) +
  geom_line() +
  geom_point() +
  geom_hline(yintercept = 0.95, linetype = "dashed", color = "red") +
  geom_vline(xintercept = n_components, linetype = "dashed", color = "blue") +
  theme_minimal() +
  labs(title = "Cumulative Variance Explained by Principal Components",
       x = "Number of Principal Components",
       y = "Cumulative Proportion of Variance Explained")

# Transform the data
pca_data <- as.data.frame(predict(pca_result, newdata = features)[,1:n_components])

cat("Number of features before dimensionality reduction:", ncol(features), "\n")
cat("Number of principal components retaining 95% variance:", n_components, "\n")

# Visualize first two principal components
pca_plot_data <- data.frame(
  PC1 = pca_result$x[,1],
  PC2 = pca_result$x[,2],
  Diabetes = target
)

ggplot(pca_plot_data, aes(x = PC1, y = PC2, color = factor(Diabetes))) +
  geom_point(alpha = 0.6) +
  theme_minimal() +
  labs(title = "First Two Principal Components",
       color = "Diabetes")

# Create final datasets
pca_final <- cbind(id = data_normalized$id, pca_data, Diabetes = target)

original_data <- data_normalized %>% select(-id)
pca_data <- pca_final %>% select(-id)

# Function to train models
train_model <- function(data, method) {
  model <- train(Diabetes ~ ., data = data, method = method)
  return(model)
}


# Train models on PCA data
log_reg_pca <- train_model(pca_data, "glm")
dt_pca <- train_model(pca_data, "rpart")
rf_pca <- train_model(pca_data, "rf")

# Print model summaries

print("Logistic Regression - PCA Data:")
print(summary(log_reg_pca))

print("Decision Tree - PCA Data:")
print(dt_pca$finalModel)

print("Random Forest - PCA Data:")
print(rf_pca)

# Visualize Decision Tree (Original Data)
rpart.plot(dt_pca$finalModel, main="Decision Tree on Original Data")

# Feature Importance for Random Forest (Original Data)
importance_scores <- importance(rf_pca$finalModel)
feat_importance <- data.frame(
  Feature = rownames(importance_scores),
  Importance = importance_scores[, "MeanDecreaseGini"]
)
feat_importance <- feat_importance[order(-feat_importance$Importance), ]

ggplot(head(feat_importance, 10), aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 10 Feature Importance in Random Forest (Original Data)",
       x = "Features",
       y = "Importance")

# Save models for later use
saveRDS(log_reg_pca, "log_reg_pca.rds")
saveRDS(dt_pca, "dt_pca.rds")
saveRDS(rf_pca, "rf_pca.rds")

calculate_f_score <- function(actual, predicted) {
  cm <- confusionMatrix(factor(predicted), factor(actual))
  precision <- cm$byClass['Pos Pred Value']
  recall <- cm$byClass['Sensitivity']
  f_score <- 2 * (precision * recall) / (precision + recall)
  return(f_score)
}

# Function to train and evaluate models
train_and_evaluate <- function(data, method) {
  set.seed(123)  # for reproducibility
  
  # Create 10-fold cross-validation
  folds <- createFolds(data$Diabetes, k = 10)
  
  f_scores <- numeric(10)
  
  for (i in 1:10) {
    # Split data into training and testing sets
    train_data <- data[-folds[[i]], ]
    test_data <- data[folds[[i]], ]
    
    # Train the model
    model <- train(Diabetes ~ ., data = train_data, method = method)
    
    # Make predictions
    predictions <- predict(model, newdata = test_data)
    
    # Calculate F-score
    f_scores[i] <- calculate_f_score(test_data$Diabetes, predictions)
  }
  
  # Return mean F-score across all folds
  return(mean(f_scores))
}

# Calculate F-scores for each model
f_score_log_reg <- train_and_evaluate(pca_data, "glm")
f_score_dt <- train_and_evaluate(pca_data, "rpart")
f_score_rf <- train_and_evaluate(pca_data, "rf")

# Print results

cat("F-score for Logistic Regression:", f_score_log_reg, "\n")
cat("F-score for Decision Tree:", f_score_dt, "\n")
cat("F-score for Random Forest:", f_score_rf, "\n")

# Create a bar plot of F-scores
f_scores <- data.frame(
  Method = c("Logistic Regression", "Decision Tree", "Random Forest"),
  F_Score = c(f_score_log_reg, f_score_dt, f_score_rf)
)

ggplot(f_scores, aes(x = Method, y = F_Score)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme_minimal() +
  labs(title = "F-scores for Different Models",
       x = "Model",
       y = "F-score") +
  ylim(0, 1)
