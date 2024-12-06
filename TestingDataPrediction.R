library(tidyverse)
library(naniar)
library(gridExtra)
library(caret)
library(GGally)
library(corrplot)
library(rplot)
library(rpart.plot)
library(randomForest)

# Read the data
data <- read_csv("C:/Users/jimbo/Desktop/Computer Science Louisville/Fall2024/Data Mining/Project 2/Datasets/test.csv")

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

# Final correlation matrix
cor_matrix_final <- cor(data_normalized[sapply(data_normalized, is.numeric)])

# Plot final correlation matrix
plot_correlation(cor_matrix_final, "Final Correlation Heatmap")


#Dimensionality Reduction
features <- data_normalized %>% select(-id)

# 1. Principal Component Analysis (PCA)
pca_result <- prcomp(features, center = TRUE, scale. = TRUE)

# Determine number of components to retain 95% of variance
cumulative_var <- cumsum(pca_result$sdev^2 / sum(pca_result$sdev^2))
n_components <- which(cumulative_var >= 0.97)[1]

# Plot variance explained
var_explained <- data.frame(
  PC = 1:length(cumulative_var),
  CumulativeVariance = cumulative_var
)

ggplot(var_explained, aes(x = PC, y = CumulativeVariance)) +
  geom_line() +
  geom_point() +
  geom_hline(yintercept = 0.97, linetype = "dashed", color = "red") +
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
  PC2 = pca_result$x[,2]
)

ggplot(pca_plot_data, aes(x = PC1, y = PC2)) +
  geom_point(alpha = 0.6) +
  theme_minimal() +
  labs(title = "First Two Principal Components")

# Create final datasets
pca_final <- cbind(id = data_normalized$id, pca_data)

original_data <- data_normalized %>% select(-id)
pca_data <- pca_final %>% select(-id)

log_reg_pca <- readRDS("log_reg_pca.rds")
dt_pca <- readRDS("dt_pca.rds")
rf_pca <- readRDS("rf_pca.rds")

# Function to make predictions using a model
predict_diabetes <- function(model, data) {
  predictions <- predict(model, newdata = data)
  return(predictions)
}

# Make predictions using each model
pca_data$Diabetes_LogReg <- predict_diabetes(log_reg_pca, pca_data)
pca_data$Diabetes_DecisionTree <- predict_diabetes(dt_pca, pca_data)
pca_data$Diabetes_RandomForest <- predict_diabetes(rf_pca, pca_data)

pca_data <- cbind(id = pca_final$id, pca_data)
final_data <- pca_data[c(1,43)]
colnames(final_data) <- c("id", "Diabetes")

# Save the results
write_csv(final_data, "test_data_predictions.csv")




