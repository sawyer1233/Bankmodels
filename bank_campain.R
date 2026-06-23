#----------------------------------------------------------------------------------------
#                            MODEL KNN
#----------------------------------------------------------------------------------------


library(readr)
library(dplyr)
install.packages("fastDummies")

# Wczytanie danych z pliku CSV
bank <- read_delim("bank-full.csv", delim = ";")

# Usunięcie wierszy z brakującymi wartościami w wybranych kolumnach
bank <- bank %>%
  filter_at(
    c("age", "job", "marital", "education", "default", "balance", "housing",
      "loan", "contact", "day", "month", "duration", "campaign", "pdays",
      "previous", "poutcome", "y"),
    ~ !is.na(.)
  )

# Wybór kolumn numerycznych z wyłączeniem "day"
numeric_cols <- sapply(bank, is.numeric)
numeric_names <- names(bank)[numeric_cols]
numeric_names <- setdiff(numeric_names, "day")

# Normalizacja danych numerycznych do przedziału [0,1]
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

bank <- bank %>%
  mutate(across(all_of(numeric_names), normalize))

bank <- data.frame(bank)

# Wydzielenie etykiet klas (zmienna zależna)
bank_labels <- bank %>% select(y)
bank <- bank %>% select(-y)

# Zmiana zmiennych kategorycznych na zmienne zero-jedynkowe
library(fastDummies)
bank <- fastDummies::dummy_cols(
  .data = bank,
  select_columns = c("job", "marital", "education", "default", "housing", 
                     "loan", "contact", "day", "month", "poutcome"),
  remove_selected_columns = TRUE,
  remove_first_dummy = FALSE
)

# Podział danych na zbiór treningowy i testowy (75/25)
set.seed(1234)
sample_index <- sample(nrow(bank), round(nrow(bank) * .75), replace = FALSE)
bank_train <- bank[sample_index,]
bank_test <- bank[-sample_index,]
bank_train_labels <- as.factor(bank_labels[sample_index,])
bank_test_labels <- as.factor(bank_labels[-sample_index,])

library(class)

# KNN z k = 1
bank_pred1 <- knn(
  train = bank_train,
  test = bank_test,
  cl = bank_train_labels,
  k = 1
)

# Ocena dokładności klasyfikatora k = 1
library(caret)
confusionMatrix(bank_pred1, bank_test_labels)

# KNN z k = 5
bank_pred2 <- knn(
  train = bank_train,
  test = bank_test,
  cl = bank_train_labels,
  k = 5
)

# Ocena dokładności klasyfikatora k = 5
confusionMatrix(bank_pred2, bank_test_labels)

# Ponowne wczytanie danych i przygotowanie do trenowania modelu z użyciem caret
bank <- read_delim("bank-full.csv", delim = ";")

# Usunięcie braków danych
bank <- bank %>%
  filter_at(
    vars(age, job, marital, education, default, balance, housing,
         loan, contact, day, month, duration, campaign, pdays,
         previous, poutcome, y),
    ~ !is.na(.)
  )

# Konwersja zmiennej celu na typ factor
bank$y <- factor(bank$y, levels = c("no", "yes"))

# Podział danych na zbiór treningowy i testowy (75/25)
set.seed(1234)
sample_set <- createDataPartition(y = bank$y, p = 0.75, list = FALSE)
bank_train <- bank[sample_set, ]
bank_test  <- bank[-sample_set, ]

# Sprawdzenie rozkładu zmiennej celu w zbiorze testowym
table(bank_test$y)

# Definicja sposobu walidacji (2-krotna walidacja krzyżowa)
ctrl <- trainControl(method = "cv", number = 2, verboseIter = TRUE)

# Trenowanie modelu KNN z automatycznym doborem liczby sąsiadów
set.seed(1234)
bank_mod <- train(
  y ~ .,
  data = bank_train,
  metric = "Accuracy",
  method = "knn",
  trControl = ctrl
)

# Predykcja na zbiorze testowym
bank_pred <- predict(bank_mod, bank_test)

# Dopasowanie poziomów w predykcji i danych testowych
levels_union <- union(levels(bank_pred), levels(bank_test$y))
bank_pred <- factor(bank_pred, levels = levels_union)
bank_test$y <- factor(bank_test$y, levels = levels_union)

# Macierz pomyłek
confusionMatrix(bank_pred, bank_test$y)

# Predykcja prawdopodobieństw dla klasy "yes"
bank_pred_prob <- predict(bank_mod, bank_test, type = "prob")
positive_class <- "yes"

library(ROCR)

# Obliczenie i narysowanie krzywej ROC
roc_pred <- prediction(predictions = bank_pred_prob[[positive_class]], labels = bank_test$y)
roc_perf <- performance(roc_pred, measure = "tpr", x.measure = "fpr")

plot(roc_perf, main = "ROC Curve", col = "red", lwd = 3)
abline(a = 0, b = 1, lwd = 3, lty = 2, col = 1)

# Obliczenie pola pod krzywą ROC (AUC)
auc_perf <- performance(roc_pred, measure = "auc")
bank_auc <- unlist(slot(auc_perf,"y.values"))
bank_auc


#----------------------------------------------------------------------------------------
#                            MODEL BAYESA
#----------------------------------------------------------------------------------------

library(readr)
library(dplyr)
library(e1071)
library(caret)

# Wczytanie danych
bank <- read_delim("bank-full.csv", delim = ";")

# Konwersja zmiennej celu na typ factor
bank$y <- factor(bank$y, levels = c("no", "yes"))

# Podział danych na zbiór treningowy i testowy (75/25)
set.seed(1234)
sample_index <- sample(nrow(bank), round(nrow(bank) * .75), replace = FALSE)
bank_train <- bank[sample_index, ]
bank_test  <- bank[-sample_index, ]

# Trenowanie modelu Naive Bayes z e1071
bank_mod <- naiveBayes(y ~ ., data = bank_train, laplace = 1)

# Predykcja klasy na zbiorze testowym
bank_pred <- predict(bank_mod, bank_test, type = "class")

# Dopasowanie poziomów
bank_test$y <- factor(bank_test$y, levels = c("no", "yes"))
bank_pred <- factor(bank_pred, levels = levels(bank_test$y))

# Ocena skuteczności modelu
confusionMatrix(bank_pred, bank_test$y)

library(ROCR)

# Predykcja prawdopodobieństw
bank_pred <- predict(bank_mod, bank_test, type = "raw")

# Obliczenie i narysowanie krzywej ROC
roc_pred <- prediction(predictions = bank_pred[, "yes"], labels = bank_test$y)
roc_perf <- performance(roc_pred, measure = "tpr", x.measure = "fpr")
plot(roc_perf, main = "ROC Curve", col = "red", lwd = 3)
abline(a = 0, b = 1, lwd = 3, lty = 2, col = 1)

# Obliczenie AUC
auc_perf <- performance(roc_pred, measure = "auc")
bank_auc <- unlist(slot(auc_perf,"y.values"))
bank_auc

library(naivebayes)

# Wczytanie danych
bank <- read_delim("bank-full.csv", delim = ";")

# Konwersja zmiennych kategorycznych do typu factor
factor_vars <- c("job", "marital", "education", "default", "housing",
                 "loan", "contact", "month", "poutcome")
bank[factor_vars] <- lapply(bank[factor_vars], factor)

# Podział danych na zbiór treningowy i testowy (75/25)
set.seed(1234)
sample_index <- sample(nrow(bank), round(nrow(bank) * 0.75))
bank_train <- bank[sample_index, ]
bank_test  <- bank[-sample_index, ]

# Konfiguracja walidacji krzyżowej
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE)

# Trenowanie modelu Naive Bayes z caret
set.seed(1234)
bank_mod <- train(
  y ~ .,
  data = bank_train,
  method = "naive_bayes",
  trControl = ctrl,
  metric = "Accuracy"
)

# Predykcja na zbiorze testowym
bank_pred <- predict(bank_mod, bank_test)

# Dopasowanie poziomów
levels_union <- union(levels(bank_pred), levels(bank_test$y))
bank_pred <- factor(bank_pred, levels = levels_union)
bank_test$y <- factor(bank_test$y, levels = levels_union)

# Ocena skuteczności
confusionMatrix(bank_pred, bank_test$y)

# Predykcja prawdopodobieństw
bank_pred_prob <- predict(bank_mod, bank_test, type = "prob")
positive_class <- "yes"

# Obliczenie i narysowanie krzywej ROC
roc_pred <- prediction(predictions = bank_pred_prob[[positive_class]], labels = bank_test$y)
roc_perf <- performance(roc_pred, measure = "tpr", x.measure = "fpr")
plot(roc_perf, main = "ROC Curve", col = "red", lwd = 3)
abline(a = 0, b = 1, lwd = 3, lty = 2, col = 1)

# Obliczenie AUC
auc_perf <- performance(roc_pred, measure = "auc")
bank_auc <- unlist(slot(auc_perf,"y.values"))
bank_auc

#----------------------------------------------------------------------------------------
#                            MODEL DRZEWA
#----------------------------------------------------------------------------------------

library(tidyverse)

# Wczytanie danych
bank <- read_delim("bank-full.csv", delim = ";")

# Podział danych na zbiór treningowy i testowy (75/25)
set.seed(1234)
sample_index <- sample(nrow(bank), round(nrow(bank) * .75), replace = FALSE)
bank_train <- bank[sample_index, ]
bank_test  <- bank[-sample_index, ]

# Proporcje klas w zbiorze pełnym i testowym
round(prop.table(table(select(bank, y))), 2)
round(prop.table(table(select(bank_test, y))), 2)

library(rpart)

# Trenowanie drzewa decyzyjnego z bardzo małym parametrem cp (umożliwia duże drzewo)
bank_mod1 <- rpart(
  y ~ .,
  method = "class",
  data = bank_train,
  cp = 0.0001
)

library(rpart.plot)

# Wizualizacja drzewa
rpart.plot(bank_mod1)

# Wykres zależności błędu od parametru cp (complexity parameter)
plotcp(bank_mod1)

# Predykcja na zbiorze testowym
bank_pred <- predict(bank_mod1, bank_test, type = "class")
bank_test$y <- factor(bank_test$y, levels = c("no", "yes"))
bank_pred <- factor(bank_pred, levels = levels(bank_test$y))

# Macierz pomyłek
confusionMatrix(bank_pred, bank_test$y)

library(ROCR)

# Predykcja prawdopodobieństw dla ROC
bank_pred <- predict(bank_mod1, bank_test, type = "prob")
roc_pred <- prediction(predictions = bank_pred[, "yes"], labels = bank_test$y)
roc_perf <- performance(roc_pred, measure = "tpr", x.measure = "fpr")

# Krzywa ROC
plot(roc_perf, main = "ROC Curve", col = "red", lwd = 3)
abline(a = 0, b = 1, lwd = 3, lty = 2, col = 1)

# Obliczenie AUC
auc_perf <- performance(roc_pred, measure = "auc")
bank_auc <- unlist(slot(auc_perf, "y.values"))
bank_auc



bank <- read_delim("bank-full.csv", delim = ";")

# Podział danych
set.seed(1234)
sample_index <- sample(nrow(bank), round(nrow(bank) * 0.75))
bank_train <- bank[sample_index, ]
bank_test  <- bank[-sample_index, ]

# Konfiguracja walidacji krzyżowej
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE)

# Trenowanie modelu drzewem decyzyjnym z użyciem caret
set.seed(1234)
bank_mod <- train(
  y ~ .,
  data = bank_train,
  method = "rpart",
  trControl = ctrl,
  metric = "Accuracy"
)

# Predykcja klas
bank_pred <- predict(bank_mod, bank_test)
levels_union <- union(levels(bank_pred), levels(bank_test$y))
bank_pred <- factor(bank_pred, levels = levels_union)
bank_test$y <- factor(bank_test$y, levels = levels_union)

# Macierz pomyłek
confusionMatrix(bank_pred, bank_test$y)

# Predykcja prawdopodobieństw
bank_pred_prob <- predict(bank_mod, bank_test, type = "prob")
positive_class <- "yes"

# Krzywa ROC
roc_pred <- prediction(predictions = bank_pred_prob[[positive_class]], labels = bank_test$y)
roc_perf <- performance(roc_pred, measure = "tpr", x.measure = "fpr")
plot(roc_perf, main = "ROC Curve", col = "red", lwd = 3)
abline(a = 0, b = 1, lwd = 3, lty = 2, col = 1)

# Obliczenie AUC
auc_perf <- performance(roc_pred, measure = "auc")
bank_auc <- unlist(slot(auc_perf, "y.values"))
bank_auc


#----------------------------------------------------------------------------------------
#                            MODEL RANDOM FOREST
#----------------------------------------------------------------------------------------

library(readr)
library(dplyr)
library(randomForest)
library(caret)

# Wczytanie danych
bank <- read_delim("bank-full.csv", delim = ";")

# Konwersja wybranych kolumn kategorycznych na faktory
factor_vars <- c("job", "marital", "education", "default", "housing",
                 "loan", "contact", "month", "poutcome", "y")
bank[factor_vars] <- lapply(bank[factor_vars], factor)

# Usunięcie wierszy zawierających NA w kluczowych kolumnach
bank <- bank %>%
  filter_at(vars(age, job, marital, education, default, balance, housing,
                 loan, contact, day, month, duration, campaign,
                 pdays, previous, poutcome, y), ~ !is.na(.))

# Podział danych na zbiór treningowy i testowy (75/25)
set.seed(1234)
sample_index <- sample(nrow(bank), round(nrow(bank) * .75), replace = FALSE)
bank_train <- bank[sample_index, ]
bank_test  <- bank[-sample_index, ]

# Trenowanie modelu Random Forest z 150 drzewami
bank_mod <- randomForest(y ~ ., data = bank_train, ntree = 150)

# Predykcja klas
bank_pred <- predict(bank_mod, bank_test)

# Ocena skuteczności modelu
confusionMatrix(bank_pred, bank_test$y)

library(ROCR)

# Predykcja prawdopodobieństw dla klasy "yes"
bank_pred <- predict(bank_mod, bank_test, type = "prob")
roc_pred <- prediction(predictions = bank_pred[, "yes"], labels = bank_test$y)
roc_perf <- performance(roc_pred, measure = "tpr", x.measure = "fpr")

# Rysowanie krzywej ROC
plot(roc_perf, main = "ROC Curve", col = "red", lwd = 3)
abline(a = 0, b = 1, lwd = 3, lty = 2, col = 1)

# Obliczenie pola pod krzywą ROC (AUC)
auc_perf <- performance(roc_pred, measure = "auc")
bank_auc <- unlist(slot(auc_perf, "y.values"))
bank_auc


bank <- read_delim("bank-full.csv", delim = ";")

# Podział danych
set.seed(1234)
sample_index <- sample(nrow(bank), round(nrow(bank) * 0.75))
bank_train <- bank[sample_index, ]
bank_test  <- bank[-sample_index, ]

# Konfiguracja walidacji krzyżowej z 5 fałdami i obliczaniem prawdopodobieństw
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE)

# Trenowanie modelu Random Forest z caret
set.seed(1234)
bank_mod <- train(
  y ~ .,
  data = bank_train,
  method = "rf",
  trControl = ctrl,
  metric = "Accuracy"
)

# Predykcja klas na zbiorze testowym
bank_pred <- predict(bank_mod, bank_test)

# Dopasowanie poziomów
levels_union <- union(levels(bank_pred), levels(bank_test$y))
bank_pred <- factor(bank_pred, levels = levels_union)
bank_test$y <- factor(bank_test$y, levels = levels_union)

# Macierz pomyłek
confusionMatrix(bank_pred, bank_test$y)

# Predykcja prawdopodobieństw
bank_pred_prob <- predict(bank_mod, bank_test, type = "prob")
positive_class <- "yes"

# Obliczenie i narysowanie krzywej ROC
roc_pred <- prediction(predictions = bank_pred_prob[[positive_class]], labels = bank_test$y)
roc_perf <- performance(roc_pred, measure = "tpr", x.measure = "fpr")
plot(roc_perf, main = "ROC Curve", col = "red", lwd = 3)
abline(a = 0, b = 1, lwd = 3, lty = 2, col = 1)

# Obliczenie AUC
auc_perf <- performance(roc_pred, measure = "auc")
bank_auc <- unlist(slot(auc_perf, "y.values"))
bank_auc


#----------------------------------------------------------------------------------------
#                            MODEL XGBOOST
#----------------------------------------------------------------------------------------
library(readr)
library(dplyr)
library(xgboost)
library(caret)
library(ROCR)

# Wczytanie danych
bank <- read_delim("bank-full.csv", delim = ";")

# Konwersja zmiennych kategorycznych do typu factor
factor_vars <- c("job", "marital", "education", "default", "housing",
                 "loan", "contact", "month", "poutcome", "y")
bank[factor_vars] <- lapply(bank[factor_vars], factor)

# Usunięcie wierszy z brakami danych
bank <- bank %>%
  filter_at(vars(age, job, marital, education, default, balance, housing,
                 loan, contact, day, month, duration, campaign,
                 pdays, previous, poutcome, y), ~ !is.na(.))

# Podział danych na zbiór treningowy i testowy (75/25)
set.seed(1234)
sample_index <- sample(nrow(bank), round(nrow(bank) * .75), replace = FALSE)
bank_train <- bank[sample_index, ]
bank_test  <- bank[-sample_index, ]

# Trening modelu XGBoost bez walidacji krzyżowej
bank_mod <- train(
  y ~ .,
  data = bank_train,
  metric = "Accuracy",
  method = "xgbTree",
  trControl = trainControl(method = "none"),
  tuneGrid = expand.grid(
    nrounds = 100,
    max_depth = 6,
    eta = 0.3,
    gamma = 0.01,
    colsample_bytree = 1,
    min_child_weight = 1,
    subsample = 1
  )
)

# Predykcja klas
bank_pred <- predict(bank_mod, bank_test)

# Ocena skuteczności
confusionMatrix(bank_pred, bank_test$y)

# Predykcja prawdopodobieństw
bank_pred_prob <- predict(bank_mod, bank_test, type = "prob")

# Krzywa ROC i AUC
roc_pred <- prediction(predictions = bank_pred_prob[, "yes"], labels = bank_test$y)
roc_perf <- performance(roc_pred, measure = "tpr", x.measure = "fpr")
plot(roc_perf, main = "ROC Curve", col = "red", lwd = 3)
abline(a = 0, b = 1, lwd = 3, lty = 2, col = 1)

auc_perf <- performance(roc_pred, measure = "auc")
bank_auc <- unlist(slot(auc_perf, "y.values"))
bank_auc



bank <- read_delim("bank-full.csv", delim = ";")

# Podział danych
set.seed(1234)
sample_index <- sample(nrow(bank), round(nrow(bank) * 0.75))
bank_train <- bank[sample_index, ]
bank_test  <- bank[-sample_index, ]

# Konfiguracja walidacji krzyżowej
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE)

# Trening modelu XGBoost z caret i walidacją
set.seed(1234)
bank_mod <- train(
  y ~ .,
  data = bank_train,
  method = "xgbTree",
  trControl = ctrl,
  metric = "Accuracy"
)

# Predykcja klas
bank_pred <- predict(bank_mod, bank_test)

# Dopasowanie poziomów
levels_union <- union(levels(bank_pred), levels(bank_test$y))
bank_pred <- factor(bank_pred, levels = levels_union)
bank_test$y <- factor(bank_test$y, levels = levels_union)

# Macierz pomyłek
confusionMatrix(bank_pred, bank_test$y)

# Predykcja prawdopodobieństw
bank_pred_prob <- predict(bank_mod, bank_test, type = "prob")
positive_class <- "yes"

# Krzywa ROC i AUC
roc_pred <- prediction(predictions = bank_pred_prob[[positive_class]], labels = bank_test$y)
roc_perf <- performance(roc_pred, measure = "tpr", x.measure = "fpr")
plot(roc_perf, main = "ROC Curve", col = "red", lwd = 3)
abline(a = 0, b = 1, lwd = 3, lty = 2, col = 1)

auc_perf <- performance(roc_pred, measure = "auc")
bank_auc <- unlist(slot(auc_perf, "y.values"))
bank_auc

#----------------------------------------------------------------------------------------
#                            MODEL SVM
#----------------------------------------------------------------------------------------

library(tidyverse)
library(e1071)
library(ROCR)
library(caret)

# Wczytanie danych
bank <- read_delim("bank-full.csv", delim = ";")

# Konwersja zmiennych kategorycznych do typu factor
factor_vars <- c("job", "marital", "education", "default", "housing",
                 "loan", "contact", "month", "poutcome", "y")
bank[factor_vars] <- lapply(bank[factor_vars], factor)

# Usunięcie braków danych
bank <- bank %>%
  filter_at(vars(age, job, marital, education, default, balance, housing,
                 loan, contact, day, month, duration, campaign,
                 pdays, previous, poutcome, y), ~ !is.na(.))

# Podział danych na zbiór treningowy i testowy (75/25)
set.seed(1234)
sample_index <- sample(nrow(bank), round(nrow(bank) * .75), replace = FALSE)
bank_train <- bank[sample_index, ]
bank_test  <- bank[-sample_index, ]

# Trening modelu SVM z e1071 z opcją probability
bank_mod <- svm(y ~ ., data = bank_train, probability = TRUE)

# Predykcja klas
bank_pred <- predict(bank_mod, bank_test)
confusionMatrix(bank_pred, bank_test$y)

# Predykcja prawdopodobieństw
bank_pred <- predict(bank_mod, bank_test, decision.values = TRUE, probability = TRUE)
bank_pred_p <- attr(bank_pred, "probabilities")

# Krzywa ROC
roc_pred <- prediction(predictions = bank_pred_p[, "yes"], labels = bank_test$y)
roc_perf <- performance(roc_pred, measure = "tpr", x.measure = "fpr")
plot(roc_perf, main = "ROC Curve", col = "red", lwd = 3)
abline(a = 0, b = 1, lwd = 3, lty = 2, col = 1)

# AUC
auc_perf <- performance(roc_pred, measure = "auc")
heart_auc <- unlist(slot(auc_perf, "y.values"))
heart_auc


bank <- read_delim("bank-full.csv", delim = ";")

# Podział danych
set.seed(1234)
sample_index <- sample(nrow(bank), round(nrow(bank) * 0.75))
bank_train <- bank[sample_index, ]
bank_test  <- bank[-sample_index, ]

# Konfiguracja walidacji
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE)

# Trening SVM z jądrem radialnym przy użyciu caret
set.seed(1234)
bank_mod <- train(
  y ~ .,
  data = bank_train,
  method = "svmRadial",
  trControl = ctrl,
  metric = "Accuracy"
)

# Predykcja klas
bank_pred <- predict(bank_mod, bank_test)
levels_union <- union(levels(bank_pred), levels(bank_test$y))
bank_pred <- factor(bank_pred, levels = levels_union)
bank_test$y <- factor(bank_test$y, levels = levels_union)
confusionMatrix(bank_pred, bank_test$y)

# Predykcja prawdopodobieństw
bank_pred_prob <- predict(bank_mod, bank_test, type = "prob")
positive_class <- "yes"

# Krzywa ROC i AUC
roc_pred <- prediction(predictions = bank_pred_prob[[positive_class]], labels = bank_test$y)
roc_perf <- performance(roc_pred, measure = "tpr", x.measure = "fpr")
plot(roc_perf, main = "ROC Curve", col = "red", lwd = 3)
abline(a = 0, b = 1, lwd = 3, lty = 2, col = 1)

auc_perf <- performance(roc_pred, measure = "auc")
bank_auc <- unlist(slot(auc_perf, "y.values"))
bank_auc

