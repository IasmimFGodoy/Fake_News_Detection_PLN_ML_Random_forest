#Importação de bibliotecas
library(xml2)
library(dplyr)
library(tidytext)
library(stringr)
library(tidyr)
library(scales)
library(ggplot2)
library(reshape2)
library(forcats)
library(caret)
library(randomForest)
# Conjuntos de dados auxiliares para análise

data(stop_words)
data(mtcars)
affin <- get_sentiments("afinn")
bing <- get_sentiments("bing")
nrc <- get_sentiments("nrc")

#Coleta:

folder <- "../Datasets/BuzzFeedNews/articles"
files <- list.files(folder, pattern = "*.xml", full.names = TRUE)

xml_docs <- lapply(files, read_xml)

df_list <- lapply(xml_docs, function(x) {
  
  author <- xml_text(xml_find_all(x, "//author"))
  title <- xml_text(xml_find_all(x, "//title"))
  maintext <- xml_text(xml_find_all(x, "//mainText"))
  portal <- xml_text(xml_find_all(x, "//portal"))
  uri <- xml_text(xml_find_all(x, "//uri"))
  veracity <- xml_text(xml_find_all(x, "//veracity"))
  author <- xml_text(xml_find_all(x, "//author"))
  
  data.frame(author, title, maintext, portal, uri, veracity, stringsAsFactors = FALSE)
})

df <- bind_rows(df_list)
df$file_code <- gsub(".xml","",basename(files))

df_author_file_code <- lapply(xml_docs, function(x) {
  
  author <- xml_text(xml_find_all(x, "//author"))
  file_code <- df$file_code <- gsub(".xml","",basename(files))
  
  data.frame(author, file_code, stringsAsFactors = FALSE)
})

df_author_file_code <- bind_rows(df_author_file_code)

news <- df %>% 
  group_by(author) %>% 
  mutate(linenumber = row_number()) %>% 
  ungroup()

# Tokenização:

tokenized_news <- news %>% 
  unnest_tokens(word, maintext)

# Limpeza de dados: Removendo as Stop Words

news_toknzd_no_stpwrds <- tokenized_news %>% 
  anti_join(stop_words)

word_counts <- news_toknzd_no_stpwrds %>% 
  count(word, sort= TRUE)

news_toknzd_no_stpwrds %>%
  count(word, sort= TRUE) %>%
  filter(n > 600) %>% 
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(n, word)) +
  geom_col() +
  labs(y = NULL)

news_toknzd_no_stpwrds_sentments_maped  <- news_toknzd_no_stpwrds %>%
  inner_join(nrc)

news_toknzd_no_stpwrds_sentments_maped <- news_toknzd_no_stpwrds_sentments_maped %>%
  inner_join(affin)

news_toknzd_no_stpwrds_sentments_maped <- news_toknzd_no_stpwrds_sentments_maped %>%
  inner_join(bing)

news_toknzd_no_stpwrds_sentments_maped %>% 
  count(word, sort = TRUE)

news_sentiment <- news_toknzd_no_stpwrds_sentments_maped %>% 
  inner_join(bing) %>% 
  count(author, portal, title, file_code, uri, veracity, word, index = linenumber, sentiment) %>% 
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>% 
  mutate(sentiment = positive - negative)

# Plotando a variância emocional dos textos
news_sentiment_plot_variance <- news_toknzd_no_stpwrds %>% 
  inner_join(bing) %>%
  count(veracity, index = linenumber %/% 5, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>% 
  mutate(sentiment = positive - negative)

ggplot(news_sentiment_plot_variance, aes(index, sentiment, fill=veracity)) + 
  geom_col(show.legend = FALSE) + 
  facet_wrap(~veracity, ncol = 2, scales = "free_x")

# Analisando a Frequência de palavras por documento: tf-idf

#variável util apenas para plotagem
words_frequency <- news_sentiment %>% 
  count(veracity, word, sort = TRUE)

total_words <- words_frequency %>% 
  group_by(veracity) %>% 
  summarize(total = sum(n))

words_frequency <- left_join(words_frequency, total_words)

ggplot(words_frequency, aes(n/total), fill=veracity) + 
  geom_histogram(show.legend = FALSE) + 
  xlim(NA, 0.1) + 
  facet_wrap(~veracity, ncol = 2, scales="free_y")


freq_by_rank <- words_frequency %>% 
  group_by(veracity) %>% 
  mutate(rank = row_number(),
         `term_frequency` = n/total) %>% 
  ungroup()

freq_by_rank %>% 
  ggplot(aes(rank, `term_frequency`, color=veracity)) + 
  geom_line(linewidth = 1.1, alpha = 0.8, show.legend = TRUE) + 
  scale_x_log10() + 
  scale_y_log10()

rank_subset <- freq_by_rank %>% 
  filter(rank < 500,
         rank > 10)


lm(log10(`term_frequency`) ~ log10(rank), data = rank_subset)
# traçando o gráfico com o valor gerado pela fórmula acima

freq_by_rank %>% 
  ggplot(aes(rank, `term_frequency`, color = veracity)) + 
  geom_abline(intercept = -0.56, slope = -1.00, 
              color = "gray50", linetype = 2) +
  geom_line(linewidth = 1.1, alpha = 0.8, show.legend = TRUE) + 
  scale_x_log10() +
  scale_y_log10()

veracity_tf_idf <- words_frequency %>% 
  bind_tf_idf(word, veracity, n)
veracity_tf_idf

#Termos com alta tf-idf - substantivos que são de fato importantes para análise textual
veracity_hight_tf_idf <- veracity_tf_idf %>% 
  select(-total) %>% 
  arrange(desc(tf_idf))

# visualização para palavras altas de tf-idf
veracity_tf_idf %>% 
  group_by(veracity) %>% 
  slice_max(tf_idf, n = 15) %>% 
  ungroup() %>% 
  ggplot(aes(tf_idf, fct_reorder(word, tf_idf), fill = veracity)) +
  geom_col(show.legend = FALSE) + 
  facet_wrap(~veracity, ncol = 5, scales = "free") + 
  labs(x = "tf-idf", y = NULL)

df_tokenized_news_word_sentiment_frequency <- news_sentiment %>% 
  count(portal, author, title, file_code, veracity , sentiment, word, sort = TRUE)

news_final_df <- df_tokenized_news_word_sentiment_frequency %>% 
  inner_join(freq_by_rank) %>% 
  inner_join(veracity_tf_idf)
news_final_df

write.csv(news_final_df,"news_final_df.csv" ,row.names = FALSE)


#Carregamento de dados para treinamento
train_data <- read.csv("news_final_df.csv")
#Carregamento de dados de teste
test_data <- read.csv("news_final_df.csv")

# Divisão do conjunto de treinamento em conjuntos de treinamento e validação usando a função createDataPartition() do pacote caret, utilizando 80% de dados para treinamento e 20% de dados para validação
set.seed(123)
train_index <- createDataPartition(train_data$veracity, p = 0.8, list = FALSE)
train_set <- train_data[train_index, ]
validation_set <- train_data[-train_index, ]

#Transformando o a variável veracity em um fator, pois contém menos de 5 classificações
train_set$veracity <- as.factor(train_set$veracity)
validation_set$veracity <- as.factor(validation_set$veracity)
test_data$veracity <- as.factor(test_data$veracity)
train_data$veracity <- as.factor(train_data$veracity)

train_set$portal <- as.factor(train_set$portal)
validation_set$portal <- as.factor(validation_set$portal)
train_data$portal <- as.factor(train_data$portal)
test_data$portal <- as.factor(test_data$portal)

train_set$word <- as.factor(train_set$word)
validation_set$word <- as.factor(validation_set$word)
train_data$word <- as.factor(train_data$word)
test_data$word <- as.factor(test_data$word)

train_set$sentiment <- as.factor(train_set$sentiment)
validation_set$sentiment <- as.factor(validation_set$sentiment)
train_data$sentiment <- as.factor(train_data$sentiment)
test_data$sentiment <- as.factor(test_data$sentiment)

#Verificando se as variáveis categóricas têm os mesmos níveis em ambos os conjuntos de dados
levels(train_set$veracity) <- levels(validation_set$veracity)
levels(train_set$portal) <- levels(validation_set$portal)
levels(train_set$word) <- levels(validation_set$word) # inconsistente
levels(train_set$sentiment) <- levels(validation_set$sentiment)


#verificando os níveis da variável word no conjunto de treinamento
unique(train_set$word)
#verificando os níveis da variável word no conjunto de validação
unique(validation_set$word)
# criando um vetor com todas as categorias presentes em ambos os conjuntos de dados
all_word_levels <- unique(c(levels(train_set$word), levels(validation_set$word)))

# Ajustando a variável word em ambos os conjuntos de dados com as mesmas categorias
train_set$word <- factor(train_set$word, levels = all_word_levels)
validation_set$word <- factor(validation_set$word, levels = all_word_levels)

# Se executar o randomForest assim dará erro, pois há muitas palavras, o máximo de categorias permitido é 53, então deve-se agrupar as categorias menos frequentes em uma categoria
# 1: verificar a frequência de cada categoria da variável word
word_freq <- table(train_set$word)
# criar uma nova variável "word_grouped" onde as categorias com frequência menor que 10 são agrupadas em "Other"
train_set$word_grouped <- ifelse(train_set$word %in% names(word_freq)[word_freq < 10], "Other", train_set$word)
validation_set$word_grouped <- ifelse(validation_set$word %in% names(word_freq)[word_freq < 10], "Other", validation_set$word)
train_data$word_grouped <- ifelse(train_data$word %in% names(word_freq)[word_freq < 10], "Other", train_data$word)
test_data$word_grouped <- ifelse(train_data$word %in% names(word_freq)[word_freq < 10], "Other", test_data$word)

# transformar a variável word_gruped em um fator e ajustar os níveis em ambos os conjuntos de dados
train_set$word_grouped <- as.factor(train_set$word_grouped)
validation_set$word_grouped <- as.factor(validation_set$word_grouped)
train_data$word_grouped <- as.factor(train_data$word_grouped)
test_data$word_grouped <- as.factor(test_data$word_grouped)

all_word_levels <- unique(c(levels(train_set$word_grouped), levels(validation_set$word_grouped), levels(train_data$word_grouped), levels(test_data$word_grouped)))
train_set$word_grouped <- factor(train_set$word_grouped, levels = all_word_levels)
validation_set$word_grouped <- factor(validation_set$word_grouped, levels = all_word_levels)
train_data$word_grouped <- factor(train_data$word_grouped, levels = all_word_levels)
test_data$word_grouped <- factor(test_data$word_grouped, levels = all_word_levels)

# Treinando o modelo de Random Forest com o conjunto de treinamento:
rf_model <- randomForest(veracity ~ portal + word_grouped + sentiment + n + total + rank + term_frequency + tf + idf + tf_idf, data = train_set, importance = TRUE)
predictions <- predict(rf_model, newdata = validation_set)
confusion_matriz <- confusionMatrix(predictions, validation_set$veracity)
confusion_matriz
#treina o modelo com o conjunto de treinamento completo
rf_model <- randomForest(veracity ~ portal + word_grouped + sentiment + n + total + rank + term_frequency + tf + idf + tf_idf, data = train_data, importance = TRUE)
predictions <- predict(rf_model, newdata = test_data)

importances <- importance(rf_model)
importances
importances_df <- data.frame(Variable = rownames(importances),
                             MeanDecreaseAccuracy = importances[, "MeanDecreaseAccuracy"],
                             MeanDecreaseGini = importances[, "MeanDecreaseGini"])
importances_df

importances_df$abs_imp <- abs(importances_df$MeanDecreaseAccuracy)

# gráfico geral de importância
ggplot(importances_df, aes(x = reorder(Variable, abs_imp), y = MeanDecreaseAccuracy)) + 
  geom_bar(stat = "identity", fill = "steelblue")+
  labs(title="Importância das variáveis para Random Forest", x="Variável", y="Importância") + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))
