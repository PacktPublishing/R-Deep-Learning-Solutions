library(tidytext)
library(janeaustenr)
library(tidyr)
library(dplyr)
library(stringr)
library(ggplot2)
library(wordcloud)
library(reshape2)
library(igraph)
library(ggraph)
library(widyr)

Pride_Prejudice <- data.frame("text" = prideprejudice, "book" = "Pride and Prejudice", "line_num" = 1:length(prideprejudice), stringsAsFactors=F)

Pride_Prejudice <- Pride_Prejudice %>% unnest_tokens(word,text)

data(stop_words)
Pride_Prejudice <- Pride_Prejudice %>% anti_join(stop_words, by="word")

most.common <- Pride_Prejudice %>% dplyr::count(word, sort = TRUE)

most.common$word  <- factor(most.common$word , levels = most.common$word)
ggplot(data=most.common[1:10,], aes(x=word, y=n, fill=word)) +
  geom_bar(colour="black", stat="identity")+
  xlab("Common Words") + ylab("N Count")+
  ggtitle("Top 10 common words")+
  guides(fill=FALSE)+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(text = element_text(size = 15))+
  theme(panel.background = element_blank(), panel.grid.major = element_blank(),panel.grid.minor = element_blank())

Pride_Prejudice_POS_NEG_sentiment <- Pride_Prejudice %>%
  inner_join(get_sentiments("bing"), by="word") %>%
  dplyr::count(book, index = line_num %/% 150, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(net_sentiment = positive - negative)

ggplot(Pride_Prejudice_POS_NEG_sentiment, aes(index, net_sentiment))+
  geom_col(show.legend = FALSE) +
  geom_line(aes(y=mean(net_sentiment)),color="blue")+
  xlab("Index (150 rows each)") + ylab("Values")+
  ggtitle("Net Sentiment (POS - NEG) of Pride and Prejudice")+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(text = element_text(size = 15))+
  theme(panel.background = element_blank(), panel.grid.major = element_blank(),panel.grid.minor = element_blank())

Pride_Prejudice_GRAN_sentiment <- Pride_Prejudice %>%
  inner_join(get_sentiments("nrc"), by="word") %>%
  dplyr::count(book, index = line_num %/% 150, sentiment) %>%
  spread(sentiment, n, fill = 0) 

ggplot(stack(Pride_Prejudice_GRAN_sentiment[,3:12]), aes(x = ind, y = values)) + geom_boxplot()+ xlab("Sentiment types") + ylab("Sections (150 words) of text")+ ggtitle("Variation across different sentiments")+ theme(plot.title = element_text(hjust = 0.5))+ theme(text = element_text(size = 15))+ theme(panel.background = element_blank(), panel.grid.major = element_blank(),panel.grid.minor = element_blank())

POS_NEG_word_counts <- Pride_Prejudice %>% inner_join(get_sentiments("bing"), by="word") %>% dplyr::count(word, sentiment, sort = TRUE) %>% ungroup()

POS_NEG_word_counts %>% group_by(sentiment) %>% top_n(10) %>%
  ungroup() %>% mutate(word = reorder(word, n)) %>% ggplot(aes(word, n, fill = sentiment)) + geom_col(show.legend = FALSE) + facet_wrap(~sentiment, scales = "free_y") + ggtitle("Top 10 positive and negative words")+ coord_flip() + theme(plot.title = element_text(hjust = 0.5))+ theme(text = element_text(size = 15))+ labs(y = NULL, x = NULL)+ theme(panel.background = element_blank(),panel.border = element_rect(linetype = "dashed", fill = NA))

Pride_Prejudice %>% inner_join(get_sentiments("bing"), by = "word") %>% dplyr::count(word, sentiment, sort = TRUE) %>% acast(word ~ sentiment, value.var = "n", fill = 0) %>% comparison.cloud(colors = c("red", "green"),max.words = 100,title.size=2,use.r.layout=TRUE, random.order=TRUE,scale=c(6,0.5))

austen_books_df <- as.data.frame(austen_books(),stringsAsFactors=F)
austen_books_df$book <- as.character(austen_books_df$book)

Pride_Prejudice_chapters <- austen_books_df %>% group_by(book) %>%    filter(book == "Pride & Prejudice") %>% mutate(chapter = cumsum(str_detect(text, regex("^chapter [\\divxlc]",ignore_case = TRUE)))) %>% ungroup() %>% unnest_tokens(word, text)

bingNEG <- get_sentiments("bing") %>% filter(sentiment == "negative") 

bingPOS <- get_sentiments("bing") %>% filter(sentiment == "positive") 

wordcounts <- Pride_Prejudice_chapters %>% group_by(book, chapter) %>% dplyr::summarize(words = n())



POS_NEG_chapter_distribution <- merge ( Pride_Prejudice_chapters %>%
                                          semi_join(bingNEG, by="word") %>%
                                          group_by(book, chapter) %>%
                                          dplyr::summarize(neg_words = n()) %>%
                                          left_join(wordcounts, by = c("book", "chapter")) %>%
                                          mutate(neg_ratio = round(neg_words*100/words,2)) %>%
                                          filter(chapter != 0) %>%
                                          ungroup(),
                                          Pride_Prejudice_chapters %>%
                                          semi_join(bingPOS, by="word") %>%
                                          group_by(book, chapter) %>%
                                          dplyr::summarize(pos_words = n()) %>%
                                          left_join(wordcounts, by = c("book", "chapter")) %>%
                                          mutate(pos_ratio = round(pos_words*100/words,2)) %>%
                                          filter(chapter != 0) %>%
                                          ungroup() )

POS_NEG_chapter_distribution$sentiment_flag <- ifelse(POS_NEG_chapter_distribution$neg_ratio > POS_NEG_chapter_distribution$pos_ratio,"NEG","POS")
table(POS_NEG_chapter_distribution$sentiment_flag)

Pride_Prejudice_chapters <- austen_books_df %>%
  group_by(book) %>%
  filter(book == "Pride & Prejudice") %>%
  mutate(linenumber = row_number(),
         chapter = cumsum(str_detect(text, regex("^chapter [\\divxlc]", 
                                                 ignore_case = TRUE)))) %>%
  ungroup() %>%
  unnest_tokens(word, text) %>%
  count(book, chapter, word, sort = TRUE) %>%
  ungroup()

total_words <- Pride_Prejudice_chapters %>% 
  summarize(total = sum(n))

Pride_Prejudice_chapters$totalwords <- total_words$total

ggplot(Pride_Prejudice_chapters, aes(n/totalwords)) +
  geom_histogram(show.legend = FALSE, bins=200) +
  xlab("Ratio wrt total words") 

freq_vs_rank <- Pride_Prejudice_chapters %>% 
  mutate(rank = row_number(), 
         term_frequency = n/totalwords)

freq_vs_rank

freq_vs_rank %>% 
  ggplot(aes(rank, term_frequency)) + 
  geom_line(size = 1.1, alpha = 0.8, show.legend = FALSE) + 
  scale_x_log10() +
  scale_y_log10()

Pride_Prejudice_chapters <- Pride_Prejudice_chapters %>%
  bind_tf_idf(word, chapter, n)
Pride_Prejudice_chapters

Pride_Prejudice_chapters %>%
  select(-totalwords) %>%
  arrange(desc(tf_idf))

Pride_Prejudice_chapters %>%
  arrange(desc(tf_idf)) %>%
  mutate(word = factor(word, levels = rev(unique(word)))) %>% 
  group_by(book) %>% 
  top_n(15) %>% 
  ungroup %>%
  ggplot(aes(word, tf_idf, fill = book)) +
  geom_col(show.legend = FALSE) +
  labs(x = NULL, y = "tf-idf") +
  facet_wrap(~book, ncol = 2, scales = "free") +
  coord_flip()