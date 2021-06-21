# Install
#install.packages('jsonlite')
#install.packages("tm")  # for text mining
#install.packages("caTools") 
#install.packages("wordcloud") # word-cloud generator 
#install.packages("randomForest") 
#install.packages("caret") 
#install.packages("ggplot2") 
#install.packages("tidyr") 
#install.packages ('ROCR')


library('tidyr')
library("tm")
library("ggplot2")
library('jsonlite')
library('caTools')
library('randomForest')
library('caret')
library('wordcloud')
library('ROCR')

#Loading data and joining the title and body of each sample
train  <- jsonlite::fromJSON('C://Users//Lenovo//Documents//GitHub//Githubs-bug-prediction-using-Random-Forest-Classifier-using-R-language//data//embold_train.json')
test  <- jsonlite::fromJSON('C://Users//Lenovo//Documents//GitHub//Githubs-bug-prediction-using-Random-Forest-Classifier-using-R-language//data//embold_test.json')


clean_data <- function(data) {
  data$title <- paste(data$title, data$body)
  text = data$title
  text <- gsub('((www\\.[^\\s]+)|(https?://[^\\s]+))', '', text) #removes http links
  text <- gsub('<.*?>', '', text) #removing html tags
  text <- gsub('  ', ' ', text) #additional white spaces
  text <- gsub("(^|[^@\\w])@(\\w{1,15})\\b",'', text) #removes user names
  
  corpus<-Corpus(VectorSource(text))
  corpus<-tm_map(corpus, removeNumbers)
  corpus<-tm_map(corpus, content_transformer(tolower))
  corpus<-tm_map(corpus, removeWords, stopwords('english'))
  corpus<-tm_map(corpus, removeWords, c("a", "b", "c", "d", "e", "f","g","h","i","j","k","l",
                                        "m","n","o","p","q","r","s","t","u","v","w","x","y","z"))
  corpus<-tm_map(corpus, removePunctuation)
  final_corpus = tm_map(corpus, stemDocument)
  return (final_corpus)
}

#cleaning train and test data  
train_corpus <- clean_data(train)
test_corpus <- clean_data(test)

#Creating matrices 
presp_train <- DocumentTermMatrix(train_corpus)
presp_test <- DocumentTermMatrix(test_corpus)

# tf-idf of train set
tdm <- TermDocumentMatrix(train_corpus)
#reduce the sparcity
tdm <- removeSparseTerms(tdm, 0.95)
dtm_m <- as.matrix(tdm)
dtm_v <- sort(rowSums(dtm_m),decreasing=TRUE)
dtm_d <- data.frame(word = names(dtm_v),freq=dtm_v)
# Display the top 10 most frequent words
head(dtm_d, 10)

#visulization 
wordcloud(dtm_d$word, dtm_d$freq, scale=c(10,.2),min.freq=1000, colors = brewer.pal(20, 'Dark2'))
barplot(dtm_d[1:10,]$freq, las = 3, names.arg = dtm_d[1:10,]$word,
        col ="lightgreen", main ="Top 10 most frequent words",
        ylab = "Word frequencies")



#creating training and testing data
presp_train = removeSparseTerms(presp_train, 0.95)
presp_test = removeSparseTerms(presp_test, 0.95)

trainSparse = as.data.frame(as.matrix(presp_train))
colnames(trainSparse) = make.names(colnames(trainSparse))
trainSparse$label = train$label

testSparse = as.data.frame(as.matrix(presp_test))
colnames(testSparse) = make.names(colnames(testSparse))
testSparse$label = test$label

prop.table(table(trainSparse$label)) #ratio of types

#train, validation and test sets
set.seed(100)
trainSparse$label = as.factor(trainSparse$label)
#testSparse$label = as.factor(testSparse$label)

split = sample.split(trainSparse$label, SplitRatio = 0.8)
trainSparse = subset(trainSparse, split==TRUE)
valSparse = subset(trainSparse, split==FALSE)


# #model and evaluation
RF_model = randomForest(label ~ ., data=trainSparse, ntree = 300, importance=TRUE)
print(RF_model)

#rf_model <-randomForest(label ~ ., data=trainSparse, ntree = 300, importance=TRUE)
rf_prediction <- predict(RF_model, trainSparse, type = "prob")

library(pROC)
ROC_rf <- roc(trainSparse$label, rf_prediction[,2])

ROC_rf_auc <- auc(ROC_rf)


# plot ROC curves
plot(ROC_rf, col = "green", main = "ROC For Random Forest (GREEN)")

# print the performance of each model
paste("Accuracy % of random forest: ", mean(test$label == round(rf_prediction[,2], digits = 0)))
paste("Area under curve of random forest: ", ROC_rf_auc)

# No. of nodes for the trees
hist(treesize(RF_model),
     main = "No. of Nodes for the Trees",
     col = "green")

# Variable Importance
varImpPlot(RF_model,
           sort = T,
           n.var = 10,
           main = "Top 10 - Variable Importance")
importance(RF_model)
varUsed(RF_model)

#confusion matrix
predictRF = predict(RF_model, newdata=valSparse)
table(valSparse$label, predictRF)
confusionMatrix(table(valSparse$label, predictRF))

plotcm <- as.data.frame(table(valSparse$label, predictRF))
ggplot(data = plotcm, mapping = aes(x = Var1, y = predictRF)) + geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) + 
  scale_fill_gradient(low = "blue",
                      high = "red",
                      trans = "log")
#error
plot(RF_model)

