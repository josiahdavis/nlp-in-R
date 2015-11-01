# ANALYSIS OF YELP DATA
# This script performs a number of NLP techniques including:
#   - Tagging Parts of Speech
#   - Filtering for only Nouns and Adjectives
#   - Removing Stopwords
#   - Stemming Words
#   - Applying a custom tokenizer
#   - Identifying the most frequent words
#   - Calculating the most interesting word
#   - Count the number of positive and negative sentiments

# Read in the data
loc <- '/Users/josiahdavis/Documents/GitHub/earl/data/'
dr <- read.csv(paste(loc, 'yelp_reviews.csv', sep=""))

# ========================== # 
# ---- PARTS OF SPEECH ----
# ========================== #

library(magrittr)
library(openNLP)

# Convert text to string format 
texts <- lapply(dr$text, as.String)

# Define types of annotations to perform
taggingPipeline <- list(
  Maxent_Sent_Token_Annotator(),
  Maxent_Word_Token_Annotator(),
  Maxent_POS_Tag_Annotator()
)

# Define function for performing the annotations
annotateEntities <- function(doc, annotation_pipeline) {
  annotations <- annotate(doc, annotation_pipeline)
  AnnotatedPlainTextDocument(doc, annotations)
}

# Annotate the texts
textsAnnotated <- texts %>% lapply(annotateEntities, taggingPipeline)

# Define the POS getter function
POSGetter <- function(doc, parts) {
  s <- doc$content
  a <- annotations(doc)[[1]]
  k <- sapply(a$features, `[[`, "POS")
  if(sum(k %in% parts) == 0){
    ""
  }else{
    s[a[k %in% parts]]
  }
}

# Identify the nouns
nouns <- textsAnnotated %>% lapply(POSGetter, parts = c("JJ", 
                                                         "JJR", 
                                                         "JJS", 
                                                         "NN", 
                                                         "NNS", 
                                                         "NNP", 
                                                         "NNPS"))
# Full list: https://goo.gl/OXLNIF

# Turn each character vector into a single string
nouns <- nouns %>% lapply(as.String)

# ============================= # 
# ---- TEXT TRANSFORMATION ----
# ============================= #

library(tm)

# Convert to dataframe
d <- data.frame(reviews = as.character(nouns))

# Replace new line characters with spaces
d$reviews <- gsub("\n", " ", d$reviews)

# Convert the relevant data into a corpus object with the tm package
d <- Corpus(VectorSource(d$reviews))

# Convert everything to lower case
d <- tm_map(d, content_transformer(tolower))

# Stem words
d <- tm_map(d, stemDocument)

# Strip whitespace
d <- tm_map(d, stripWhitespace)

# Remove punctuation
d <- tm_map(d, removePunctuation)

# ============================== # 
# ---- DOCUMENT TERM MATRIX ----
# ============================== #

# Create a frequency-based document term matrix of unigrams
dtm1 <- DocumentTermMatrix(d)

# Define a custom tokenizer
BigramTokenizer <- function(x) {
  unlist(lapply(ngrams(words(x), c(1, 2)), paste, collapse = " "), 
         use.names = FALSE)  
}

# Create a tf-idf weighted document term matrix with a custom tokenizer
dtm2 <- DocumentTermMatrix(d, control = list(weighting = weightTfIdf,
                                            tokenize = BigramTokenizer))

# Convert from sparse to dense matrix
dtm2 <- as.matrix(dtm2)

# Identify the most interesting word for each review
words <- colnames(dtm2)
getInterestingWord <- function(x){
  words[which.max(x)]
}

# Apply the previously created function to each review
dr$word <- apply(dtm2, MARGIN = 1, FUN = function(x) getInterestingWord(x))

# Calculate words length
dr$length <- apply(as.matrix(dtm1), MARGIN = 1, FUN = sum)

# ============================== # 
# ---- SENTIMENT ANALYSIS ----
# ============================== #

library(syuzhet)
library(plyr)

# For each review, calculate the count of negative and positive sentiments
getSentiment <- function(x){
  colSums(get_nrc_sentiment(get_sentences(as.character(x$text)))[c("negative", "positive")])
}

# Apply function to each row of the dataframe
dr <- adply(dr, 1, function(x) getSentiment(x))

# Create two helper variables
dr$positivity <- dr$positive / dr$wordsLength
dr$negativity <- dr$negative / dr$wordsLength
