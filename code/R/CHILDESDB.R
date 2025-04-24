# Load Library - make sure to run install.packages("data_name") in terminal
library(childesr)
library(dplyr)

# Get English -NA transcripts 
corpora_not_include <- c('POLER-Controls', 'EllisWeismer', 'Peters/Wilson')
d_eng_na <- get_transcripts(collection = "Eng-NA")
d_eng_na_filtered <- d_eng_na[!(d_eng_na$corpus_name %in% corpora_not_include),]

# Grab transcripts with only children utterances and not in excluded corpora
transcripts <- get_utterances(collection = 'Eng-NA', role = 'Target_Child')
transcripts_filtered <- transcripts[!(transcripts$corpus_name %in% corpora_not_include),]
dim(transcripts_filtered)
write.csv(transcripts_filtered, "childesdb_filtered_transcripts.csv")
