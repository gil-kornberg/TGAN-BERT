import os

##################
# make master train and test files
# make a set of all te unique words
# make an embedding for each word (?)
##################

filepath = '/Users/gilkornberg/Desktop/tasks_1-20_v1-2/en-10k/train/'
# filepath = '/Users/gilkornberg/Desktop/BERT_GAN/master/master_train.txt'


def read_and_clean_file(filename):
    file = open(filename)
    file_contents = file.read()
    contents_split = file_contents.splitlines()
    totalSentenceList = []
    totalSentence = ""
    for i, line in enumerate(contents_split):
        currLine = line.rsplit()
        for token in currLine:
            if token.isdigit():
                curr = int(token)
                if curr == int(1) and curr == int(currLine[0]):
                    print(totalSentence)
                    totalSentenceList.append(totalSentence)
                    totalSentence = ""
            elif token == "\n":
                continue
            else:
                totalSentence = totalSentence + " " + token
    # for i in range(len(totalSentenceList)):
    #     if totalSentenceList[i] != "\n":
    #         print(totalSentenceList[i])


def create_vocab(master_file, vocab):
    file = open(master_file)
    file_contents = file.read()
    contents_split = file_contents.splitlines()
    for i, line in enumerate(contents_split):
        currLine = line.rsplit()
        for token in currLine:
            vocab.add(token)


if __name__ == '__main__':
    masterVocab = set()
    for f in os.listdir(filepath):
        read_and_clean_file(filepath + f)

    # for word in masterVocab:
    #     print(word)
    #     # read_and_clean_file(filepath + f)