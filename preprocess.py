import pandas as pd
import glob

def read_file(file):
    try:
        with open(file) as f:
            content = f.read()
        return content
    except:
        print(file)
        raise

def read_files(files):
    return [read_file(file) for file in files]

folder = '/Users/allen_kuo/Downloads/aclImdb'

neg_sentences = read_files(glob.glob('%s/train/neg/*.txt'%(folder)))

pos_sentences = read_files(glob.glob('%s/train/pos/*.txt'%(folder)))

df = pd.DataFrame()
df['comment_text'] = neg_sentences + pos_sentences
df['label'] = [0]*len(neg_sentences) + [1]*len(pos_sentences)
df.to_csv('train.csv')