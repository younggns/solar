import os
import pandas as pd
from collections import Counter
import gc
import pickle

invalid_authors =  ['flignir','SnausageFest','Phteven_j','Judgement_Bot_AITA','AITAMod','techiesgoboom','mary-anns-hammocks','InAHandbasket','Moggehh','tenaciousfall','AutoModerator','AmItheAsshole-ModTeam','grovesofoak','FunFatale','ElectricMayhem123','[deleted]']

def remove_invalid_authors(df):
    df.drop(df[df.author.isin(invalid_authors)].index, inplace=True)
    df.dropna(subset=['body'])
    return df

def check_judgment(text):
    if str(text) != str(text):
        return -1
    text = str(text)
    
    yta = ['YTA', 'YWBTA', 'ESH']
    nta = ['NTA', 'YWNBTA', 'NAH']
    results = [0,0]
    for elem in yta:
        if elem in text:
            results[0] = 1
    for elem in nta:
        if elem in text:
            results[1] = 1
    if sum(results) == 0:
        return -1
    if sum(results) == 2:
        return 2
    else:
        return results.index(1)

def redditors_count(df):
    df.drop(df[df.judgment.isin([-1,2])].index, inplace=True)

    author_cnt = Counter()
    author_cnt.update(df['author'].tolist())
    return author_cnt

if __name__ == "__main__":
    directory = os.fsencode(os.getcwd()+'/data/')
    author_to_cnt = {}

    df_subs = pd.read_csv(os.getcwd()+'/data/submissions.csv', sep=',')
    df_subs.dropna(subset=['url', 'selftext', 'title'], inplace=True)
    valid_subIDs = df_subs['id'].tolist()
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.startswith("comments_"):
            filepath = os.path.join(os.getcwd()+'/data/', filename)

            print(f"Processing a file {filepath}")
            _df = pd.read_csv(filepath, sep=',')
            orig_len = len(_df)

            _df = remove_invalid_authors(_df)
            _percentage = int((orig_len-len(_df))/orig_len*100)
            print(f"\t{orig_len-len(_df)}({_percentage}%) of {orig_len} instances are from moderators. Removed.")
            _df['judgment'] = _df['body'].apply(lambda x: check_judgment(x))
            judgment_cnt = Counter()
            judgment_cnt.update(_df['judgment'].tolist())
            print(f"\tNo judgment: {judgment_cnt[-1]}, Multiple judgments: {judgment_cnt[2]}, YTA: {judgment_cnt[0]}, NTA: {judgment_cnt[1]}")

            _author_cnt = redditors_count(_df)

            for key in _author_cnt:
                _author, _cnt = key, _author_cnt[key]
                if _author in author_to_cnt:
                    author_to_cnt[_author] += _cnt
                else:
                    author_to_cnt[_author] = _cnt
            
            del _df
            del _author_cnt
            gc.collect()

    with open(os.getcwd()+'/data/redditor_counts.pkl', 'wb') as f:
        pickle.dump(author_to_cnt, f)