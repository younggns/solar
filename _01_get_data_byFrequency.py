import os
import sys
import pandas as pd
from tqdm import tqdm
from collections import Counter
import pickle
import gc

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

def combine_data(df_subs, df_comm, active_redditors, id_to_title, id_to_subAuthor, it_to_selftext):
    """
    df_sub: id(subID)   url created_utc author  title   selftext    score   num_comments
    df_comm: parent_id(t3_subID or t1_commID)   link_id(t3_subID)   id(commID)  created_utc author  body    score
    resulting: subID    subAuthor   title   selftext    commID  commAuthor  comment judgment
    """
    subIDs, subAuthors, titles, selftexts, commIDs, commAuthors, comments, judgments = [],[],[],[],[],[],[],[]
    print(f"\tLength of dataframe: {len(df_comm)}")

    no_subID_cnt, not_active_cnt, removed_comment_cnt, not_judgment_cnt, multi_judgment_cnt = 0, 0, 0, 0, 0
    for idx, inst in tqdm(df_comm.iterrows()):
        subId = inst['link_id'].replace('t3_','').replace('t1_','')
        if subId not in id_to_title:
            no_subID_cnt += 1
            continue
        selftext = it_to_selftext[subId]
        title = id_to_title[subId]
        subAuthor = id_to_subAuthor[subId]

        if inst['author'] not in active_redditors:
            not_active_cnt += 1
            continue

        if str(inst['body']).strip() == '[removed]' or str(inst['body']).strip() == '[deleted]':
            removed_comment_cnt += 1
            continue

        judgment = check_judgment(inst['body'])
        if judgment == 2:
            multi_judgment_cnt += 1
            continue
        elif judgment == -1:
            not_judgment_cnt += 1
            continue

        subIDs.append(subId)
        subAuthors.append(subAuthor)
        titles.append(title)
        selftexts.append(selftext)
        commIDs.append(inst['id'])
        commAuthors.append(inst['author'])
        comments.append(inst['body'])
        judgments.append(judgment)

    df_combined = pd.DataFrame(data={'subID':subIDs, 'subAuthor':subAuthors, 'title':titles, 'selftext':selftexts, 'commID':commIDs, 'commAuthor':commAuthors, 'comment':comments, 'judgment':judgments})
    
    print(f"\tNo submission ID: {no_subID_cnt}, Not active redditors: {not_active_cnt}, Removed comment: {removed_comment_cnt}, Not a judgment: {not_judgment_cnt}, Multiple judgments: {multi_judgment_cnt}")
    print(f"\tWriting {len(df_combined)} instances")
    return df_combined

if __name__ == "__main__":
    raw_directory = f'{os.getcwd()}/data/raw/'
    data_directory = f'{os.getcwd()}/data/'
    out_directory = f'{os.getcwd()}/data/intermediate'

    print("01. Loading a submission file")
    df_subs = pd.read_csv(f'{raw_directory}/submissions.csv', sep=',')
    df_subs.dropna(subset=['url', 'selftext', 'title'], inplace=True)

    id_to_title, id_to_subAuthor, it_to_selftext = {}, {}, {}
    for i, t, a, s in zip(df_subs['id'].tolist(), df_subs['title'].tolist(), df_subs['author'].tolist(), df_subs['selftext'].tolist()):
        id_to_title[i] = t
        id_to_subAuthor[i] = a
        it_to_selftext[i] = s

    print("02. Loading redditor counts")
    with open(f'{data_directory}/redditor_counts.pkl', 'rb') as f:
        author_to_cnt = pickle.load(f)
    
    active_thr = 7500
    author_to_cnt_sorted = sorted(author_to_cnt.items(), key=lambda x:x[1], reverse=True)
    active_redditors = [elem[0] for elem in author_to_cnt_sorted][:active_thr]
    # print(active_redditors)

    print("03. Process comment files")
    directory = os.fsencode(raw_directory)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.startswith("comments_"):
            filepath = os.path.join(raw_directory, filename)
            print(f"\tReading {filename}")

            df_comm = pd.read_csv(filepath, sep=',')
            df_results = combine_data(df_subs, df_comm, active_redditors, id_to_title, id_to_subAuthor, it_to_selftext)

            range_ = filename.replace('comments_','').replace('.csv','')
            df_results.to_csv(f'{out_directory}/activeComms_byFrequency_{range_}.tsv', sep='\t', index=False)

            del df_comm
            del df_results
            gc.collect()
            print()