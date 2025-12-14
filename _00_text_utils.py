import re
import tiktoken

# To get the tokeniser corresponding to a specific model in the OpenAI API:
encoding = tiktoken.encoding_for_model("gpt-4o")

active_redditors = ['Ansuz07', 'huadpe', 'Z7-852', 'championofobscurity', 'kabukistar', 'chadonsunday',
        'VertigoOne', 'ZeusThunder369', 'ShiningConcepts', 'DrinkyDrank', 'physioworld', 'Snorrrlax',
        'beengrim32', 'BrawndoTTM', 'SeanFromQueens', 'krausyaoj', 'garaile64', 'AbiLovesTheology',
        'Andalib_Odulate', 'beesdaddy', 'Helicase21', 'WaterDemonPhoenix', 'chrislstark', ]
most_active_redditors = ['Ansuz07', 'huadpe', 'Z7-852', 'championofobscurity', 'kabukistar', 'VertigoOne', 'DrinkyDrank', 'physioworld']

def title_process(text):
    _placeholders = ['aita for', 'wibta for', 'aita if', 'wibta if', 'aitah', 'wibtah', 'aitas', 'wibtas', 'aita', 'wibta']
    for _plh in _placeholders:
        if text.lower().startswith(_plh):
            text = text[len(_plh):]

    text = text.strip()
    if len(text) < 1:
        return text
    if text[0] in ['?', '-', '!', ':', ',']:
        text = text[1:]
    return text.strip()

def truncate_text(text, thr=1024):
    if len(text) == 0:
        return text
    encoded = encoding.encode(text)
    return encoding.decode(encoded[:thr])

def count_text(text):
    return len(encoding.encode(text))

def remove_mode_text(text):
    text = re.sub(r"\s+To /u/\S+,\s", "@@##$$@@##$$", text)
    return text.split("@@##$$@@##$$")[0]

def convert_selftext(text):
    if text.strip() in ['[removed]', '[deleted by user]', '[deleted]', '[removed by user]']:
        return ""
    else:
        text = text.replace("&gt; *This is a footnote from the CMV moderators. ","@@##$$@@##$$")
        text = text.replace("&gt; *Hello, users of CMV! This is a footnote from your moderators. ","@@##$$@@##$$")
        return text.split("@@##$$@@##$$")[0]

def process_quoting(text):
    text_splits = text.split("\n")
    resulting_texts = []
    for elem in text_splits:
        if elem.startswith('&gt;'):
            _alt_text = elem.replace('&gt;','You mentioned that “') + '”. '
            resulting_texts.append(_alt_text)
        else:
            resulting_texts.append(elem)
    return "\n".join(resulting_texts).replace("“ ","“").replace("”, \n\n","”, ")

def process_RoT(text):
    text = text.replace('1.','-').replace('2.','-')
    if '\n' in text:
        return text
    else:
        if '"' in text:
            return text.split('"')[1] + '\n' + text.split('"')[3]
        return text.replace('; ','\n').replace('. ','\n')

def comment_process(text):
    yta = ['YTA', 'YWBTA', 'ESH']
    nta = ['NTA', 'YWNBTA', 'NAH']
    for elem in yta+nta:
        text = text.replace(elem,' ')
    text = ' '.join(text.split())

    text = text.strip()
    if len(text) < 1:
        return text
    while True:
        if text[0] in ['?', '-', '!', ':', '.', ',']:
            text = text[1:]
        else:
            break

    return text.strip()

def selftext_process(text):
    if "\nedit" in text.lower():
        _idx = text.lower().index("\nedit")
        text = text[:_idx]
    return text.strip()

def situation_value_process(text):
    elems = [item for item in text.split('\n') if item != '']
    results = []
    for elem in elems:
        refined_elem = elem.replace('versus','vs.').strip()
        refined_elem = refined_elem.replace('**','').replace('- ','').replace('*','')
        if refined_elem[0].isdigit():
            refined_elem = refined_elem[1:]
        if refined_elem[0].isdigit():
            refined_elem = refined_elem[1:]
        if refined_elem.startswith('.'):
            refined_elem = refined_elem[1:]

        refined_elem = refined_elem.strip()
        results.append(refined_elem)

    if 'X:' in results[0]:
        xs = [item.replace('X:','').strip() for item in results if 'X:' in item]
        ys = [item.replace('Y:','').strip() for item in results if 'Y:' in item]
        assert len(xs)==len(ys)
        results = [item1+' vs. '+item2 for item1,item2 in zip(xs, ys)]
    return results

def process_json_outputs(item):
    item = item.replace("'Conflicting Values'",'"Conflicting Values"').replace("'Winning Values'",'"Winning Values"')
    item = item.replace("\'Conflicting Values\'",'"Conflicting Values"').replace("\'Winning Values\'",'"Winning Values"')
    item = item.replace("],\n}","]\n}")
    item = item.replace(",\n    ]","\n    ]")
    item = item.replace("    '",'    "').replace("',\n",'",\n').replace("'\n",'"\n')
    
    if item.startswith("```json"):
        item = item.replace("```json","")
        item = item.split("```")[0]
    elif item.startswith("```python"):
        item = item.replace("```python","")
        item = item.split("```")[0]
        
    if "//" in item:
        item1 = item.split("//")[0]
        item2 = item.split("//")[1].split("\n")[0]
        item = item.replace(item2,'').replace("//","")
    if "}" not in item:
        item += "    ]\n}"
        item = item.replace(",\n    ]","\n    ]")
        
    return item.strip()