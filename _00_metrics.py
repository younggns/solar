from evaluate import load
import numpy as np

rouge = load("rouge")
bertscore = load("bertscore")
f1 = load("f1")
accuracy = load("accuracy")

def compute_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

def compute_macrof1(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, (np.ndarray, np.generic)):
        predictions = np.argmax(logits, axis=-1)
    elif isinstance(logits, tuple):
        predictions = np.argmax(logits[0], axis=-1)
    return f1.compute(predictions=predictions, references=labels, average='macro')

def compute_rouge_metrics(eval_pred):
    predictions, labels = eval_pred

    # Replace -100 in the labels as we can't decode them.
    preds= np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    try:
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    except:
        nltk.download('punkt')
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return result

def compute_bertscore_metrics(eval_pred):
    predictions, labels = eval_pred

    # Replace -100 in the labels as we can't decode them.
    preds= np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)    
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en", idf=True, model_type="distilbert-base-uncased")
    result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en", idf=True)
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return result

def compute_action_accuracy(eval_pred):
    predictions, labels = eval_pred

    # Replace -100 in the labels as we can't decode them.
    preds= np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)    
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    def text_to_score(pred_text, gold_text):
        pred_text = pred_text.replace('REMOVE','').strip()
        gold_text = gold_text.replace('REMOVE','').strip()
        if gold_text == 'NONE' and pred_text == 'NONE':
            return 1
        elif gold_text != 'NONE' and pred_text == 'NONE':
            return 0
        elif gold_text == 'NONE' and pred_text != 'NONE':
            return 0
        else:
            gold_items = gold_text.split(',')
            gold_items = [int(elem.strip()) for elem in gold_items]

            pred_items = pred_text.split(',')
            pred_items = [int(elem.strip()) for elem in pred_items]

            _correct_cnt = 0
            for elem in pred_items:
                if elem in gold_items:
                    _correct_cnt += 1
            return _correct_cnt / len(pred_items)

    result = [text_to_score(pred, gold) for pred, gold in zip(decoded_preds, decoded_labels)]
    
    return {'accuracy': np.mean(result)}

if __name__ == "__main__":
    pass