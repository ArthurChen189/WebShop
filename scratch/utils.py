from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
from sentence_transformers import SentenceTransformer

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data_convert import recover_action


def load_model(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args["lm_path"])
    lm_model = AutoModelForSeq2SeqLM.from_pretrained(args["lm_path"])
    lm_model.eval() 
    lm_model.to(device)
    if args["sbert"]:
        sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    else:
        sbert_model = None 

    llm_model = None 

    return lm_model, tokenizer, sbert_model, llm_model


def findValidActionNew(predictions, valid_actions, sbert_model, logger, k=5):    
    # 1) if action in top k is valid, choose it
    found_valid_in_top = False
    action = None
    valid_prefix = ["click[", "search"]
    for pred in predictions[:k]:
        prefix = pred[:6] # this will be 'click[' or 'search'
        if prefix.strip() in valid_prefix:
            found_valid_in_top = True
            action = pred
            break
    if found_valid_in_top:
        return action 
    else:
        logger.info(f"No valid action found in top k={k} predictions.")
        valid_actions.sort(key=lambda x: len(x))
        logger.info("Valid Predictions: "+ str(valid_actions)) 
 

    # 2) else, find most similar action

    if sbert_model:    
        pred_vectors = sbert_model.encode(predictions[:5], batch_size=5, show_progress_bar=False)
        valid_action_vectors = sbert_model.encode(valid_actions, batch_size=min(len(valid_actions), 128), show_progress_bar=False)

        # Calculate cosine similarity between each vector in pred_vectors and all vectors in valid_action_vectors
        similarity_matrix = cosine_similarity(pred_vectors, valid_action_vectors)

        # Take the sum of cosine similarities for each vector in valid_action_vectors
        sum_similarities = similarity_matrix.sum(axis=0)

        # Find the indices of the k vectors with the highest sum of cosine similarities
        # Change this to the number of top vectors you want to retrieve
        # we set N=1 because we only want to find the most similar action
        N = 1
        top_indices = np.argpartition(sum_similarities, -N)[-N:]
        logger.info("The most similar valid actions to the predictions:")
        for ti in top_indices:
            logger.info("\t\t - "+valid_actions[ti])
        action = valid_actions[top_indices[0]] 
    if action is None:
        raise ValueError("No valid action found")
    return action 
 
    
def get_model_output(args, input_str, tokenizer, lm_model, device, logger): 
    input_ids = tokenizer(input_str, return_tensors="pt", max_length=args["max_input_len"] , truncation=True).input_ids

    sample_outputs = lm_model.generate(
        input_ids.to(device),
        max_length=args["max_output_len"],
        num_return_sequences=args['beams'],
        num_beams=args['beams'],
    )
 
    lm_pred = sample_outputs

    # Take the first prediction that is not "look around"
    logger.info("Top N Predictions:")
    predStrs = []
    for i, pred in enumerate(lm_pred):
        text = tokenizer.decode(pred)
        text = post_process_generation(text)
        logger.info("\t" + str(i) + "\t" + str(text) )
        predStrs.append(text)

    return predStrs

def post_process_generation(raw_pred):
    answer_match = re.match(r'.*\'action\': \'(.*)\',.*\'ref\': \'(.*)\'.*', raw_pred)
    if answer_match:
        action, ref = answer_match.group(1), answer_match.group(2)
        return recover_action({"action": action, "ref": ref})
    else:
        return sanitize_pred(raw_pred)

def sanitize_pred(pred):
    pred = pred.replace("\'action\': ", "").replace("\'ref\':", "")
    pred = pred.replace('<unk>', '').replace('<pad>', '').replace('</s>', '')
    return pred.strip()


if __name__ == "__main__":  
    print()