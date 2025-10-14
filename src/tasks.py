import yaml
from model import MyModel
import regex as re
import pandas as pd

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

path = config['model_config']['path']

mod = MyModel(path)


def mft_questionnaire():
    '''
    This function ask a model to answer the MFQ-2 questionnaire: https://link.springer.com/article/10.1007/s12144-024-06097-z

    TASK: rank survey items on a scale from 1 to 5, where 1 is "strongly disagree" and 5 is "strongly agree"

    '''
    results = {}
    foundations = config['tasks']['internal_validity']['items']
    
    for item in foundations:

        for v in foundations[item]:

        
            res = mod.estimate_uncertainty(config['tasks']['internal_validity']['prompt'].format(item=v))
            try: 
                answ = re.search(r'[1-5]', res.generation_text).group() 
            except: 
                answ = res.generation_text
            if item not in results:
                results[item] = {'answer': [answ], 'uncertainty': [res.uncertainty]}
            else:
                results[item]['answer'].append(answ) 
                results[item]['uncertainty'].append(res.uncertainty)              

    return results





def mft_social_media(test=20):
    '''
    This function ask a model to annotate a social media post according to the moral foundation that it expresses

    TASK: classify the post into one of the following foundations: authority, care, equality, loyalty, proportionality, purity"

    '''
    df = pd.read_csv('data/mfrc_only_moral.csv')
    if test:
        df = df[:test]
    
    l = [(x.text,x.annotation) for _,x in df.iterrows()]

    results = []
    for item in l:
        
        res = mod.estimate_uncertainty(config['tasks']['external_morality']['prompt'].format(item=item[0]))
        try:
            answ = re.search(r'authority|care|equality|loyalty|proportionality|purity', res.generation_text).group()
        except:
            answ = res.generation_text
        
        results.append({
            'annotation': item[1].lower(),
            'answer': answ,
            'uncertainty': res.uncertainty
        })
    
    return results


def mft_offensiveness(rater='R_1jeL8FDSqA0H73N'):
    '''
    This function ask a model to annotate the social media posts annotated from a given user according to their offensiveness

    TASK: rank the post on a scale from 0 to 4, where 0 is "not offensive" and 4 is "very offensive"

    '''

    ratings = pd.read_csv('data/d3-ratings.csv')
    ratings = ratings[ratings.rater_id==rater]
    texts = pd.read_csv('data/d3-items.csv')
    texts = texts.merge(ratings)
    l = [(x.text,x.item_id,x.rating_raw) for _,x in texts[:5].iterrows()]

    results = []
    for item in l:
        
        res = mod.estimate_uncertainty(config['tasks']['external_offensiveness']['prompt'].format(item=item[0]))
        try:
            answ = re.search(r'[0-4]', res.generation_text).group()
        except:
            answ = res.generation_text
        
        results.append({
            'item_id': item[1],
            'annotation': item[-1],
            'answer': answ,
            'uncertainty': res.uncertainty
        })
    
    return results

print(mft_questionnaire())