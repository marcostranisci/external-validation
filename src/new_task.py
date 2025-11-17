import yaml
from conformal_model import ConformalGeneration
import regex as re
import pandas as pd
import itertools, numpy as np

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

path = config['model_config']['path']
internal_labels = config['tasks']['internal_validity']['labels']

external_morality = config['tasks']['external_morality']['labels']
external_offensiveness = config['tasks']['external_offensiveness']['labels']

mod = ConformalGeneration(path,'auto',external_morality)


def mft_questionnaire():
    '''
    This function ask a model to answer the MFQ-2 questionnaire: https://link.springer.com/article/10.1007/s12144-024-06097-z

    TASK: rank survey items on a scale from 1 to 5, where 1 is "strongly disagree" and 5 is "strongly agree"

    '''
    results = {}
    foundations = config['tasks']['internal_validity']['items']
    
    for item in foundations:

        for v in foundations[item]:
            res = mod.generate_probs(config['tasks']['internal_validity']['prompt'].format(item=v))
            max_key = max(res,key=res.get)

            conformities,score = mod.brier(res,1)
            if item not in results:
                results[item] = {'answer': [max_key], 'uncertainty': [score.item()]}
            else:
                results[item]['answer'].append(max_key) 
                results[item]['uncertainty'].append(score.item())
                       

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
        
        for item in l:
        
            res = mod.generate_probs(config['tasks']['external_morality']['prompt'].format(item=item[0]))
            max_key = max(res,key=res.get)
            conformities,score = mod.brier(res,1)
        
        results.append({
            'annotation': item[1].lower(),
            'answer': max_key,
            'uncertainty': score.item()
        })
    
    return results

def choose_raters(a_df,profile=[1.2,3.2,2,3.4,0.9,4.1], n=10):
    age = list(set(a_df.Age.to_list()))
    gender = list(set(a_df.Gender.to_list()))
    region = list(set(a_df.Region.to_list()))

    intersections = list(itertools.product(age, gender, region))
    d = dict()
    for i in intersections:
        tmp = a_df[(a_df.Age==i[0])&(a_df.Gender==i[1])&(a_df.Region==i[2])]
        l = list()
        for _,row in tmp.iterrows():
            rater = row[['care','equality','proportionality','authority','loyalty','purity']].to_list()
            sim = np.dot(profile, rater / (np.linalg.norm(profile) * np.linalg.norm(rater)))
            l.append(sim)
        tmp['similarity'] = l

        tmp = tmp.sort_values(by='similarity', ascending=False)
        d['_'.join(list(i))] = {}
        d['_'.join(list(i))]['nearest'] = tmp.iloc[:n].rater_id.to_list()
        d['_'.join(list(i))]['farthest'] = tmp.iloc[-n:].rater_id.to_list()

    return d

def mft_offensiveness(rater='R_1hMVgTaJkuUgaDQ'):
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
        res = mod.generate_probs(config['tasks']['external_offensiveness']['prompt'].format(item=item[0]))
        max_key = max(res,key=res.get)
        conformities,score = mod.brier(res,1)
        
        results.append({
            'item_id': item[1],
            'annotation': item[-1],
            'answer': max_key,
            'uncertainty': score.item()
        })
    
    return results

x = choose_raters(pd.read_csv('data/d3-raters.csv'))

print(len(x))