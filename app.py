import datetime 
import numpy as np
import pandas as pd
import re
import json
import os
import glob

import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from torch import nn
from transformers import BertModel
from transformers import AutoTokenizer

import argparse
from bs4 import BeautifulSoup
import requests

def split_essay_to_sentence(origin_essay):
    origin_essay_sentence = sum([[a.strip() for a in i.split('.')] for i in origin_essay.split('\n')], [])
    essay_sent = [a for a in origin_essay_sentence if len(a) > 0]
    return essay_sent

def get_first_extraction(text_sentence):
    row_dict = {}
    for row in tqdm(text_sentence):
        question = 'what is the feeling?'
        answer = question_answerer(question=question, context=row)
        row_dict[row] = answer
    return row_dict


def get_sent_labeldata():
    label =pd.read_csv('./rawdata/sentimental_label.csv', encoding = 'cp949', header = None)
    label[1] = label[1].apply(lambda x : re.findall(r'[가-힣]+', x)[0])
    label_dict =label[label.index % 10 == 0].set_index(0).to_dict()[1]
    emo2idx = {v : k for k, v in enumerate(label_dict.items())}
    idx2emo = {v : k[1] for k, v in emo2idx.items()}
    return emo2idx, idx2emo


class myDataset_for_infer(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        sentences =  tokenizer(self.X[idx], return_tensors = 'pt', padding = 'max_length', max_length = 96, truncation = True)
        return sentences
    
    
def infer_data(model, main_feeling_keyword):
    #ds = myDataset_for_infer()
    df_infer = myDataset_for_infer(main_feeling_keyword)

    infer_dataloader = torch.utils.data.DataLoader(df_infer, batch_size= 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        model = model.cuda()

    result_list = []
    with torch.no_grad():
        for idx, infer_input in tqdm(enumerate(infer_dataloader)):
            mask = infer_input['attention_mask'].to(device)
            input_id = infer_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            result = np.argmax(output.logits, axis=1).numpy()
            result_list.extend(result)
    return result_list

def get_word_emotion_pair(cls_model, origin_essay_sentence, idx2emo):

    import re
    def get_noun(sent):
        return [re.sub(r'[을를]+', '', vocab) for (vocab, pos) in nlp(sent) if len(vocab) > 1 and pos == 'NOUN']
    def get_adj(sent):
        return [re.sub(r'[을를]+', '', vocab) for (vocab, pos) in nlp(sent) if len(vocab) > 1 and pos == 'ADJ']
    def get_verb(sent):
        return [re.sub(r'[을를]+', '', vocab) for (vocab, pos) in nlp(sent) if len(vocab) > 1 and pos == 'VERB']

    result_list = infer_data(cls_model, origin_essay_sentence)
    final_result = pd.DataFrame(data = {'text': origin_essay_sentence , 'label' : result_list})
    final_result['emotion'] = final_result['label'].map(idx2emo)
    
    nlp=lambda x:[(x[t["start"]:t["end"]],t["entity_group"]) for t in pipeline(x)]
    #essay_sent_pos = [nlp(i) for i in tqdm(essay_sent)]
    #final_result['text_pos'] = essay_sent_pos
    final_result['noun_list'] = final_result['text'].map(get_noun)
    final_result['adj_list'] = final_result['text'].map(get_adj)
    final_result['verb_list'] = final_result['text'].map(get_verb)
    
    final_result['title'] = 'none'
    file_made_dt = datetime.datetime.now()
    file_made_dt_str = datetime.datetime.strftime(file_made_dt, '%Y%m%d_%H%M%d')
    os.makedirs(f'./result/{nickname}/{file_made_dt_str}/', exist_ok = True)
    final_result.to_csv(f"./result/{nickname}/{file_made_dt_str}/essay_result.csv", index = False)

    return final_result, file_made_dt_str


def get_essay_base_analysis(file_made_dt_str, nickname):
    essay1 = pd.read_csv(f"./result/{nickname}/{file_made_dt_str}/essay_result.csv")
    essay1['noun_list_len'] = essay1['noun_list'].apply(lambda x : len(x))
    essay1['noun_list_uniqlen'] = essay1['noun_list'].apply(lambda x : len(set(x)))
    essay1['adj_list_len'] = essay1['adj_list'].apply(lambda x : len(x))
    essay1['adj_list_uniqlen'] = essay1['adj_list'].apply(lambda x : len(set(x)))
    essay1['vocab_all'] = essay1[['noun_list','adj_list']].apply(lambda x : sum((eval(x[0]),eval(x[1])), []), axis=1)
    essay1['vocab_cnt'] = essay1['vocab_all'].apply(lambda x : len(x))
    essay1['vocab_unique_cnt'] = essay1['vocab_all'].apply(lambda x : len(set(x)))
    essay1['noun_list'] = essay1['noun_list'].apply(lambda x : eval(x))
    essay1['adj_list'] = essay1['adj_list'].apply(lambda x : eval(x))
    d = essay1.groupby('title')[['noun_list','adj_list']].sum([]).reset_index()
    d['noun_cnt'] = d['noun_list'].apply(lambda x : len(set(x)))
    d['adj_cnt'] = d['adj_list'].apply(lambda x : len(set(x)))

    # 문장 기준 최고 감정
    essay_summary =essay1.groupby(['title'])['emotion'].value_counts().unstack(level =1)

    emo_vocab_dict = {}
    for k, v in essay1[['emotion','noun_list']].values:
      for vocab in v:
        if (k, 'noun', vocab) not in emo_vocab_dict:
          emo_vocab_dict[(k, 'noun', vocab)] = 0

        emo_vocab_dict[(k, 'noun', vocab)] += 1

    for k, v in essay1[['emotion','adj_list']].values:
      for vocab in v:
        if (k, 'adj', vocab) not in emo_vocab_dict:
          emo_vocab_dict[(k, 'adj', vocab)] = 0

        emo_vocab_dict[(k, 'adj', vocab)] += 1
    vocab_emo_cnt_dict = {}
    for k, v in essay1[['emotion','noun_list']].values:
      for vocab in v:
        if (vocab, 'noun') not in vocab_emo_cnt_dict:
          vocab_emo_cnt_dict[('noun', vocab)] = {}
        if k not in vocab_emo_cnt_dict[( 'noun', vocab)]:
          vocab_emo_cnt_dict[( 'noun', vocab)][k] = 0

        vocab_emo_cnt_dict[('noun', vocab)][k] += 1

    for k, v in essay1[['emotion','adj_list']].values:
      for vocab in v:
        if ('adj', vocab) not in vocab_emo_cnt_dict:
          vocab_emo_cnt_dict[( 'adj', vocab)] = {}
        if k not in vocab_emo_cnt_dict[( 'adj', vocab)]:
          vocab_emo_cnt_dict[( 'adj', vocab)][k] = 0

        vocab_emo_cnt_dict[('adj', vocab)][k] += 1

    vocab_emo_cnt_df = pd.DataFrame(vocab_emo_cnt_dict).T
    vocab_emo_cnt_df['total'] = vocab_emo_cnt_df.sum(axis=1)
    # 단어별 최고 감정 및 감정 개수
    all_result=vocab_emo_cnt_df.sort_values(by = 'total', ascending = False)

    # 단어별 최고 감정 및 감정 개수 , 형용사 포함 시
    adj_result=vocab_emo_cnt_df.sort_values(by = 'total', ascending = False)

    # 명사만 사용 시
    noun_result=vocab_emo_cnt_df[vocab_emo_cnt_df.index.get_level_values(0) == 'noun'].sort_values(by = 'total', ascending = False)

    final_file_name = f"essay_all_vocab_result.csv"
    adj_file_name = f"essay_adj_vocab_result.csv"
    noun_file_name = f"essay_noun_vocab_result.csv"
    
    os.makedirs(f'./result/{nickname}/{file_made_dt_str}/', exist_ok = True)
    
    all_result.to_csv(f"./result/{nickname}/{file_made_dt_str}/essay_all_vocab_result.csv", index = False)
    adj_result.to_csv(f"./result/{nickname}/{file_made_dt_str}/essay_adj_vocab_result.csv", index = False)
    noun_result.to_csv(f"./result/{nickname}/{file_made_dt_str}/essay_noun_vocab_result.csv", index = False)
    
    return all_result, adj_result, noun_result, essay_summary, file_made_dt_str


from transformers import pipeline
#model_name = 'AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru'
model_name = 'monologg/koelectra-base-v2-finetuned-korquad'
question_answerer = pipeline("question-answering", model=model_name)

from transformers import AutoTokenizer,AutoModelForTokenClassification,TokenClassificationPipeline
tokenizer=AutoTokenizer.from_pretrained("KoichiYasuoka/roberta-large-korean-upos")
posmodel=AutoModelForTokenClassification.from_pretrained("KoichiYasuoka/roberta-large-korean-upos")

pipeline=TokenClassificationPipeline(tokenizer=tokenizer,
                                     model=posmodel,
                                     aggregation_strategy="simple",
                                     task = 'token-classification')
nlp=lambda x:[(x[t["start"]:t["end"]],t["entity_group"]) for t in pipeline(x)]

from transformers import AutoModelForSequenceClassification
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def all_process(origin_essay, nickname):
    essay_sent =split_essay_to_sentence(origin_essay)
    row_dict = {}
    for row in tqdm(essay_sent):
        question = 'what is the feeling?'
        answer = question_answerer(question=question, context=row)
        row_dict[row] = answer
    emo2idx, idx2emo = get_sent_labeldata()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    cls_model = AutoModelForSequenceClassification.from_pretrained('seriouspark/bert-base-multilingual-cased-finetuning-sentimental-6label')
    #cls_model = AutoModelForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels = 6)
    
    final_result, file_name_dt = get_word_emotion_pair(cls_model, essay_sent, idx2emo)
    all_result, adj_result, noun_result, essay_summary, file_made_dt_str = get_essay_base_analysis(file_name_dt, nickname)
    
    summary_result = pd.concat([adj_result, noun_result]).fillna(0).sort_values(by = 'total', ascending = False).fillna(0).reset_index()[:30]
    with open(f'./result/{nickname}/{file_name_dt}/summary.json','w') as f:
        json.dump( essay_summary.to_json(),f)
    with open(f'./result/{nickname}/{file_made_dt_str}/all_result.json','w') as f:
        json.dump( all_result.to_json(),f)    
    with open(f'./result/{nickname}/{file_made_dt_str}/adj_result.json','w') as f:
        json.dump( adj_result.to_json(),f)  
    with open(f'./result/{nickname}/{file_made_dt_str}/noun_result.json','w') as f:
        json.dump( noun_result.to_json(),f)  
    #return essay_summary, summary_result
    total_cnt = essay_summary.sum(axis=1).values[0]
    essay_summary_list = sorted(essay_summary.T.to_dict()['none'].items(), key = lambda x: x[1], reverse =True)
    essay_summary_list_str = ' '.join([f'{row[0]} {int(row[1]*100 / total_cnt)}%' for row in essay_summary_list])
    summary1 = f"""{nickname}님, 당신의 글 속에서 느껴지는 감정분포는 [{essay_summary_list_str}] 입니다"""

    return summary1

def get_similar_vocab(message):
    #print(re.findall('[가-힣]+',message))
    if (len(message) > 0) & (len(re.findall('[가-힣]+',message))>0):
        vocab = message
        all_dict_url = f"https://dict.naver.com/search.dict?dicQuery={vocab}&query={vocab}&target=dic&ie=utf8&query_utf=&isOnlyViewEE="
        response = requests.get(all_dict_url)

        html_content = response.text
        # BeautifulSoup로 HTML 파싱
        soup = BeautifulSoup(html_content, 'html.parser')
        resulttext = soup.find('script').string

        # "similarWordName" 다음의 단어 추출
        similar_words = re.findall(r'similarWordName:"([^"]+)"', resulttext)
        similar_words_final = list(set(sum([re.findall('[가-힣]+', i) for i in similar_words], [])))

        return similar_words_final
    else:
        return '단어를 입력해 주세요'
    
def get_similar_means(vocab):
    
    all_dict_url = f"https://dict.naver.com/search.dict?dicQuery={vocab}&query={vocab}&target=dic&ie=utf8&query_utf=&isOnlyViewEE="
    response = requests.get(all_dict_url)

    html_content = response.text
    # BeautifulSoup로 HTML 파싱
    soup = BeautifulSoup(html_content, 'html.parser')
    resulttext = soup.find('script').string

    # "meanList" 다음의 리스트 추출 (리스트 내용을 문자열로 추출)
    mean_list_str = re.findall(r'meanList:(\[.*?\])', resulttext, re.DOTALL)

    matches_list = []
    for i in range(len(mean_list_str)):
        matches = re.findall(r'mean:"(.*?)"', mean_list_str[i])
        matches_list.append(matches)

    mean_list_str_final = [i for i in sum(matches_list, []) if (len(re.findall(r'[A-Za-z0-9]', i) )==0 ) & (len(re.findall(r'[가-힣]', i) )!=0 )]
    
    return mean_list_str_final

info_dict = {}
#info_dict = {}
def run_all(message, history):
    global info_dict
    
    if message.find('닉네임:')>=0:
        global nickname 
        nickname = message.replace('닉네임','').replace(':','').strip()
        #global nickname
        info_dict[nickname] = {}
        return f'''좋아요! 시작할게요 {nickname}님. 
지금 머릿속에 떠오르는 단어를 하나 입력해주세요.
\n\n\n단어를 입력할 땐 \"단어: \" 를 포함해주세요
예시 <단어: 커피>
'''
    try :
        #print(nickname)
        if message.find('단어:')>=0:
            clear_message = message.replace('단어','').replace(':','').strip()
            info_dict[nickname]['main_word'] = clear_message
            vocab_mean_list = []
            similar_words_final = get_similar_vocab(message)
            similar_words_final_with_main = similar_words_final + [message]
            if len(similar_words_final_with_main)>0:
                for w in similar_words_final_with_main:
                    temp_means = get_similar_means(w)
                    vocab_mean_list.append(temp_means)
                fixed_similar_words_final = list(set([i for i in sum(vocab_mean_list, []) if len(i) > 10]))[:10]


                word_str = ' \n'.join([str(idx) + ") " + i for idx, i in enumerate(similar_words_final, 1)])
                sentence_str = ' \n'.join([str(idx) + ") " + i for idx, i in enumerate(fixed_similar_words_final, 1)])
                return f'''<{clear_message}> 을 활용한 글쓰기를 시작해볼까요? 
우선, 유사한 단어부터 확인해볼게요. 
{word_str} \n
유사한 단어들의 뜻은 아래와 같습니다. 
{sentence_str}\n
위 뜻 중에 원하는 뜻을 골라 입력해주세요 
\n\n\n 입력시엔 \"문장:\" 을 포함해주세요. 예시도 보여드릴게요.
\n 예시 <문장: 일정한 주제나 줄거리를 가진 이야기>
    '''
            else:
                return '\"단어:\" 를 포함해서 단어를 입력해주세요 (단어: 커피)'

        elif message.find('문장:')>=0:
            clear_message = message.replace('문장','').replace(':','').strip()
            info_dict[nickname]['selected_sentence'] = clear_message
            return f'''[{clear_message}]를 고르셨네요. 
\n 위 문장을 활용해 짧은 글쓰기를 해볼까요?
\n\n\n 입력시엔\"짧은글: \"을 포함해주세요. 예시도 보여드릴게요.
\n 예시 <짧은글: 지금 밥을 먹고 있는 중이다>

            '''

        elif message.find('짧은글:')>=0:
            clear_message = message.replace('짧은글','').replace(':','').strip()
            info_dict[nickname]['short_contents'] = clear_message

            return f'''<{clear_message}>라고 입력해주셨네요.
\n 위 문장을 활용해 긴 글쓰기를 해볼까요? 500자 이상 작성해주시면 좋아요.
\n\n\n 입력시엔\"긴글: \"을 포함해주세요. 예시도 보여드릴게요.
\n 예시 <긴글: 지금 밥을 먹고 있는 중이다. 밥을 먹을때 마다 나는 밥알을 혓바닥으로 굴려본다. ... (생략) >
            '''
        elif message.find('긴글:')>=0:
            long_message = message.replace('긴글','').replace(':','').strip()

            length_of_lm = len(long_message)
            if length_of_lm >= 500:
                info_dict['long_contents'] = long_message
                os.makedirs(f"./result/{nickname}/", exist_ok = True)
                with open(f"./result/{nickname}/contents.txt",'w') as f:
                    f.write(long_message)
                return f'입력해주신 글은 {length_of_lm}자 입니다. 이 글은 분석해볼만 해요. 분석을 원하신다면 "분석시작" 이라고 입력해주세요'
            else :
                return f'입력해주신 글은 {length_of_lm}자 입니다. 분석하기에 조금 짧아요. 조금 더 입력해주시겠어요?'

        elif message.find('분석시작')>=0:
            with open(f"./result/{nickname}/contents.txt",'r') as f:
                    orign_essay = f.read()
            summary = all_process(orign_essay, nickname)
            
            #print(summary)
            return summary
        else:
            return '처음부터 시작해주세요'

    except:
        return '에러가 발생했어요. 처음부터 시작합니다. 닉네임: 을 입력해주세요'
        
import gradio as gr
import requests
history = []
info_dict = {}
iface = gr.ChatInterface(
    fn=run_all,
    chatbot = gr.Chatbot(),
    textbox = gr.Textbox(placeholder='챗봇의 요청 접두사를 포함하여 입력해주세요', container = True, scale = 7),
    title = 'MooGeulMooGeul',
    description = '당신의 닉네임부터 정해서 알려주세요. "닉네임: " 을 포함해서 입력해주세요.',
    theme = 'soft',
    examples = ['닉네임: 커피러버',
                '단어: 커피',
                '문장: 일정한 주제나 줄거리를 가진 이야기',
                '짧은글: 어떤 주제나 줄거리에 대해서도 이야기를 잘 하는 사람이 하나 있었다. 나의 이모. 그 사람은 커피 한잔만 있다면 어떤 이야기든 내게 들려주었다.',
                '''긴글: 어떤 주제나 줄거리에 대해서도 이야기를 잘 하는 사람이 하나 있었다. 나의 이모. 그 사람은 커피 한 잔만 있다면 어떤 이야기든 할 수 있었다.
                어린시절의 나는 그 이야기를 듣기 위해 필사적으로 집으로 돌아왔다. 유치원때는 집에 가야 한다며 떼를 쓰고 울었다고 했다. 
                초등학생이 되어서는 4교시 땡! 하는 소리가 들리면 가방을 재빨리 싸서 집으로 돌아왔다. 집에는 항상 나를 기다리고 있는 이모와 이모의 커피 냄새가 있었다.
                따뜻한 믹스커피냄새, 그리고 고요한 집안에 울리던 이야깃거리가 생생하다. 이모는 어떻게 그 많은 이야기를 알고 있었을까. 
                한번은 정말 물어본 적이 있었다. 어떻게 해서 그런 이야기를 알고 있느냐고. 그럴때 마다 이모는 내게 어른이 되라고 말해줬다. 
                
                '어른이 되면 알 수 있어. 어른이 되렴.'
                어른, 그 당시의 나는 장래희망으로 <어른>을 써넣을 정도였다. 
                '''],
    cache_examples = False,
    retry_btn = None,
    undo_btn = 'Delete Previous',
    clear_btn = 'Clear',
                   
)
iface.launch(share=True)