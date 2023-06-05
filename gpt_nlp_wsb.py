## Import libaries
from collections import defaultdict
from configs import PATH_TO_CIRDIR
from configs import OPENAI_API_KEY, HUGGINGFACEHUB_API_TOKEN
from datetime import datetime as dt
from langchain import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.chains import (
    SequentialChain, 
    LLMChain
)
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
import numpy as np
import openai
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.sql import text

import tiktoken
import time


## Read data from db
engine = create_engine(f'sqlite:///{PATH_TO_CIRDIR}save_pandas.db', echo=False)
df = pd.read_sql('wallstreetbets_posts',con=engine)

## for partial processing
size_process = 3
df1 = df[df.flag_complete==False][:size_process]

## maximum num_of_tokens
chunk_size = 500
# To get the tokeniser corresponding to a specific model in the OpenAI API:
#enc = tiktoken.encoding_for_model("gpt-4")
enc = tiktoken.get_encoding("cl100k_base")

## build dataframe df2 to accumulate results
df2 = pd.DataFrame(columns=['translation','summarization'])
df2['url']  = df1.urls
df2['token_num'] = df1.posts.apply(lambda x: len(enc.encode(x)))
df2['chunk_num'] = df2.token_num.apply(lambda x: x//chunk_size + 1)
df2['flag_incomplete'] = True 

## split text
textsplitter = CharacterTextSplitter.from_tiktoken_encoder(
    # set a reasonable chunksize
    chunk_size=chunk_size,
    chunk_overlap = int(0.1*chunk_size),
)
df2['splitted_text'] = df.posts.apply(lambda x: textsplitter.split_text(x))

## define LLM in use
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
#llm = HuggingFaceHub(repo_id='gpt2', huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)

## result containers
result_dict = defaultdict(list)
embedding_dict = defaultdict(list)
abstract_dict = defaultdict(list)
summary_dict = defaultdict(list)

## Processing using langchain
for j, jj in zip(df2.index, df2.chunk_num):
    print('-'*20)
    print('doc number:',j, 'containing', df2.token_num[j],'tokens, splitted into', jj,'chunks.')
    print('-'*20)
    ## for every chunk in each list of document chunks.
    for i, doc in enumerate(df2.splitted_text[j]):
        print(f'chunk_num:\t{i}' )
        ## if only one chunk available:
        if i == 0:
            prompt_text=PromptTemplate(
                    input_variables=['text'],
                    template="Translate the following paragraph from English to Chinese: \n{text}"
                )
            translation_chain = LLMChain(llm=llm, 
                                            prompt=prompt_text, 
                                            output_key='translation', 
                                            )
            prompt_summary = PromptTemplate(
                input_variables=['translation'],
                template="Summarize the following paragraph into 100-200 words in Chinese: {translation}"
            )
            summarization_chain = LLMChain(llm=llm, 
                                            prompt=prompt_summary,
                                            output_key='summary'
                                            )

            sequential_chain = SequentialChain(chains=[translation_chain, summarization_chain], 
                                                input_variables=['text'], 
                                                output_variables=['translation', 'summary'])
            response = sequential_chain({'text':doc})
            result_dict[j].append(response['translation'])
            summary_dict[j].append(response['summary'])
            time.sleep(2.5)
        else:  # i>0
            # template should reuse the previous items
            prompt_text=PromptTemplate(
                    input_variables=['prior_result', 'text'],
                    template="""With '{prior_result}' as prior translation result, translate the following from English to Chinese: \n{text}\n """
                )
            translation_chain = LLMChain(llm=llm, 
                                            prompt=prompt_text, 
                                            output_key='translation', 
                                            )
            prompt_summary=PromptTemplate(
                    input_variables=['prior_summary', 'translation'],
                    template="""With '{prior_summary}' as prior summarization result, summarize the following paragraph into 100-200 words in Chinese: \n{translation}\n """
                )
            sequential_chain = SequentialChain(chains=[translation_chain, summarization_chain], 
                                                input_variables=['text', 'prior_result','prior_summary'], 
                                                output_variables=['translation', 'summary'])
            response = sequential_chain({'text':doc, 'prior_result':result_dict[j][i-1], 'prior_summary':summary_dict[j][i-1]})
            result_dict[j].append(response['translation'])
            summary_dict[j].append(response['summary'])
            time.sleep(2.5)
        #print(response['abstract'])
    df2['translation'][j] = result_dict[j]
    df2['summarization'][j] = summary_dict[j]
    df2['flag_incomplete'] = False 


## After processing, mark all the items in df as complete
df.reset_index().merge(df2.reset_index()).flag_complete=True

### TODO save the flag into wsb_db

## Abstract generation
df2['CN_abstract'] = ''
df2['CN_keywords'] = ''


## Use openAI to generate abstract and keywords
for j, jj in zip(df2.index, df2.summarization):
    print('--'*20)
    print('running doc ',j)
    print('--'*20)
    list_summary = jj
    def current_step(j, list_summary, df2):
        response = openai.ChatCompletion.create(
                                                model="gpt-3.5-turbo",
                                                messages=[
                                                        {"role": "system", "content": "You are a helpful assistant with financial domain knowledge. The input of the summarization task is separated using '; ' with approximately 10% tokens overlaps. "},
                                                        {"role": "user", "content": f"""Summarize the following chunks of summaries in to one abstract in Chinese: \n "{'; '.join(list_summary) }" """}
                                                    ]
        )
        abstract_result = response['choices'][0]['message']['content']
        df2['CN_abstract'][j] = abstract_result
        response = openai.ChatCompletion.create(
                                                model="gpt-3.5-turbo",
                                                messages=[
                                                        {"role": "system", "content": "You are a helpful assistant with financial domain knowledge. The input of this task is separated using '; ' with approximately 10% tokens overlaps. Number of keywords may smaller than 10."},
                                                        {"role": "user", "content": f"""Extract top 10 keywords from the following text: \n "{'; '.join(list_summary) }" """}
                                                    ]
        )
        CN_keywords = response['choices'][0]['message']['content']
        df2['CN_keywords'][j] = CN_keywords


    try:
        current_step(j, list_summary, df2)
    except openai.error.RateLimitError as e:
        retry_time = e.retry_after if hasattr(e, 'retry_after') else 60
        print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        current_step(j, list_summary,df2)

## Generate embedding
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
ada_emb_ser = df2.CN_abstract.apply(lambda x: get_embedding(x))


## convert list of strings into ';' separated string for SQL IO
df3 = df2.copy()
df3.translation = df3.translation.apply(lambda x: ' '.join(x))
df3.summarization = df3.summarization.apply(lambda x: ' '.join(x))
df3.drop(columns=['splitted_text'], inplace=True)
df3['processed_time'] = dt.now()
df3.to_sql('wallstreetbets_posts_processed', con=engine, if_exists='append', index=False)


## df for saving embeddings
df4 = pd.DataFrame.from_dict(dict(zip(ada_emb_ser.index, ada_emb_ser.values)) ).transpose()
df4['url'] = df2.url
df4['processed_time'] = dt.now()
df4.to_sql('wallstreetbets_posts_processed_embedding', con=engine, if_exists='append', index=False)

## update the flag in wsb_posts table

with engine.connect() as conn:
    for ii in df2.url:
        cmd = f'''
        UPDATE wallstreetbets_posts
    	SET flag_complete='1'
    	WHERE urls='{ii}' 
            '''
        conn.execute(text(cmd))

