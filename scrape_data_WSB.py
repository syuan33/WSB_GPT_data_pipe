from configs import *
from selenium import webdriver
import praw
import datetime
from sqlalchemy import text, create_engine, exc
import pandas as pd

def scrape_reddit():
    reddit = praw.Reddit(client_id = CLIENT_ID, 
            client_secret = SECRET, 
            #user = 'sam02151015',
            #password = PSWD, 
            user_agent='sam02151015',
            ratelimit_seconds = 300
            )

    subreddit = reddit.subreddit('wallstreetbets')

    with webdriver.Firefox() as driver:
        url = 'https://www.reddit.com/r/wallstreetbets/?f=flair_name%3A%22DD%22&sort=new'
        driver.get(url)
        links = driver.find_elements('xpath','//*[@class="_eYtD2XCVieq6emjKBH3m"]')
        links_list = []

        for a in links:
            link = a.find_element('xpath','../..').get_attribute('href')
            if link:
                print(link)
                links_list.append(link)

    # find the new ones using set operations
    link_set = set(links_list)
    print(f'sqlite:///{PATH_TO_CIRDIR}save_pandas.db')
    engine = create_engine(f'sqlite:///{PATH_TO_CIRDIR}save_pandas.db', echo=False)
    try:
        existing_links_set = set(pd.read_sql('wallstreetbets_posts',con=engine)['urls'].to_list())
    except exc.OperationalError:
        existing_links_set = set([])
    new_links_list = list(link_set-existing_links_set)

    # download the contents of the urls
    post_contents_list = []

    for url in new_links_list:
        print(url)
        submission = reddit.submission(url=url)
        try:
            print(submission.selftext)
            post_contents_list.append(submission.selftext)
        except:
            post_contents_list.append('')

    # use dataframe to hold and save to database
    if new_links_list != []:
        df4 = pd.DataFrame(zip(post_contents_list, links_list))
        df4.columns = ['posts','urls']
        df4['access_time'] = datetime.datetime.now()
        df4['flag_complete'] = False 
        df4.to_sql('wallstreetbets_posts', con=engine, if_exists='append', index=False)
    
    return 'done'

scrape_reddit()
