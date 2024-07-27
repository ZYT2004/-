import os
from bs4 import BeautifulSoup

def extract_reuters_text(data_dir, corpus_file):
    with open(corpus_file, 'w') as outfile:
        for filename in os.listdir(data_dir):
            if filename.endswith(".sgm"):
                with open(os.path.join(data_dir, filename), 'r', encoding='latin1') as file:
                    content = file.read()
                    soup = BeautifulSoup(content, 'html.parser')
                    for article in soup.find_all('reuters'):
                        text = article.find('text')
                        if text:
                            outfile.write(text.get_text() + '\n')

data_dir = './reuters21578'  # 提取的reuters数据集目录
corpus_file = './corpus.txt'
extract_reuters_text(data_dir, corpus_file)
