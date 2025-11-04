import matplotlib.pyplot as plt
from wordcloud import WordCloud
from itertools import groupby
import re

file_content = ""
with open('Szekspir/RomeoJuliet.txt', 'r', encoding='utf-8') as file:
    file_content = file.read()

book_content = "\n".join(file_content.splitlines()[7:-1])

book_content = re.sub('\n\s*\w*\.', ' ', book_content)
book_content = re.sub('(\W|_)', ' ', book_content)
words = book_content.lower().split()

stop_words = set()
with open('Szekspir/stop_words_english.txt', 'r', encoding='utf-8') as file:
    file_content = file.read()
    stop_words = set(file_content.splitlines())

filtered_words = [word for word in words if word not in stop_words and len(word) > 2]

pairs = [(w , 1) for w in filtered_words ]

pairs.sort()
word = lambda pair: pair [0]
grouped_pairs = [(w, sum(1 for _ in g) ) for w, g in groupby(pairs, key = word)]
grouped_pairs.sort(key = lambda pair : pair [1], reverse = True)
print(f"No. of words {len(grouped_pairs)}")
print(f"Most  popular words: {grouped_pairs[:25]}")

wc = WordCloud(width=1024, height=1024, background_color='white', max_words=1000).generate_from_frequencies(dict(grouped_pairs))
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.savefig('plots/zad19_worcloud.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0)
