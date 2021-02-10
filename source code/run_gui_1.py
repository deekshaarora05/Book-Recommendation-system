import numpy as np
import tkinter.font as font 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.sparse import csr_matrix
from tkinter import *
from PIL import ImageTk, Image

books = pd.read_csv('books.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')
bookTags = pd.read_csv('book_tags.csv')
toRead = pd.read_csv('to_read.csv')
genres = pd.read_csv('genres.csv')

def read_book():
	global txt
	s=""
	for x in range(50):
		s+=str(x+1)+". "+books['original_title'][x]+"\n"
	s+="And many more .... "
	txt1.insert(INSERT,s)

def topRecommendations(title):
	txt2.delete('1.0',END)
	index = indices[title]
	similarityScore = list(enumerate(cosineSim[index]))
	similarityScore = sorted(similarityScore, key = lambda x: x[1], reverse = True)
	similarityScore = similarityScore[1:10]
	bookIndex = [i[0] for i in similarityScore]
	s=""
	arr=list(bookTitles.iloc[bookIndex])
	for x in range(8):
		s+=str(x+1)+". "+arr[x]+"\n"
	txt2.insert(INSERT,s)

showingTagName = pd.merge(bookTags, tags, on = 'tag_id')
mostUsedTags = showingTagName.groupby(['tag_name'], as_index = False) \
                      .agg({'goodreads_book_id' : 'count'}) \
                      .rename(columns = {'goodreads_book_id' : 'number'}) \
                      .sort_values('number', ascending = False)
genreList = genres['tag_name'].tolist()
genreTags = tags.loc[tags['tag_name'].isin(genreList)]
mostCommonTags = pd.merge(bookTags, genreTags, on = ['tag_id'])
stringedTags = mostCommonTags.groupby('goodreads_book_id')['tag_name'].apply(lambda x: "%s" % ' '.join(x)).reset_index()
stringedTags = pd.merge(stringedTags, books[['book_id', 'authors', 'title']], left_on = ['goodreads_book_id'], \
                       right_on = ['book_id']).drop('book_id', axis = 1)
stringedTags['authors'] = stringedTags['authors'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
stringedTags['authors'] = stringedTags['authors'].astype('str').apply(lambda x: str.lower(x.replace(",", " ")))
stringedTags['all_tags'] = stringedTags['tag_name'] + " " + stringedTags['authors']

countVec = CountVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = 'english')
tagMatrix = countVec.fit_transform(stringedTags['all_tags'])
cosineSim = cosine_similarity(tagMatrix, tagMatrix)
stringedTags = stringedTags.reset_index()
bookTitles = stringedTags['title']
indices = pd.Series(stringedTags.index, index = bookTitles)

root=Tk()

l1=Label(root,text='BOOK RECOMMENDATION SYSTEM',font=font.Font(size=20),bd=1,bg='blue',fg='white')
l1.grid(row=0,column=0,columnspan=2,ipadx=50,pady=10)

l2=Label(root,text='Developed by - Rakshita Joshi, Mansi Bhatnagar and Deeksha Arora   ',font=font.Font(size=12))
l2.grid(row=1,column=0,columnspan=2,padx=20,pady=10)

l31=Label(root,text='Content base filtering',bg='green',fg='white',font=font.Font(size=14))
l32=Label(root,text='Book we have',bg='green',fg='white',font=font.Font(size=14))
l31.grid(row=2,column=0,ipadx=10,ipady=4,pady=10)
l32.grid(row=2,column=1,ipadx=10,ipady=4,pady=10)

f1=Frame(root)
l4=Label(f1,text='Enter name of a book and\n get recommendations.',font=font.Font(size=10))
l4.grid(row=0,column=0,sticky=N,pady=10,padx=10)
t1=Entry(f1,width=27,font=font.Font(size=12))
t1.grid(row=1,column=0,pady=10)
b1=Button(f1,command=lambda: topRecommendations(t1.get()),text='GET RECOMMENDATION',bg='blue',fg='white',font=font.Font(size=14))
b1.grid(row=2,column=0,sticky=N,padx=15,pady=20)
f1.grid(row=3,column=0,padx=30,pady=2)

txt1=Text(root,width=40,height=10,wrap=WORD)
txt1.grid(row=3,column=1,padx=30,pady=12,sticky=N)
read_book()

f2=Frame(root)
l5=Label(f2,text='Recommended for you',font=font.Font(size=12))
l5.grid(row=0,column=0,pady=8)
txt2=Text(f2,width=40,height=15)
txt2.grid(row=1,column=0)
f2.grid(row=4,column=0,padx=30,pady=20,stick=N)

f3=Frame(root)
c1=Canvas(f3)
c1.grid(row=0,column=0,padx=10,pady=10)  
image=Image.open('test_image_1.jpg')
img=ImageTk.PhotoImage(image.resize((400, 250), Image.ANTIALIAS))  
c1.create_image(0, 0, image=img, anchor=NW)  
f3.grid(row=4,column=1,padx=30,pady=5)

root.mainloop()