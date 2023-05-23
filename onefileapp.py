from tkinter import *
from tkinter import ttk
import numpy as np
from numpy.linalg import norm

def core():
    global pb
    global main,mainl
    global sentence_embeddings,model,preprocess,dftemp
    import warnings
    import numpy as np
    warnings.filterwarnings("ignore")
    import pandas as pd
    main_df = pd.read_excel(r'C:\Users\DELL\OneDrive\Desktop\Internship\df_withoutdup_final_origi1.xlsx')
    dftemp=main_df.copy()
    main_df_new = pd.DataFrame()
    main_df_new["combined data"] = "-"
    main_df_new["combined data"] = main_df["Subject"] + " " + main_df["Message"]
    main_df_new.head()
    pb['value']+=20
    mainl.config(text="Performing Preprocessing")
    main.update()
    import string
    import nltk
    import re
    from nltk.stem import WordNetLemmatizer
    from string import punctuation
    from nltk.corpus import stopwords

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    def sent_tokens_func(text):
        return nltk.sent_tokenize(text)

    def word_tokens_func(text):
        return nltk.word_tokenize(text)  

    def to_lower(text):
        if not isinstance(text,str):
            text = str(text)
        return text.lower()

    def number_omit_func(text):
        output = ''.join(c for c in text if not c.isdigit())
        return output

    def remove_punctuation(text):
        return ''.join(c for c in text if c not in punctuation) 

    def stopword_remove_func(sentence):
        stop_words = stopwords.words('english')
        return ' '.join([w for w in nltk.word_tokenize(sentence) if not w in stop_words])

    def lemmatize(text):
            wordnet_lemmatizer = WordNetLemmatizer()
            lemmatized_word = [wordnet_lemmatizer.lemmatize(word)for word in nltk.word_tokenize(text)]
            return " ".join(lemmatized_word)

    def preprocess(text):
            lower_text = to_lower(text)
            sentence_tokens = sent_tokens_func(lower_text)
            word_list = []
            for each_sent in sentence_tokens:
                lemmatizzed_sent = lemmatize(each_sent)
                clean_text = number_omit_func(lemmatizzed_sent)
                clean_text = remove_punctuation(clean_text)
                clean_text = stopword_remove_func(clean_text)
                word_tokens = word_tokens_func(clean_text)
                for i in word_tokens:
                    word_list.append(i)
            return " ".join(word_list)
    pb['value']+=10
    mainl.config(text="Loading Transformer")
    main.update()
    sample_data = main_df_new['combined data']
    sample_data = sample_data.apply(preprocess)
    sample_data_new=main_df['Solution']
    sample_data_new = sample_data_new.apply(preprocess)
    main_df_new['preprocessed combined data'] = sample_data
    main_df['Solution']=sample_data
    main_df_new.to_csv("Preprocessed_data.csv",index=False)
    df_all_rows = pd.concat([main_df['Solution'],main_df_new],axis=1)
    df_all_rows.to_csv("Preprocessed__data.csv",index=False)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    pb['value']+=35
    mainl.config(text="Performing Encoding ... This May Take a While !")
    main.update()
    sentences=main_df_new['preprocessed combined data'].values.tolist()
    sentence_embeddings = np.array(model.encode(sentences))
    pb['value']+=35
    main.update()

def task():
    core()
    main.destroy()

main = Tk()
main.title("Loading ... Please Wait !")
mainl = Label(main, text="Loading Dataset")
mainl.grid(row=2,column=1,sticky="news",padx=10,pady=10)
pb = ttk.Progressbar(
    main,
    orient='horizontal',
    mode='determinate',
    length=280
)
pb.grid(row=1,column=1,sticky="news",padx=10,pady=10)
main.after(200, task)
main.mainloop()

def getsol(*args):
    global sentence_embeddings,model,preprocess,dftemp
    intext=str(e1.get())
    intext=preprocess(intext)
    intext_embedding=np.transpose(np.array(model.encode([intext])))
    cosine=np.dot(sentence_embeddings,intext_embedding)/(norm(sentence_embeddings,axis=1)*norm(intext_embedding))
    index = np.where(cosine == np.amax(cosine))
    index=list(index[0])
    e2.delete(0,"end")
    e2.insert(0,dftemp['Solution'][index[0]])
    try:
        e3.delete(0,"end")
        e3.insert(0,dftemp['Solution'][index[1]])
    except:
        pass
    try:
        e4.delete(0,"end")
        e4.insert(0,dftemp['Solution'][index[2]])
    except:
        pass
    try:
        e5.delete(0,"end")
        e5.insert(0,dftemp['Solution'][index[3]])
    except:
        pass

root=Tk()
root.title("Query")
l1=Label(root,text="Please Enter Your Query").grid(row=1,column=1,sticky="news",padx=10,pady=10)
l2=Label(root,text="Recommended Solution").grid(row=2,column=1,rowspan=4,sticky="new",padx=10,pady=10)
e1=Entry(root)
e2=Entry(root)
e3=Entry(root)
e4=Entry(root)
e5=Entry(root)
e1.grid(row=1,column=2,sticky="news",padx=10,pady=10)
e2.grid(row=2,column=2,sticky="news",padx=10,pady=10)
e3.grid(row=3,column=2,sticky="news",padx=10,pady=10)
e4.grid(row=4,column=2,sticky="news",padx=10,pady=10)
e5.grid(row=5,column=2,sticky="news",padx=10,pady=10)
e1.bind('<Return>', getsol)
root.mainloop()