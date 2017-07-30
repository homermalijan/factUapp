import sqlite3
import newspaper
import time

#connect to database
conn = sqlite3.connect('db/newsDb.db')

#build connection with website
adoboChronicles_paper = newspaper.build('https://adobochronicles.com/')

#parse every article
for article in adoboChronicles_paper.articles:
    #download and parse article
    try:
        article.download()
        time.sleep(3)
        article.parse()
        time.sleep(1)
    except:
        print ("ERROR DOWNLOADING HTML FILE.")
        continue

    #extract information
    authors = ','.join(article.authors)
    title = article.title
    pub_date = article.publish_date
    url = article.url
    body = article.text
    print(title)
    print(authors)
    print(pub_date)
    print(url)
    print(body)

    #insert into DB
    try:
        conn.execute('''INSERT INTO news (TITLE, AUTHORS, PUBLISH_DATE, URL, BODY, LEGIT)
            VALUES (?, ?, ?, ? ,?, ?)''', (title, authors, pub_date, url, body, 0))
        #commit changes
        conn.commit()
    except:
        print ("DUPLICATE NEWS")
    time.sleep(3)
#close Db
conn.close()
