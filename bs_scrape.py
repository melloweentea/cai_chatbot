from bs4 import BeautifulSoup
import requests 

def scrape_reviews():
    html_text = requests.get("https://www.7smesupportcenter.com/standard/").text
    soup = BeautifulSoup(html_text, "lxml")
    h5 = soup.find_all("h5")
    content = soup.find_all("p")
    
    h5_text = [i.get_text() for i in h5]
    content_text = [i.get_text() for i in content]
    h5_text_cleaned =[]
    for i in h5_text:
        new_i = i.replace("\t", "")
        new_i_2 = new_i.replace("\n", "")
        h5_text_cleaned.append(new_i_2)
        
    str_write = ""
    for i in content_text:
        str_write += f"\n {i}"
    
    print(str_write)
    with open(f"data/text/website.txt", "w") as f:
        f.write(str_write)
    f.close()

if __name__ == "__main__":
    scrape_reviews()