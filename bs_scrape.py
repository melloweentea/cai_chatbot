from bs4 import BeautifulSoup
import requests 

def scrape_reviews():
    html_text = requests.get("https://www.7smesupportcenter.com/contact/").text
    soup = BeautifulSoup(html_text, "lxml")
    title = soup.find_all("strong")
    content = soup.find("div", class_="sow-accordion").text.strip()
    
    # h5_text = [i.get_text() for i in h5]
    # content_text = [i.get_text() for i in content]
    # h5_text_cleaned =[]
    # for i in h5_text:
    #     new_i = i.replace("\t", "")
    #     new_i_2 = new_i.replace("\n", "")
    #     h5_text_cleaned.append(new_i_2)
        
    # str_write = ""
    # for i in content_text:
    #     str_write += f"\n {i}"
    
    print(title[2].text, content)
    with open(f"data/text/{title[2].text}.txt", "w") as f:
        f.write(f"{title[2].text}\n{content}")
    f.close()

if __name__ == "__main__":
    scrape_reviews()