import bs4
import requests

def download_link_to_file(link, file_path):
    with open(file_path, 'wb') as f:
        f.write(requests.get(link).content)

url = 'https://dcapswoz.ict.usc.edu/wwwdaicwoz/'
# download all files in the directory
r = requests.get(url)
data = bs4.BeautifulSoup(r.text, 'html.parser')
for l in data.find_all('a'):
    # check if parent of l is a td tag
    if l.get('href') != '/' and l.parent.name == 'td':
        link = url + l['href']
        print(f'Downloading {l.get("href")}')
        print(l.get_text())
        # get content of a tag
        download_link_to_file(url + l.get('href'), '/Volumes/Files/monash/daic_woz/' + l.get_text())


