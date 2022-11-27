import bs4
import requests
import multiprocessing
import wget
import sys
sys.setrecursionlimit(10000)

url = 'https://dcapswoz.ict.usc.edu/wwwdaicwoz/'

def download_link_to_file(link, file_path):
    with open(file_path, 'wb') as f:
        print(f'Downloading {link}')
        f.write(requests.get(link).content)
    # wget.download(link, out=file_path, no_check_certificate=True)
    return True

def download(l):
    return download_link_to_file(url + l.get('href'), '/Volumes/Files/monash/daic_woz/' + l.get_text())

if __name__ == '__main__':
    # download all files in the directory
    r = requests.get(url)
    data = bs4.BeautifulSoup(r.text, 'html.parser')
    function_arguments = []
    for l in data.find_all('a'):
        if l.get('href') != '/' and l.parent.name == 'td' and l.get_text().split('_')[0].isdigit() and int(l.get_text().split('_')[0]) >= 413:
            link = url + l['href']
            # get content of a tag
            function_arguments.append(l)

    pool = multiprocessing.Pool(processes=4)
    outputs = pool.map(download, function_arguments)

