import urllib2

import mechanize as mechanize

__author__ = 'ramon'

from bs4 import BeautifulSoup
from selenium import webdriver

URL_ROOT = "http://stackoverflow.com/users?page="
# NUM_PAGES = 138413
NUM_PAGES = 2

browser = mechanize.Browser()


class SO_scraper():
    def retrieve_all_pics(self):
        driver = webdriver.Firefox()

        for page in range(NUM_PAGES):
            self.get_profile_pic(driver, page)

    def get_profile_pic(self, driver, page):
        url = URL_ROOT + str(page)
        driver.get(url)
        html = driver.page_source
        soup = BeautifulSoup(html)
        links = soup.findAll("div", {"class": "user-gravatar48"})
        for link in links:
            a = [x['href'] for x in link.findAll('a')]
            id = a[0].split('/')[2]
            name = a[0].split('/')[3]
            img_url = [x['src'] for x in link.findAll('img')][0]

            self.save_pic(id, name, img_url)

            print  a, img_url
            # pdb.set_trace()

    def save_pic(self, id, name, img_url):
        hdr = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.87 Safari/537.36'

            ,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
            'Accept-Encoding': 'none',
            'Accept-Language': 'en-US,en;q=0.8',
            'Connection': 'keep-alive'}

        if "imgur" not in img_url:
            info = ""
            req = urllib2.Request(
                img_url, headers=hdr)

            try:
                page = urllib2.urlopen(req)
                info = page.info()['Content-Type']
                print info
            except urllib2.HTTPError, e:
                print e.fp.read()

            content = page.read()
            f = open(id + "_" + name + "." + info.split("/")[1], 'w')
            f.write(content)
            f.close()


if __name__ == '__main__':
    SO_scraper().retrieve_all_pics()
