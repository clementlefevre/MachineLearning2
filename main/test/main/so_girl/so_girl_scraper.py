import urllib2

import mechanize as mechanize
import os

__author__ = 'ramon'

from bs4 import BeautifulSoup
from selenium import webdriver

URL_ROOT = "http://stackoverflow.com/users?page="
# NUM_PAGES = 138413
NUM_PAGES = 2

PIC_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "so_pics")

browser = mechanize.Browser()


class SO_scraper():
    def retrieve_all_pics(self):
        driver = webdriver.Firefox()

        for page in range(1235, 100000):
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
        img_url512 = img_url.replace("s=48", "s=512")

        if "imgur" not in img_url:
            info = ""
            req = urllib2.Request(
                img_url512)

            try:
                page = urllib2.urlopen(req)
                info = page.info()['Content-Type']

            except urllib2.HTTPError, e:
                print e.fp.read()

            if "jpeg" in info:
                self.save_profil_pic(id, info, name, page)

    def save_profil_pic(self, id, info, name, page):
        content = page.read()
        f_name = id + "_" + name + "." + info.split("/")[1]
        f = open(os.path.join(PIC_FOLDER, f_name), 'w')
        f.write(content)
        f.close()


if __name__ == '__main__':
    SO_scraper().retrieve_all_pics()
