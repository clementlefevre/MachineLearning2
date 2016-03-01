import json
import urllib

import os

prefix = 'http://ft.bootstrap.fyre.co/bs3/v3.1/ft.fyre.co/378157/'
suffix = '/init'

users_dic = {}

urlListFile = 'ft_comments_json_list.txt'

CHART_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "ft_json_files")


class FTScraper():
    def main(self):
        self.readJson()

    def retrieveUrls(self):
        urlList = [line.rstrip('\n') for line in open(urlListFile)]
        return urlList

    def generateJson(self):
        urlToParse = []
        urlToParse = self.retrieveUrls()

        for urls in urlToParse:
            req = urllib.urlretrieve(prefix + urls + suffix, os.path.join(CHART_DIR, urls + ".json"))

    def readJson(self):
        jsonList = self.retrieveUrls()
        all_comments = []

        for json_file in jsonList:
            with open(os.path.join(CHART_DIR, json_file + '.json')) as data_file:
                data = json.load(data_file)
                comments = self.retrieve_comments(data)
                if comments is not None:
                    all_comments = all_comments + self.retrieve_comments(data)

        all_authors = self.create_authors_list(all_comments)

        with open("all_ft_comments.json", 'wb') as outfile:
            json.dump(all_comments, outfile)

        with open("all_authors.json", 'wb') as outfile:
            json.dump(all_authors, outfile)
        print "over"

    def retrieve_comments(self, data):
        list = []
        if 'headDocument' in data:
            authors = data['headDocument']['authors']
            content_list = data['headDocument']['content']
            post = self.create_dict(content_list, authors)
            list = list + post

            return list

    def create_dict(self, content_list, authors):
        posts = []
        for post in content_list:
            try:
                comments_dic = {}
                comments_dic['id'] = post['content']['id']
                comments_dic['authorId'] = post['content']['authorId']
                comments_dic['comment'] = post['content']['bodyHtml']
                nameInfos = authors.get(post['content']['authorId'])
                comments_dic['authorName'] = nameInfos['displayName']

                posts.append(comments_dic)
            except KeyError, e:
                print 'I got a KeyError - reason "%s"' % str(e) + post['content']['id']
        return posts

    def retrieve_name(self, authorsId_list, authors):
        authors_name_list = []
        for id in authorsId_list:
            # pdb.set_trace()
            name_infos = authors.get(id)
            if name_infos is None:
                name = "No Name for this ID"
            else:
                name = name_infos[u'displayName']

            authors_name_list.append(name)
        return authors_name_list

    def create_authors_list(self, all_comments):
        all_authors = []
        id = 0
        for comment in all_comments:
            author_dict = {}
            author_dict['id'] = id
            author_dict['authorId'] = comment['authorId']
            author_dict['authorName'] = comment['authorName']
            all_authors.append(author_dict)
            id += 1

        return all_authors


if __name__ == "__main__":
    FTScraper().main()
