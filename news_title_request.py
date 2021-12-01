# Created by Peihong Man 3/28/19

import io, time, json
import requests
from bs4 import BeautifulSoup
import csv
import datetime
from selenium import webdriver
from selenium.webdriver.support.ui import Select


class GetData:
    def __init__(self, website_nickname, url):
        self.nickname = website_nickname
        self.url = url
        self.previous_url = self.url
        news = self.extract_news(self.url)
        print(news)
        self.save_as_csv(news, "{}.csv".format(self.nickname))

    def parse_page_cryptocurrencynews(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        # print(soup.prettify())   # This step is used to manual select features from the raw html file
        # (like picking a piece of meat from a bowl of soup)
        url_next = soup.find('link', rel='next')
        if url_next:
            url_next = url_next.get('href')
            print(url_next)
        else:
            url_next = None
        news = soup.find_all('article')
        news_list = []
        for n in news:
            title = n.find('a', rel="bookmark").text[1:]
            date = n.find('span', {"class": "entry-meta-date updated"}).text
            new = {'title': title, 'date': date}
            news_list.append(new)
        return news_list, url_next

    def parse_page_ccn(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        # print(soup.prettify())   # This step is used to manual select features from the raw html file
        # (like picking a piece of meat from a bowl of soup)
        url_next = soup.find('link', rel='next')
        if url_next:
            url_next = url_next.get('href')
            self.previous_url = url_next
            print(url_next)
        else:
            # url_next = self.previous_url[:-1] + str(int(self.previous_url[-1])+1)
            url = None
        news = soup.find_all('article')
        news_list = []
        for n in news:
            try:
                title = n.find('a', id="featured-thumbnail").get("title")
                link = n.find('a', id="featured-thumbnail").get("href")
                r = requests.get(link)
                soup_in_bowl = BeautifulSoup(r.text, 'html.parser')
                time = soup_in_bowl.find_all('meta', itemprop="name")
                time = time[0].find('meta', property="article:published_time").get("content")
                date = time[: 10]
                # date = datetime.datetime.strptime(time, '%Y-%m-%d')
                new = {'title': title, 'date': date}
                print(new)
                news_list.append(new)
            except IndexError:
                pass
        return news_list, url_next

    def parse_page_cryptonews(self, html):
        driver = webdriver.Chrome()
        driver.get(self.url)
        # loadmore = driver.find_element_by_class_name("cn-section-controls")
        loadmore = driver.find_element_by_xpath("//div[@class='cn-section-controls']/a[1]")
        # driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # driver.execute_script("arguments[0].scrollIntoView(true);", loadmore)
        driver.execute_script("arguments[0].scrollIntoViewIfNeeded(true);", loadmore)
        time.sleep(5)
        loadmore.click()
        while True:
            try:
                # loadmore = driver.find_element_by_xpath("//div[@class='cn-section-controls']/a[1]")
                driver.execute_script("arguments[0].scrollIntoViewIfNeeded(true);", loadmore)
                time.sleep(1.5)
                loadmore.click()
                time.sleep(3)
            except Exception as e:     # NoSuchElementException
                try:
                    # loadmore = driver.find_element_by_xpath("//div[@class='cn-section-controls']/a[1]")
                    driver.execute_script("arguments[0].scrollIntoViewIfNeeded(true);", loadmore)
                    time.sleep(1.5)
                    loadmore.click()
                    time.sleep(3)
                except Exception as e:
                    print(e)
                    print("Reached bottom of page")
                    break

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        # soup = BeautifulSoup(html, 'html.parser')
        # print(soup.prettify())   # This step is used to manual select features from the raw html file
        # (like picking a piece of meat from a bowl of soup)
        url_next = soup.find('link', rel='next')
        if url_next:
            url_next = url_next.get('href')
            print(url_next)
        else:
            url_next = None
        news = soup.find_all('div', class_="cn-tile article")
        news_list = []
        for n in news:
            title = n.find("h4").find('a').text
            date = n.find('time').get("datetime")[:10]
            new = {'title': title, 'date': date}
            print(new)
            news_list.append(new)
        return news_list, url_next

    def extract_news(self, url):
        headers = {'Accept': 'text/html'} # Pass custom headers to simulate your request more like from a real browser
        # headers = {'User-Agent': 'Mozilla/5.0'}
        # headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:63.0) Gecko/20100101 Firefox/63.0'}
        r = requests.get(url, headers)
        news_list, url_next = getattr(self, "parse_page_{}".format(self.nickname))(r.text)
        while url_next:
            time.sleep(1)
            r = requests.get(url_next)
            news, url_next = getattr(self, "parse_page_{}".format(self.nickname))(r.text)
            news_list.extend(news)
        return news_list

    def save_as_csv(self, news, filename):
        csv_columns = [key for key in news[0]]
        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for new in news:
                writer.writerow(new)


def main():
    # url = "https://cryptonews.com/news/bitcoin-news/"     # Major challenge: No "next page", only "load more".
    url = "https://www.ccn.com/tag/bitcoin"    # nickname: ccn
    # url = "https://cryptocurrencynews.com/daily-news/bitcoin-news/"   # nickname: cryptocurrencynews
    GetData(website_nickname="ccn", url=url)
    # GetData(website_nickname="cryptonews", url=url)
    # GetData(website_nickname="cryptocurrencynews", url=url)


if __name__ == '__main__':
    main()

