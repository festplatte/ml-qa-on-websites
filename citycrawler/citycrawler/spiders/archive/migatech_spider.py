import scrapy
from pathlib import Path
from urllib.parse import urlparse


class MigatechSpider(scrapy.Spider):
    name = "migatech"
    allowed_domains = ['migatech.de']
    start_urls = ['https://www.migatech.de/']

    # def start_requests(self):
    #     urls = [
    #         'http://quotes.toscrape.com/page/1/',
    #         'http://quotes.toscrape.com/page/2/',
    #     ]
    #     for url in urls:
    #         yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        self.save_to_file(response)

        for href in response.css('a::attr(href)').getall():
            if self.check_href(href):
                yield response.follow(href, callback=self.parse)
        # yield from response.follow_all(css='a', callback=self.parse)

    def check_href(self, href):
        parsed_link = urlparse(href)
        return parsed_link.scheme == 'http' or parsed_link.scheme == 'https' or (not parsed_link.scheme and parsed_link.path)

    def save_to_file(self, response):
        page_name = urlparse(response.url).path
        if page_name[-1:] == "/":
            page_name = page_name + "index.html"
        if not page_name:
            page_name = "/index.html"
        file_name = "result" + page_name
        path_name = file_name[:file_name.rfind("/")]

        Path(path_name).mkdir(parents=True, exist_ok=True)
        with open(file_name, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % file_name)
