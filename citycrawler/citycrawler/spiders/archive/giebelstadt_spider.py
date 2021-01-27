import scrapy
from pathlib import Path
from urllib.parse import urlparse


CONTENT_SELECTOR = 'h1 ::text, h2::text, h3 ::text, h4 ::text, h5 ::text, h6 ::text, p ::text, span ::text, td ::text'


class GiebelstadtSpider(scrapy.Spider):
    name = "giebelstadt"
    allowed_domains = ['giebelstadt.de']
    start_urls = ['https://www.giebelstadt.de/']

    def parse(self, response):
        # self.save_to_file(response)
        if type(response) is scrapy.http.response.html.HtmlResponse:
            # extract content
            yield {
                'name': response.url,
                'text': self.parse_content(response)
            }
            # yield {
            #     'url': response.url,
            #     'title': response.xpath('//title/text()').get(),
            #     'content': self.parse_content(response),
            #     'html': response.body
            # }

            # look for follow up links
            for href in response.css('a::attr(href)').getall():
                if self.check_href(href):
                    yield response.follow(href, callback=self.parse)

    def parse_content(self, response):
        content_parts = response.css('body ::text').getall()
        # content_parts = response.css(
        #     CONTENT_SELECTOR).getall()
        content_parts = map(lambda x: x.strip(), content_parts)
        return "\n\n".join(content_parts)

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
