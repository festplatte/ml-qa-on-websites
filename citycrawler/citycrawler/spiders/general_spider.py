import scrapy
import spacy
from pathlib import Path
from urllib.parse import urlparse
import re

default_content_selector = 'h1 ::text, h2::text, h3 ::text, h4 ::text, h5 ::text, h6 ::text, p ::text, span ::text, td ::text'

class GeneralSpider(scrapy.Spider):
    name = "general-spider"
    allowed_domains = ['']
    allowed_url_paths = None
    disallowed_url_paths = None
    start_urls = ['']
    xpath_content_selector = None
    css_content_selector = None
    full_pages = True
    min_length = 50
    min_similarity = 0.9
    spacy_model = None
    spacy_nlp = None

    def __init__(self, name=None, **kwargs):
        if self.spacy_model:
            spacy.prefer_gpu()
            self.spacy_nlp = spacy.load(self.spacy_model)
        super().__init__(name, **kwargs)

    def parse(self, response):
        if type(response) is scrapy.http.response.html.HtmlResponse:
            # extract content
            content = self.parse_content(response)

            if self.full_pages and len(content) > self.min_length:
                yield {
                    'name': response.url,
                    'text': content
                }
            else:
                if self.spacy_nlp:
                    content = self.semantic_joining(content)
                else:
                    content = filter(lambda x: len(x) > self.min_length, content)
                for i, content_part in enumerate(content):
                    yield {
                        'name': response.url + "#" + str(i),
                        'text': content_part
                    }

            # look for follow up links
            for href in response.css('a::attr(href)').getall():
                cleaned_href = href.split('?')[0]
                if self.check_href(response.urljoin(cleaned_href)):
                    yield response.follow(cleaned_href, callback=self.parse)

    def semantic_joining(self, content_parts):
        result = []
        if len(content_parts) == 0:
            return result
        
        acc = content_parts[0]
        for i in range(1,len(content_parts)):
            part = content_parts[i]
            if len(acc) <= self.min_length or self.spacy_nlp(acc).similarity(self.spacy_nlp(part)) > self.min_similarity:
                acc += ' ' + part
            else:
                result.append(acc)
                acc = part
        result.append(acc)
        return result
        
    def join_content_parts(self, part):
        result = part.getall()
        result = map(lambda x: re.sub(r"\s", " ", x).strip(), result)
        result = filter(lambda x: x, result)
        return " ".join(result)

    def parse_content(self, response):
        content_parts = self.select_content_parts(response)
        if self.full_pages:
            return self.join_content_parts(content_parts)
        else:
            results = []
            for part in content_parts:
                results.append(self.join_content_parts(part.xpath(".//text()")))
            return results

    def select_content_parts(self, response):
        if self.xpath_content_selector:
            return response.xpath(self.xpath_content_selector)
        if self.css_content_selector:
            return response.css(self.css_content_selector)
        return response.css(default_content_selector)

    def check_href(self, href):
        is_allowed_url_path = not bool(self.allowed_url_paths)
        if self.allowed_url_paths:
            for path in self.allowed_url_paths:
                is_allowed_url_path = is_allowed_url_path or path in href

        is_disallowed_url_path = False
        if self.disallowed_url_paths:
            for path in self.disallowed_url_paths:
                is_disallowed_url_path = is_disallowed_url_path or path in href

        parsed_link = urlparse(href)
        is_allowed_url_scheme = (parsed_link.scheme == 'http' or parsed_link.scheme == 'https' or (not parsed_link.scheme and parsed_link.path))
        return is_allowed_url_scheme and is_allowed_url_path and not is_disallowed_url_path
