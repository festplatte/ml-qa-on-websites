from .general_spider import GeneralSpider

class ReddeerCaCustomSpider(GeneralSpider):
    name = "reddeerca-custom-spider"
    allowed_domains = ['reddeer.ca']
    start_urls = ['https://www.reddeer.ca/']
    xpath_content_selector = "//*[contains(@class, 'landing-page') or contains(@class, 'inside-page')]//*[not(self::script) and not(self::style)]/text()"
    custom_settings = {
        'ELASTICSEARCH_INDEX': 'scrapy-reddeerca-custom'
    }
