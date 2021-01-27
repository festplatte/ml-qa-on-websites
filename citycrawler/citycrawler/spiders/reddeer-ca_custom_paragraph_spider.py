from .general_spider import GeneralSpider

class ReddeerCaCustomSpider(GeneralSpider):
    name = "reddeerca-custom-paragraph-spider"
    allowed_domains = ['reddeer.ca']
    start_urls = ['https://www.reddeer.ca/']
    xpath_content_selector = "//*[contains(@class, 'landing-page') or contains(@class, 'inside-page')]//*[self::p or self::ul]"
    full_pages = False
    custom_settings = {
        'ELASTICSEARCH_INDEX': 'scrapy-reddeerca-custom-paragraph'
    }
