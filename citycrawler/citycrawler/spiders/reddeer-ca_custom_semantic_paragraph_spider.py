from .general_spider import GeneralSpider

class ReddeerCaCustomSpider(GeneralSpider):
    name = "reddeerca-custom-semantic-paragraph-spider"
    allowed_domains = ['reddeer.ca']
    start_urls = ['https://www.reddeer.ca/']
    xpath_content_selector = "//*[contains(@class, 'landing-page') or contains(@class, 'inside-page')]//*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6 or self::p or self::ul or self::ol or self::table]"
    full_pages = False
    spacy_model = "en_core_web_lg"
    custom_settings = {
        'ELASTICSEARCH_INDEX': 'scrapy-reddeerca-custom-semantic-paragraph'
    }
