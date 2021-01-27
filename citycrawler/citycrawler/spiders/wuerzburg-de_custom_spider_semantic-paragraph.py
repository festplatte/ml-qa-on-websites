from .general_spider import GeneralSpider

class WuerzburgDeCustomSpider(GeneralSpider):
    name = "wuerzburgde-custom-semantic-paragraph-spider"
    allowed_domains = ['wuerzburg.de']
    start_urls = ['https://www.wuerzburg.de/']
    xpath_content_selector = "//*[@id = 'embhl']//*[not(self::script) and not(self::style)]/*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6 or self::p or self::ul or self::ol or self::table]"
    full_pages = False
    spacy_model = "de_core_news_lg"
    custom_settings = {
        'ELASTICSEARCH_INDEX': 'scrapy-wuerzburgde-custom-semantic-paragraph'
    }
