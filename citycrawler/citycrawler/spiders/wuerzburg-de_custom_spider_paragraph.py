from .general_spider import GeneralSpider

class WuerzburgDeCustomSpider(GeneralSpider):
    name = "wuerzburgde-custom-paragraph-spider"
    allowed_domains = ['wuerzburg.de']
    start_urls = ['https://www.wuerzburg.de/']
    xpath_content_selector = "//*[@id = 'embhl']//*[not(self::script) and not(self::style)]/*[self::p or self::ul]"
    full_pages = False
    custom_settings = {
        'ELASTICSEARCH_INDEX': 'scrapy-wuerzburgde-custom-paragraph'
    }
