from .general_spider import GeneralSpider

class NuernbergDeCustomSpider(GeneralSpider):
    name = "nuernbergde-custom-spider"
    allowed_domains = ['nuernberg.de']
    disallowed_url_paths = ['tourismus.nuernberg.de/shop']
    start_urls = ['https://www.nuernberg.de/']
    xpath_content_selector = "//*[@id = 'content' or @id = 'inhalt' or @id = 'wrapper']//*[not(self::script) and not(self::style)]/text()"
    custom_settings = {
        'ELASTICSEARCH_INDEX': 'scrapy-nuernbergde-custom'
    }
