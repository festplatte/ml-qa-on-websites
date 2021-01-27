from .general_spider import GeneralSpider

class NuernbergDeCustomSpider(GeneralSpider):
    name = "nuernbergde-custom-semantic-paragraph-spider"
    allowed_domains = ['nuernberg.de']
    disallowed_url_paths = ['tourismus.nuernberg.de/shop']
    start_urls = ['https://www.nuernberg.de/']
    xpath_content_selector = "//*[@id = 'content' or @id = 'inhalt' or @id = 'wrapper']//*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6 or self::p or self::ul or self::ol or self::table]"
    full_pages = False
    spacy_model = "de_core_news_lg"
    custom_settings = {
        'ELASTICSEARCH_INDEX': 'scrapy-nuernbergde-custom-semantic-paragraph'
    }
