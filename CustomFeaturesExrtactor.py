import math
import re
from sympy import nextprime
import tldextract
from urllib.parse import urlparse
import whois
import pandas as pd
import xxhash
class CustomFeatureExtractor:
    def __init__(self):
        self.SUSPICIOUS_WORDS = {"login", "secure", "verify", "update", "password", "PayPal", "signin", "bank",
                                 "account", "update",
                                 "free", "lucky", "service", "bonus", "ebayissapi", "webscr"}
        self.C2_TLDS = {
            ".best",
            ".cf",
            ".cyou",
            ".ga",
            ".gq",
            ".info",
            "pw",
            "su",
            "ws"
        }
        self.SUSPICIOUS_TLDS = {
            ".py",
            ".bid",
            ".click",
            ".download",
            ".faith",
            ".loan",
            ".men",
            ".quest",
            ".review",
            ".sbs",
            ".support",
            ".win",
            ".zip",
            ".asia",
            ".autos",
            ".bio",
            ".blue",
            ".buzz",
            ".cc",
            ".cfd",
            ".charity",
            ".club",
            ".country",
            ".dad",
            ".degree",
            ".earth",
            ".email",
            ".fit",
            ".fund",
            ".futbol",
            ".fyi",
            ".gdn",
            ".gives",
            ".gold",
            ".guru",
            ".haus",
            ".homes",
            ".id",
            ".in",
            ".ink",
            ".jetzt",
            ".kim",
            ".lat",
            ".life",
            ".live",
            ".lol",
            ".ltd",
            ".makeup",
            ".mom",
            ".monster",
            ".mov",
            ".ninja",
            ".online",
            ".pics",
            ".plus",
            ".pro",
            ".pub",
            ".racing",
            ".realtor",
            ".ren",
            ".rip",
            ".rocks",
            ".rodeo",
            ".run",
            ".shop",
            ".skin",
            ".space",
            ".support",
            ".tokyo",
            ".uno",
            ".vip",
            ".wang",
            ".wiki",
            ".work",
            ".world",
            ".xin",
            ".zone",
            ".cm",
            ".cn",
            ".cricket",
            ".ge",
            ".il",
            ".lk",
            ".me",
            ".ng",
            ".party",
            ".pk",
            ".ru",
            ".sa",
            ".science",
            ".site",
            ".stream",
            ".th",
            ".tn",
            ".top",
            ".trade",
            ".wtf"
        }
        self.MALWARE_TLDS = {
            ".icu",
            ".am",
            ".bd",
            ".cd",
            ".date",
            ".ke",
            ".zw"
        }
        self.C2_MALICIOUS_TLDS = {
            ".ml",
            ".tk",
            ".xyz"

        }
        self.PHISHING_TLDS = {
            ".xn--2scrj9c",
            ".xn--5tzm5g",
            ".xn--6frz82g",
            ".xn--czrs0t",
            ".xn--fjq720a",
            ".xn--s9brj9c",
            ".xn--unup4y",
            ".xn--vhquv",
            ".xn--xhq521b",
            ".cfd",
            ".help",
            ".rest",
            ".xn--*",
            ".bar",
            ".casa",
            ".accountant",
            ".accountants",
            ".link"
        }
        self.SENSITIVE_TLDS = {
            ".porn",
            ".sex",
            ".xxx",
            ".adult",
            ".adult",
            ".bet",
            ".cam",
            ".casino",
            ".poker",
            ".sexy",
            ".tube",
            ".webcam",
            ".webcam"
        }
        self.ccTLD_to_region = {
            ".ac": "Ascension Island",
            ".ad": "Andorra",
            ".ae": "United Arab Emirates",
            ".af": "Afghanistan",
            ".ag": "Antigua and Barbuda",
            ".ai": "Anguilla",
            ".al": "Albania",
            ".am": "Armenia",
            ".an": "Netherlands Antilles",
            ".ao": "Angola",
            ".aq": "Antarctica",
            ".ar": "Argentina",
            ".as": "American Samoa",
            ".at": "Austria",
            ".au": "Australia",
            ".aw": "Aruba",
            ".ax": "Åland Islands",
            ".az": "Azerbaijan",
            ".ba": "Bosnia and Herzegovina",
            ".bb": "Barbados",
            ".bd": "Bangladesh",
            ".be": "Belgium",
            ".bf": "Burkina Faso",
            ".bg": "Bulgaria",
            ".bh": "Bahrain",
            ".bi": "Burundi",
            ".bj": "Benin",
            ".bm": "Bermuda",
            ".bn": "Brunei Darussalam",
            ".bo": "Bolivia",
            ".br": "Brazil",
            ".bs": "Bahamas",
            ".bt": "Bhutan",
            ".bv": "Bouvet Island",
            ".bw": "Botswana",
            ".by": "Belarus",
            ".bz": "Belize",
            ".ca": "Canada",
            ".cc": "Cocos Islands",
            ".cd": "Democratic Republic of the Congo",
            ".cf": "Central African Republic",
            ".cg": "Republic of the Congo",
            ".ch": "Switzerland",
            ".ci": "Côte d'Ivoire",
            ".ck": "Cook Islands",
            ".cl": "Chile",
            ".cm": "Cameroon",
            ".cn": "China",
            ".co": "Colombia",
            ".cr": "Costa Rica",
            ".cu": "Cuba",
            ".cv": "Cape Verde",
            ".cw": "Curaçao",
            ".cx": "Christmas Island",
            ".cy": "Cyprus",
            ".cz": "Czech Republic",
            ".de": "Germany",
            ".dj": "Djibouti",
            ".dk": "Denmark",
            ".dm": "Dominica",
            ".do": "Dominican Republic",
            ".dz": "Algeria",
            ".ec": "Ecuador",
            ".ee": "Estonia",
            ".eg": "Egypt",
            ".er": "Eritrea",
            ".es": "Spain",
            ".et": "Ethiopia",
            ".eu": "European Union",
            ".fi": "Finland",
            ".fj": "Fiji",
            ".fk": "Falkland Islands",
            ".fm": "Federated States of Micronesia",
            ".fo": "Faroe Islands",
            ".fr": "France",
            ".ga": "Gabon",
            ".gb": "United Kingdom",
            ".gd": "Grenada",
            ".ge": "Georgia",
            ".gf": "French Guiana",
            ".gg": "Guernsey",
            ".gh": "Ghana",
            ".gi": "Gibraltar",
            ".gl": "Greenland",
            ".gm": "Gambia",
            ".gn": "Guinea",
            ".gp": "Guadeloupe",
            ".gq": "Equatorial Guinea",
            ".gr": "Greece",
            ".gs": "South Georgia and the South Sandwich Islands",
            ".gt": "Guatemala",
            ".gu": "Guam",
            ".gw": "Guinea-Bissau",
            ".gy": "Guyana",
            ".hk": "Hong Kong",
            ".hm": "Heard Island and McDonald Islands",
            ".hn": "Honduras",
            ".hr": "Croatia",
            ".ht": "Haiti",
            ".hu": "Hungary",
            ".id": "Indonesia",
            ".ie": "Ireland",
            ".il": "Israel",
            ".im": "Isle of Man",
            ".in": "India",
            ".io": "British Indian Ocean Territory",
            ".iq": "Iraq",
            ".ir": "Iran",
            ".is": "Iceland",
            ".it": "Italy",
            ".je": "Jersey",
            ".jm": "Jamaica",
            ".jo": "Jordan",
            ".jp": "Japan",
            ".ke": "Kenya",
            ".kg": "Kyrgyzstan",
            ".kh": "Cambodia",
            ".ki": "Kiribati",
            ".km": "Comoros",
            ".kn": "Saint Kitts and Nevis",
            ".kp": "Democratic People's Republic of Korea (North Korea)",
            ".kr": "Republic of Korea (South Korea)",
            ".kw": "Kuwait",
            ".ky": "Cayman Islands",
            ".kz": "Kazakhstan",
            ".la": "Laos",
            ".lb": "Lebanon",
            ".lc": "Saint Lucia",
            ".li": "Liechtenstein",
            ".lk": "Sri Lanka",
            ".lr": "Liberia",
            ".ls": "Lesotho",
            ".lt": "Lithuania",
            ".lu": "Luxembourg",
            ".lv": "Latvia",
            ".ly": "Libya",
            ".ma": "Morocco",
            ".mc": "Monaco",
            ".md": "Moldova",
            ".me": "Montenegro",
            ".mf": "Saint Martin (French part)",
            ".mg": "Madagascar",
            ".mh": "Marshall Islands",
            ".mk": "North Macedonia",
            ".ml": "Mali",
            ".mm": "Myanmar",
            ".mn": "Mongolia",
            ".mo": "Macao",
            ".mp": "Northern Mariana Islands",
            ".mq": "Martinique",
            ".mr": "Mauritania",
            ".ms": "Montserrat",
            ".mt": "Malta",
            ".mu": "Mauritius",
            ".mv": "Maldives",
            ".mw": "Malawi",
            ".mx": "Mexico",
            ".my": "Malaysia",
            ".mz": "Mozambique",
            ".na": "Namibia",
            ".nc": "New Caledonia",
            ".ne": "Niger",
            ".nf": "Norfolk Island",
            ".ng": "Nigeria",
            ".ni": "Nicaragua",
            ".nl": "Netherlands",
            ".no": "Norway",
            ".np": "Nepal",
            ".nr": "Nauru",
            ".nu": "Niue",
            ".nz": "New Zealand",
            ".om": "Oman",
            ".pa": "Panama",
            ".pe": "Peru",
            ".pf": "French Polynesia",
            ".pg": "Papua New Guinea",
            ".ph": "Philippines",
            ".pk": "Pakistan",
            ".pl": "Poland",
            ".pm": "Saint Pierre and Miquelon",
            ".pn": "Pitcairn",
            ".pr": "Puerto Rico",
            ".ps": "Palestinian Territory",
            ".pt": "Portugal",
            ".pw": "Palau",
            ".py": "Paraguay",
            ".qa": "Qatar",
            ".re": "Réunion",
            ".ro": "Romania",
            ".rs": "Serbia",
            ".ru": "Russia",
            ".rw": "Rwanda",
            ".sa": "Saudi Arabia",
            ".sb": "Solomon Islands",
            ".sc": "Seychelles",
            ".sd": "Sudan",
            ".se": "Sweden",
            ".sg": "Singapore",
            ".sh": "Saint Helena",
            ".si": "Slovenia",
            ".sj": "Svalbard and Jan Mayen",
            ".sk": "Slovakia",
            ".sl": "Sierra Leone",
            ".sm": "San Marino",
            ".sn": "Senegal",
            ".so": "Somalia",
            ".sr": "Suriname",
            ".ss": "South Sudan",
            ".st": "São Tomé and Príncipe",
            ".sv": "El Salvador",
            ".sx": "Sint Maarten (Dutch part)",
            ".sy": "Syria",
            ".sz": "Eswatini",
            ".tc": "Turks and Caicos Islands",
            ".td": "Chad",
            ".tf": "French Southern Territories",
            ".tg": "Togo",
            ".th": "Thailand",
            ".tj": "Tajikistan",
            ".tk": "Tokelau",
            ".tl": "Timor-Leste",
            ".tm": "Turkmenistan",
            ".tn": "Tunisia",
            ".to": "Tonga",
            ".tr": "Turkey",
            ".tt": "Trinidad and Tobago",
            ".tv": "Tuvalu",
            ".tw": "Taiwan",
            ".tz": "Tanzania",
            ".ua": "Ukraine",
            ".ug": "Uganda",
            ".uk": "United Kingdom",
            ".us": "United States",
            ".uy": "Uruguay",
            ".uz": "Uzbekistan",
            ".va": "Vatican City",
            ".vc": "Saint Vincent and the Grenadines",
            ".ve": "Venezuela",
            ".vg": "British Virgin Islands",
            ".vi": "U.S. Virgin Islands",
            ".vn": "Vietnam",
            ".vu": "Vanuatu",
            ".wf": "Wallis and Futuna",
            ".ws": "Samoa",
            ".ye": "Yemen",
            ".yt": "Mayotte",
            ".za": "South Africa",
            ".zm": "Zambia",
            ".zw": "Zimbabwe"
        }
        self.label_mapping_url = {
            'benign': 0,
            'defacement': 1,
            'phishing': 2,
            'malware': 3
        }
        self.N = nextprime(10 ** 7)

    def fast_hash_encode(self, category, salt="my_salt"):
        category = str(category)
        hashed_value = xxhash.xxh64(salt + category).intdigest()
        return hashed_value % self.N

    def vetorize_and_clear_features(self, df):
        binary_features = [
            "contains_ip", "abnormal_url", "shortening_service", "c2_tld", "suspicious_tld",
            "malware_tld", "c2_malicious_tld", "phishing_tld", "sensitive_tld", "contains_suspicious_words"
        ]
        for feature in binary_features:
            if df[feature].isin([0, 1]).all():
                continue
            else:
                print(f"[WARN] Invalid binary value in {feature}, setting to -1")
                df[feature] = df[feature].apply(lambda x: x if x in [0, 1] else -1)

        if not isinstance(df.at[0, 'region'], str) or not df.at[0, 'region'].isalpha():
            print(f"[WARN] Invalid region: {df.at[0, 'region']}, setting to 'unknown'")
            df.at[0, 'region'] = "unknown"

        df_vectorized = df.copy()
        df_vectorized['extract_root_domain_vector'] = df['root_domain'].apply(self.fast_hash_encode)
        df_vectorized['get_url_region_vector'] = df['region'].apply(self.fast_hash_encode)


        df_vectorized.drop(columns=['root_domain', 'region', 'url'], inplace=True, errors='ignore')

        if df_vectorized.empty or df_vectorized.shape[0] == 0:
            print(f"[ERROR] Feature vector is empty after processing")
            return None

        return df_vectorized

    def extract_features(self, url):
        features = {}
        features["url"] = url
        features["is_https"] = self.check_protocol(url)
        features["url_length"] = self.calculate_url_length(url)
        features["num_special_chars"] = self.special_char_count(url)
        features["letters_amount"] = self.count_letters(url)
        features["digit_amount"] = self.count_digits(url)

        features["percent_amount"] = self.count_percent(url)
        features["ques_amount"] = self.count_ques(url)
        features["hyphen_amount"] = self.count_hyphen(url)
        features["equal_amount"] = self.count_equal(url)
        features["dot_amount"] = self.count_dot(url)
        features["www_amount"] = self.count_www(url)
        features["atrate_amount"] = self.count_atrate(url)

        features["contains_ip"] = self.if_contains_ip(url)
        features["abnormal_url"] = self.abnormal_url(url)
        features["shortening_service"] = self.shortening_service(url)

        features["root_domain"] = self.extract_root_domain(url)
        features["domain_length"], features["num_subdomains"] = self.domain_information(url)

        features["c2_tld"] = self.C2_TLD_check(url)
        features["suspicious_tld"] = self.suspicious_TLD_check(url)
        features["malware_tld"] = self.malware_TLD_check(url)
        features["c2_malicious_tld"] = self.C2_malicious_TLD_check(url)
        features["phishing_tld"] = self.phishing_TLD_check(url)
        features["sensitive_tld"] = self.sensitive_TLD_check(url)
        features["contains_suspicious_words"] = self.suspicious_word_check(url)

        features["url_entropy"] = self.calculate_entropy(url)
        features["region"] = self.get_url_region(url)


        return self.vetorize_and_clear_features(pd.DataFrame([features]))



    def calculate_url_length(self, url):
        return len(url)


    def special_char_count(self, url):
        return len(re.findall(r"[/?=&@%-]", url))


    def count_percent(self, url):
        return url.count('%')


    def count_ques(self, url):
        return url.count('?')


    def count_hyphen(self, url):
        return url.count('-')


    def count_equal(self, url):
        return url.count('=')


    def count_dot(self, url):
        count_dot = url.count('.')
        return count_dot


    def count_www(self, url):
        url.count('www')
        return url.count('www')


    def count_atrate(self, url):
        return url.count('@')


    def count_letters(self, url):
        num_letters = sum(char.isalpha() for char in url)
        return num_letters


    def count_digits(self, url):
        num_digits = sum(char.isdigit() for char in url)
        return num_digits


    def if_contains_ip(self, url):
        match = re.search(
            '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
            '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'
            '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'
            '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', str(url))
        if match:
            return 1
        else:
            return 0


    def shortening_service(self, url):
        match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                          'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                          'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                          'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                          'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                          'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                          'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                          'tr\.im|link\.zip\.net',
                          str(url))
        if match:
            return 1
        else:
            return 0


    def abnormal_url(self, url):
        parsed_url = urlparse(url)
        netloc = parsed_url.netloc
        if netloc:
            netloc = str(netloc)
            match = re.search(netloc, url)
            if match:
                return 1
        return 0


    def check_protocol(self, url):
        if "https" in url.lower():
            return 1
        else:
            return 0


    def extract_root_domain(self, url):
        extracted = tldextract.extract(url)
        root_domain = extracted.domain
        return root_domain


    def domain_information(self, url):
        domain_info = tldextract.extract(url)
        return len(domain_info.domain), len(domain_info.subdomain.split(".")) if domain_info.subdomain else 0


    def C2_TLD_check(self, url):
        domain_info = tldextract.extract(url)
        return 1 if f".{domain_info.suffix}" in self.C2_TLDS else 0


    def suspicious_TLD_check(self, url):
        domain_info = tldextract.extract(url)
        return 1 if f".{domain_info.suffix}" in self.SUSPICIOUS_TLDS else 0


    def malware_TLD_check(self, url):
        domain_info = tldextract.extract(url)
        return 1 if f".{domain_info.suffix}" in self.MALWARE_TLDS else 0


    def C2_malicious_TLD_check(self, url):
        domain_info = tldextract.extract(url)
        return 1 if f".{domain_info.suffix}" in self.C2_MALICIOUS_TLDS else 0


    def phishing_TLD_check(self, url):
        domain_info = tldextract.extract(url)
        return 1 if f".{domain_info.suffix}" in self.PHISHING_TLDS else 0


    def sensitive_TLD_check(self, url):
        domain_info = tldextract.extract(url)
        return 1 if f".{domain_info.suffix}" in self.SENSITIVE_TLDS else 0


    def suspicious_word_check(self, url):
        return int(any(word in url.lower() for word in self.SUSPICIOUS_WORDS))


    def calculate_entropy(self, url):
        prob = [float(url.count(c)) / len(url) for c in set(url)]
        return -sum(p * math.log2(p) for p in prob)


    def domain_age(self, url):
        extracted = tldextract.extract(url)
        domain = extracted.domain
        try:
            domain_info = whois.whois(domain)
            creation_date = domain_info.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            age_days = (pd.Timestamp.today() - pd.Timestamp(creation_date)).days if creation_date else -1
            return age_days if age_days > 0 else -1
        except:
            return -1


    def get_url_region(self, url):
        domain_info = tldextract.extract(url)
        primary_domain = f".{domain_info.suffix}"
        for ccTLD in self.ccTLD_to_region:
            if primary_domain.endswith(ccTLD):
                return self.ccTLD_to_region[ccTLD]
        return "Global"