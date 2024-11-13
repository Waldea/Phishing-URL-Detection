    import re
    from urllib.parse import urlparse, urlunparse
    import math
    from collections import Counter
    import aiohttp
    import asyncio
    import socket
    import whois
    import pandas as pd
    from bs4 import BeautifulSoup
    import Levenshtein
    import numpy as np
    from difflib import SequenceMatcher

    class URLFeatureExtractor:
        def __init__(self, url: str, known_phishing_urls=None, live_check_timeout=10, char_repetition_threshold=3,
                 perform_live_check=True, known_brands=None, well_known_domains=None, ref_urls_csv=None):
  

            self.url = self.normalize_url(url)
            self.parsed_url = urlparse(self.url)
            self.live_check_timeout = live_check_timeout
            self.char_repetition_threshold = char_repetition_threshold
            self.perform_live_check = perform_live_check
            self.page_content = None
            self.known_phishing_urls = known_phishing_urls if known_phishing_urls else []
            self.known_brands = known_brands if known_brands else []
            self.well_known_domains = well_known_domains if well_known_domains else []
            self.ref_urls = self.load_reference_urls(ref_urls_csv) if ref_urls_csv else []

            # Example list of known brands
            self.known_brands = [
                "att", "paypal", "microsoft", "dhl", "meta", "irs", "verizon", "mufg", "adobe", "amazon", "apple",
                "wellsfargo", "ebay", "swisspost", "naver", "instagram", "whatsapp", "rakuten", "jreast", "americanexpress",
                "kddi", "office", "chase", "aeon", "optus", "coinbase", "bradesco", "caixa", "jcb", "ing", "hsbc",
                "netflix", "smbc", "nubank", "npa", "allegro", "inpost", "correos", "fedex", "linked", "usps", "google",
                "bankofamerica", "dpd", "itau", "steam", "swisscom", "orange", "google"
            ]

            # Example list of phishing words
            self.phishing_words = [
                "account", "update", "login", "verify", "secure", "bank", "free", "gift", "password"
            ]
        
        def normalize_url(self, url: str) -> str:
            url = url.strip()
            if not re.match(r'^(http|https)://', url):
                if url.startswith('www.'):
                    url = 'http://' + url
                else:
                    url = 'http://' + url
            if re.match(r'^(http|https):/{1,3}', url):
                url = re.sub(r'^(http|https):/{1,}', r'\1://', url)
            parsed_url = urlparse(url)
            scheme = parsed_url.scheme
            netloc = parsed_url.netloc
            path = re.sub(r'/{2,}', '/', parsed_url.path)
            if not netloc:
                netloc = parsed_url.path
                path = ''
            normalized_url = urlunparse((scheme, netloc, path, '', '', ''))
            if normalized_url.endswith('/') and len(urlparse(normalized_url).path) > 1:
                normalized_url = normalized_url[:-1]
            return normalized_url

        async def fetch_page_content(self):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.url, timeout=self.live_check_timeout) as response:
                        if response.status == 200:
                            self.page_content = await response.text()
                        else:
                            self.page_content = None
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                print(f"Error fetching page content for {self.url}: {e}")
                self.page_content = None

        def get_title(self):
            if self.page_content:
                soup = BeautifulSoup(self.page_content, 'html.parser')
                title_tag = soup.find('title')
                return title_tag.get_text().strip() if title_tag else None
            return None
        def has_brand_name_in_domain(self):
        
            domain = self.parsed_url.netloc.lower()
            return any(brand.lower() in domain for brand in self.known_brands)

        def get_tld(self):
            return self.parsed_url.netloc.split('.')[-1] if '.' in self.parsed_url.netloc else None

        def get_domain(self):
            parts = self.parsed_url.netloc.split('.')
            if len(parts) >= 2:
                return parts[-2] + '.' + parts[-1]
            return self.parsed_url.netloc

        def common_phishing_words(self):       
            url_lower = self.url.lower()
            return sum(1 for word in self.phishing_words if word in url_lower)

        def typosquatting_distance(self):
            if self.well_known_domains:
                domain = self.get_domain()
                return min(Levenshtein.distance(domain, known_domain) for known_domain in self.well_known_domains)

        def get_description(self):
            if self.page_content:
                soup = BeautifulSoup(self.page_content, 'html.parser')
                description_tag = soup.find('meta', attrs={'name': 'description'})
                return description_tag['content'].strip() if description_tag and 'content' in description_tag.attrs else None
            return None

        def get_total_links(self):
            if self.page_content:
                soup = BeautifulSoup(self.page_content, 'html.parser')
                return len(soup.find_all('a'))
            return 0

        def get_external_links(self):
            if self.page_content:
                soup = BeautifulSoup(self.page_content, 'html.parser')
                links = soup.find_all('a', href=True)
                external_links = [link['href'] for link in links if self.parsed_url.netloc not in link['href']]
                return len(external_links)
            return 0

        def get_digit_ratio_in_url(self):
            digits = sum(c.isdigit() for c in self.url)
            return digits / len(self.url) if len(self.url) > 0 else 0

        def get_url_length(self):
            return len(self.url)
        
        def get_domain_length(self):
            return len(self.parsed_url.netloc)
        
        def is_https(self):
            return 1 if self.parsed_url.scheme == 'https' else 0
        
        def get_num_subdomains(self):
            return len(self.parsed_url.netloc.split('.')) - 2 if len(self.parsed_url.netloc.split('.')) > 2 else 0
        
        def get_num_subdirectories(self):
            return len([p for p in self.parsed_url.path.split('/') if p]) if self.parsed_url.path else 0
        
        def get_num_query_params(self):
            return len(self.parsed_url.query.split('&')) if self.parsed_url.query else 0
        
        def get_path_length(self):
            return len(self.parsed_url.path)
        
        def get_num_slashes(self):
            return self.parsed_url.path.count('/')
        
        def get_domain_entropy(self):
            domain = self.parsed_url.netloc
            probabilities = [n_x / len(domain) for n_x in Counter(domain).values()]
            return -sum(p * math.log2(p) for p in probabilities)
        
        def char_repetition(self):
            char_counts = Counter(self.url)
            return sum(1 for count in char_counts.values() if count >= self.char_repetition_threshold)
        
        def has_ip_address(self):
            try:
                socket.inet_aton(self.parsed_url.netloc)
                return 1
            except socket.error:
                return 0
        
        def shortened_url(self):
            shortened_domains = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly"]
            return 1 if any(domain in self.parsed_url.netloc for domain in shortened_domains) else 0
        
        def common_phishing_words(self):
            url_lower = self.url.lower()
            return sum(1 for word in self.phishing_words if word in url_lower)
        
        
        def has_hyphen(self):
            return 1 if '-' in self.parsed_url.netloc else 0
        
        def contains_homograph_chars(self):
            homograph_chars = re.compile(r'[а-яА-Яѐ-ӹ]')
            return 1 if homograph_chars.search(self.parsed_url.netloc) else 0

        def has_social_net(self):
            social_networks = ["facebook", "twitter", "instagram", "linkedin", "pinterest"]
            return any(social in self.url.lower() for social in social_networks)
        
        def url_is_random(self):
            entropy = self.get_domain_entropy()
            return entropy > 3.5

        def path_suspicious_keywords(self):
            suspicious_keywords = ["confirm", "secure", "login", "bank", "account", "password"]
            path = self.parsed_url.path.lower()
            return sum(1 for keyword in suspicious_keywords if keyword in path)

        def query_suspicious_keywords(self):
            suspicious_keywords = ["user", "token", "auth", "login", "id"]
            query = self.parsed_url.query.lower()
            return sum(1 for keyword in suspicious_keywords if keyword in query)

        def get_domain_age(self):
            try:
                domain_info = whois.whois(self.parsed_url.netloc)
                if domain_info.creation_date:
                    creation_date = domain_info.creation_date
                    if isinstance(creation_date, list):
                        creation_date = creation_date[0]
                    return (pd.Timestamp.now() - pd.Timestamp(creation_date)).days
                return None
            except Exception as e:
                print(f"WHOIS error for domain age: {e}")
                return None

        def get_days_to_expiry(self):
            try:
                domain_info = whois.whois(self.parsed_url.netloc)
                if domain_info.expiration_date:
                    expiration_date = domain_info.expiration_date
                    if isinstance(expiration_date, list):
                        expiration_date = expiration_date[0]
                    return (pd.Timestamp(expiration_date) - pd.Timestamp.now()).days
                return None
            except Exception as e:
                print(f"WHOIS error for days to expiry: {e}")
                return None

        def get_registration_type(self):
            try:
                domain_info = whois.whois(self.parsed_url.netloc)
                return domain_info.get("registrar", "Unknown")
            except Exception as e:
                print(f"WHOIS error for registration type: {e}")
                return "Unknown"
        
        async def is_website_live(self):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.url, timeout=self.live_check_timeout) as response:
                        return response.status == 200
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                print(f"Error checking if website is live: {e}")
                return False

        async def has_redirect(self):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.url, timeout=self.live_check_timeout, allow_redirects=False) as response:
                        return 300 <= response.status < 400
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                print(f"Error checking if website has redirect: {e}")
                return False

        def load_reference_urls(self, ref_urls_csv: str) -> List[str]:
          """ Load reference URLs from CSV file. """
          try:
              ref_urls_df = pd.read_csv(ref_urls_csv)
              return ref_urls_df['url'].tolist()
          except Exception as e:
              print(f"Error loading reference URLs from {ref_urls_csv}: {e}")
              return []

        # Similarity Feature: Domain Title Match Score
        def url_similarity_score(self, top_n: int = 2) -> float:
            """ Calculate URL similarity score using reference URLs. """
            if not self.ref_urls:
                return 0
            # Initialize vectorizer and fit on reference URLs
            vectorizer = TfidfVectorizer(ngram_range=(4, 16), analyzer="char_wb")
            ref_vectors = vectorizer.fit_transform(self.ref_urls)

            # Cluster the reference URL vectors to get center vectors for comparison
            n_clusters = min(50, len(self.ref_urls))  # Adjust to a maximum of 50 clusters or fewer if fewer reference URLs
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(ref_vectors)
            cluster_centers = kmeans.cluster_centers_

            # Transform current URL
            url_vector = vectorizer.transform([self.url])
            url_vec_1d = url_vector.toarray().flatten()  # Convert to 1D array

            # Calculate similarity scores with cluster centers
            scores = [1 - cosine(url_vec_1d, center.flatten()) for center in cluster_centers]
            top_scores = sorted(scores, reverse=True)[:top_n]
            avg_top_score = np.mean(top_scores)

            return avg_top_score

        # Similarity Feature: Title-Description Similarity
        def title_description_similarity(self):
            title = self.get_title()
            description = self.get_description()
            if title and description:
                return Levenshtein.ratio(title, description)
            return 0

        # Placeholder for title randomness
        def title_is_random(self):
            title = self.get_title()
            if title:
                return len(set(title)) < len(title) * 0.5  # Simple placeholder heuristic for randomness
            return False

        # Placeholder for description randomness
        def description_is_random(self):
            description = self.get_description()
            if description:
                return len(set(description)) < len(description) * 0.5  # Simple placeholder heuristic for randomness
            return False

        def get_similarity_bin(self, score):
            # Define thresholds for bins
            bins = [0, 0.25, 0.5, 0.75, 1.0]
            labels = ["Unlikely", "Less Likely", "Likely", "Very Likely"]
            for i in range(len(bins) - 1):
                if bins[i] <= score < bins[i + 1]:
                    return labels[i]
            return labels[-1]
        
        
        #calculates the similarity score between the domain and the page title, then bins the score into one of the defined categories ("Unlikely", "Less Likely", "Likely", "Very Likely").    
        def title_similarity_bin(self):
            title = self.get_title()
            if not title:
                return "No Title"
            domain = self.get_domain()
            similarity_score = Levenshtein.ratio(domain, title)
            return self.get_similarity_bin(similarity_score)

        
        def description_similarity_bin(self):
            description = self.get_description()
            if not description:
                return "No Description"
            domain = self.get_domain()
            similarity_score = Levenshtein.ratio(domain, description)
            return self.get_similarity_bin(similarity_score)

        def similarity_bin(self):
            # Aggregates the bins from both title_similarity_bin and description_similarity_bin
            # to give an overall similarity assessment
            title_bin = self.title_similarity_bin()
            description_bin = self.description_similarity_bin()
            if title_bin == "Very Likely" or description_bin == "Very Likely":
                return "Very Likely"
            elif title_bin == "Likely" or description_bin == "Likely":
                return "Likely"
            elif title_bin == "Less Likely" or description_bin == "Less Likely":
                return "Less Likely"
            else:
                return "Unlikely"
        def has_www(self):
            return 1 if 'www.' in self.parsed_url.netloc else 0

        def is_expired(self):
            days_to_expiry = self.get_days_to_expiry()
            if days_to_expiry is None or days_to_expiry > 0:
                return False
            return True

        def get_registration_duration(self):
            domain_age = self.get_domain_age()
            days_to_expiry = self.get_days_to_expiry()
            if domain_age is not None and days_to_expiry is not None:
                return domain_age + days_to_expiry
            return 0


        def url_title_match_score(self) -> float:
                """Calculate a match score between the full URL and the title."""
                
                title = self.get_title()
                if not title:
                    return 0.0
                
                try:
                    # Convert URL and title to lowercase for consistency
                    url = self.url.lower()
                    title = title.lower() if title else ""

                    # Remove non-alphanumeric characters for simpler matching
                    url_words = re.findall(r'\b\w+\b', url)
                    title_words = re.findall(r'\b\w+\b', title)

                    # Calculate word overlap ratio
                    common_words = set(url_words) & set(title_words)
                    word_overlap_score = len(common_words) / max(len(url_words), len(title_words), 1)

                    # Calculate string similarity using SequenceMatcher
                    string_similarity_score = SequenceMatcher(None, url, title).ratio()

                    # Combine both scores into an overall match score
                    match_score = 0.5 * word_overlap_score + 0.5 * string_similarity_score

                    return match_score
                except Exception as e:
                    print(f"Error calculating URL-title match score for URL '{url}' and title '{title}': {e}")
                    return 0.0
                
        def calculate_expiration_risk(self):
            days_to_expiry = self.get_days_to_expiry()
            if days_to_expiry is None:
                return "unknown"
            elif days_to_expiry < 30:
                return "high"
            elif days_to_expiry < 90:
                return "medium"
            else:
                return "low"
        async def extract_all_features(self):
            print(f"Extracting features for: {self.url}")
            await self.fetch_page_content()  # Fetch the page content asynchronously before using content-based features
            
            features = {
                'url': self.url,
                'url_length': self.get_url_length(),
                'domain_length': self.get_domain_length(),
                'is_https': self.is_https(),
                'num_subdomains': self.get_num_subdomains(),
                'num_subdirectories': self.get_num_subdirectories(),
                'num_query_params': self.get_num_query_params(),
                'path_length': self.get_path_length(),
                'num_slashes': self.get_num_slashes(),
                'domain_entropy': self.get_domain_entropy(),
                'char_repetition': self.char_repetition(),
                'has_ip_address': self.has_ip_address(),
                'shortened_url': self.shortened_url(),
                'has_hyphen': self.has_hyphen(),
                'contains_homograph_chars': self.contains_homograph_chars(),
                'has_social_net': self.has_social_net(),
                'url_is_random': self.url_is_random(),
                'domain_age': self.get_domain_age() or 0,
                'days_to_expiry': self.get_days_to_expiry() or 0,
                'registration_type': self.get_registration_type() or "Unknown",
                'title': self.get_title() or "No Title",
                'description': self.get_description() or "No Description",
                'total_links': self.get_total_links(),
                'external_links': self.get_external_links(),
                'digit_ratio_in_url': self.get_digit_ratio_in_url(),
                'title_is_random': self.title_is_random(),
                'description_is_random': self.description_is_random(),
                'url_similarity_score': self.url_similarity_score(),
                'url_title_match_score': self.url_title_match_score(),
                'title_description_similarity': self.title_description_similarity(),
                'has_brand_name_in_domain': self.has_brand_name_in_domain(),
                'tld': self.get_tld(),
                'domain': self.get_domain(),
                'common_phishing_words': self.common_phishing_words(),
                'typosquatting_distance': self.typosquatting_distance() or 0,
                'path_suspicious_keywords': self.path_suspicious_keywords(),
                'query_suspicious_keywords': self.query_suspicious_keywords(),
                'title_similarity_bin': self.title_similarity_bin(),
                'description_similarity_bin': self.description_similarity_bin(),
                'similarity_bin': self.similarity_bin(),
                'expiration_risk': self.calculate_expiration_risk(),'is_expired': self.is_expired(), 
                'registration_duration': self.get_registration_duration(),
                'has_www': self.has_www() 
            }
            
            if self.perform_live_check:
                features['is_website_live'] = await self.is_website_live()
                features['has_redirect'] = await self.has_redirect()

            return features

#Example usage

test_urls = [
    "example.com",
    "https:///example.com",
    "http://example.com//path",
    "  www.example.com  ",
    "https://www.example.com////",
    "example.com/path/to/resource",
    "google.com",
    "facebook.com"
]

# Run the main function to test and create DataFrame
#feature_list = []
#for url in test_urls:
    #extractor = URLFeatureExtractor(url, live_check_timeout=30, char_repetition_threshold=4, perform_live_check=True, ref_urls_csv = "/content/ref_urls.csv")
    #features = await extractor.extract_all_features()
    #feature_list.append(features)

# Create a DataFrame to visualize the features
#df = pd.DataFrame(feature_list)
#df

        