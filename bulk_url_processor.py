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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
from datetime import datetime, timezone
import concurrent.futures
from typing import List
import ssl




# Apply nest_asyncio if you're in Jupyter or Colab
import nest_asyncio
nest_asyncio.apply()


class URLFeatureExtractor:
    def __init__(
            self,
            url: str,
            known_phishing_urls=None,
            live_check_timeout=10,
            char_repetition_threshold=3,
            perform_live_check=True,
            known_brands=None,
            well_known_domains=None,
            ref_urls_csv=None,
            session=None,
            max_concurrent_requests=10,
            batch_size=None,
            max_retries=1,
            request_timeout=10):

        self.url = self.normalize_url(url)
        self.normalized_url = self.url
        self.parsed_url = urlparse(self.url)
        self.live_check_timeout = live_check_timeout
        self.char_repetition_threshold = char_repetition_threshold
        self.perform_live_check = perform_live_check
        self.page_content = None
        self.known_phishing_urls = known_phishing_urls if known_phishing_urls else []
        self.known_brands = known_brands if known_brands else []
        self.well_known_domains = well_known_domains if well_known_domains else []
        self.ref_urls = ref_urls_csv if ref_urls_csv else []
        self.session = session
        self.own_session = session is None
        self.max_concurrent_requests = max_concurrent_requests
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.own_session = False
        self.vectorizer = None
        self.cluster_centers = None
        
        if self.ref_urls:
            self.vectorizer = TfidfVectorizer(ngram_range=(4, 16), analyzer="char_wb")
            ref_vectors = self.vectorizer.fit_transform(self.ref_urls)

            # Fit KMeans to reference vectors
            n_clusters = min(50, len(self.ref_urls))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(ref_vectors)
            self.cluster_centers = kmeans.cluster_centers_

        if not self.session:
            self.session = aiohttp.ClientSession()
            self.own_session = True

        # Example list of known brands
        self.known_brands = [
            "att",
            "paypal",
            "microsoft",
            "dhl",
            "meta",
            "irs",
            "verizon",
            "mufg",
            "adobe",
            "amazon",
            "apple",
            "wellsfargo",
            "ebay",
            "swisspost",
            "naver",
            "instagram",
            "whatsapp",
            "rakuten",
            "jreast",
            "americanexpress",
            "kddi",
            "office",
            "chase",
            "aeon",
            "optus",
            "coinbase",
            "bradesco",
            "caixa",
            "jcb",
            "ing",
            "hsbc",
            "netflix",
            "smbc",
            "nubank",
            "npa",
            "allegro",
            "inpost",
            "correos",
            "fedex",
            "linked",
            "usps",
            "google",
            "bankofamerica",
            "dpd",
            "itau",
            "steam",
            "swisscom",
            "orange"]

        # Example list of phishing words
        self.phishing_words = [
            "account",
            "update",
            "login",
            "verify",
            "secure",
            "bank",
            "free",
            "gift",
            "password"]
        
        self.suspicious_keywords = ["confirm", "secure", "login", "bank", "account", "password"]

    async def close_session(self):
        if self.own_session and self.session:
            await self.session.close()
    # URL Normalization
    def normalize_url(self, url: str) -> str:
        url = url.strip()
        if not re.match(r'^(http|https)://', url):
            if url.startswith('www.'):
                url = 'https://' + url
            else:
                url = 'https://' + url
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
        if normalized_url.endswith(
                '/') and len(urlparse(normalized_url).path) > 1:
            normalized_url = normalized_url[:-1]
        return normalized_url
    
    
    async def determine_best_scheme(self):
        schemes = ['https://', 'http://']
        for scheme in schemes:
            full_url = f"{scheme}{self.parsed_url.netloc}{self.parsed_url.path}"
            try:
                async with self.session.get(full_url, timeout=self.live_check_timeout) as response:
                    if response.status == 200:
                        # Update the normalized URL if scheme is determined successfully
                        self.url = full_url
                        self.parsed_url = urlparse(self.url)
                        return full_url
            except Exception as e:
                # Ignore errors to continue with the next scheme
                continue
        # If none worked, keep the original normalized URL
        return self.normalized_url
    
    
    # Network-Based Features
    async def fetch_multiple_pages(self, urls):
        """
        Fetch multiple pages concurrently given a list of URLs.
        Uses the default configurations set during initialization.

        :param urls: List of URLs to fetch.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async with aiohttp.ClientSession() as session:
            batch_size = self.batch_size if self.batch_size is not None else len(urls)

            tasks = []
            for i in range(0, len(urls), batch_size):
                batch_urls = urls[i:i + batch_size]
                for url in batch_urls:
                    task = self.fetch_with_retries(url, session, self.max_retries, self.request_timeout, semaphore)
                    tasks.append(task)

                # Gather tasks for the current batch
                results = await asyncio.gather(*tasks, return_exceptions=True)
                tasks.clear()

                # Process results here if needed
                for url, content in zip(batch_urls, results):
                    if isinstance(content, Exception):
                        print(f"Failed to fetch {url} due to {content[:50]}")
                    else:
                        print(f"Content fetched from {url}: {content[:50]}...")  # Print first 100 chars

    

    async def fetch_with_retries(self, url, session, max_retries, timeout, semaphore):
        retries = 0
        while retries < max_retries:
            async with semaphore:
                try:
                    async with session.get(url, timeout=timeout) as response:
                        if response.status == 200:
                            return await response.text()
                        else:
                            print(f"Non-200 status code received: {response.status} for URL {url}")
                            retries += 1
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    print(f"Error fetching page content for {url}: {e}")
                    retries += 1
        return None



    async def fetch_page_content(self):
        await self.determine_best_scheme()
        if not self.session:
            raise ValueError("Session not set. Please set a shared session.")

        try:
            async with self.session.get(self.url, timeout=self.live_check_timeout) as response:
                print(
                    f"Fetching URL: {self.url}, Status Code: {response.status}")
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')
                    if 'charset' in content_type:
                        encoding = content_type.split('charset=')[-1].split(';')[0].strip()
                    else:
                        encoding = 'utf-8'
                    try:
                        self.page_content = await response.text(encoding=encoding)
                    except UnicodeDecodeError:
                        # Fallback to ISO-8859-1 or ignore errors if utf-8 decoding fails
                        self.page_content = await response.text(encoding='ISO-8859-1', errors='ignore')
                    except UnicodeDecodeError:
                        self.page_content = await response.text(errors='ignore')
                      

                else:
                    self.page_content = None
                    print(f"Non-200 status code received: {response.status}")
        except (aiohttp.ClientConnectorError, aiohttp.ClientError, asyncio.TimeoutError, ssl.SSLError) as e:
            print(f"Error fetching page content for {self.url}: {e}")
            self.page_content = None


    async def is_website_live(self):
        try:
            async with self.session.get(self.url, timeout=self.live_check_timeout) as response:
                return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"Error checking if website is live: {e}")
            return False

    async def has_redirect(self):
        try:
            async with self.session.get(self.url, timeout=self.live_check_timeout, allow_redirects=False) as response:
                return 300 <= response.status < 400
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"Error checking if website has redirect: {e}")
            return False

    # Content-Based Features
    def get_title(self):
        if self.page_content:
            soup = BeautifulSoup(self.page_content, 'html.parser')
            title_tag = soup.find('title')
            return title_tag.get_text().strip() if title_tag else None
        return None

    def get_description(self):
        if self.page_content:
            soup = BeautifulSoup(self.page_content, 'html.parser')
            description_tag = soup.find('meta', attrs={'name': 'description'})
            return description_tag['content'].strip(
            ) if description_tag and 'content' in description_tag.attrs else None
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
            external_links = [
                link['href'] for link in links if self.parsed_url.netloc not in link['href']]
            return len(external_links)
        return 0

    @staticmethod
    # Domain and Security Features
    def get_domain_age_sync(domain):
        try:
            domain_info = whois.whois(domain)
            creation_date = domain_info.creation_date
            
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            if creation_date:
                if creation_date.tzinfo is None:
                    creation_date = creation_date.replace(tzinfo=timezone.utc)
                current_date = datetime.now(timezone.utc)  # Make timezone-aware
                return (current_date - creation_date).days if creation_date else None
            return None
        except Exception as e:
            print(f"WHOIS error for domain age: {e}")
            return None
        
    async def get_domain_age(self):
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, self.get_domain_age_sync, self.parsed_url.netloc)

    @staticmethod
    def get_days_to_expiry_sync(domain):
        try:
            domain_info = whois.whois(domain)
            expiration_date = domain_info.expiration_date
            if isinstance(expiration_date, list):
                expiration_date = expiration_date[0]

            if expiration_date:
              if expiration_date.tzinfo is None:
                expiration_date = expiration_date.replace(tzinfo=timezone.utc)
              current_date = datetime.now(timezone.utc)  # Make timezone-aware
              return (expiration_date - current_date).days 
            return None
        except Exception as e:
            print(f"WHOIS error for days to expiry: {e}")
            return None

    async def get_days_to_expiry(self):
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool,self.get_days_to_expiry_sync, self.parsed_url.netloc)

    def get_registration_type(self):
        try:
            domain_info = whois.whois(self.parsed_url.netloc)
            return domain_info.get("registrar", "Unknown")
        except Exception as e:
            print(f"WHOIS error for registration type: {e}")
            return "Unknown"

    def has_ip_address(self):
        try:
            socket.inet_aton(self.parsed_url.netloc)
            return 1
        except socket.error:
            return 0

    def get_domain(self):
        return self.parsed_url.netloc

    def get_tld(self):
        return self.parsed_url.netloc.split('.')[-1] if '.' in self.parsed_url.netloc else None

    # Similarity Feature: Title-Description Similarity
    def title_description_similarity(self):
        title = self.get_title()
        description = self.get_description()
        if title and description:
            return Levenshtein.ratio(title, description)
        return 0

    # Phishing and Suspicious Features
    def common_phishing_words(self):
        url_lower = self.url.lower()
        return sum(1 for word in self.phishing_words if word in url_lower)

    def typosquatting_distance(self):
        if self.well_known_domains:
            domain = self.get_domain()
            return min(Levenshtein.distance(domain, known_domain)
                       for known_domain in self.well_known_domains)

    def contains_homograph_chars(self):
        homograph_chars = re.compile(r'[а-яА-Яѐ-ӹ]')
        return 1 if homograph_chars.search(self.parsed_url.netloc) else 0

    def has_brand_name_in_domain(self):
        domain = self.parsed_url.netloc.lower()
        return any(brand.lower() in domain for brand in self.known_brands)

    # URL Parsing and Basic Features
    def get_url_length(self) -> int:
        return len(self.url)

    def get_domain_length(self):
        return len(self.parsed_url.netloc)

    def is_https(self):
        return 1 if self.parsed_url.scheme == 'https' else 0

    def get_num_subdomains(self):
        components = self.parsed_url.netloc.split('.')
        # Remove 'www' from components if it exists
        components = [comp for comp in components if comp.lower() != 'www']
        return len(components) - 2 if len(components) > 2 else 0

    def get_num_subdirectories(self):
        return len([p for p in self.parsed_url.path.split('/') if p]
                   ) if self.parsed_url.path else 0

    def get_num_query_params(self):
        return len(self.parsed_url.query.split('&')
                   ) if self.parsed_url.query else 0

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
        return sum(count for count in char_counts.values()
                   if count >= self.char_repetition_threshold)

    def shortened_url(self):
        shortened_domains = [
            "bit.ly",
            "tinyurl.com",
            "goo.gl",
            "t.co",
            "ow.ly",
            "is.gd",
            "buff.ly"]
        return 1 if any(
            domain in self.parsed_url.netloc for domain in shortened_domains) else 0

    def get_digit_ratio_in_url(self):
        digits = sum(c.isdigit() for c in self.url)
        return digits / len(self.url) if len(self.url) > 0 else 0

    def has_hyphen(self):
        return 1 if '-' in self.parsed_url.netloc else 0

    def has_social_net(self):
        social_networks = [
            "facebook",
            "twitter",
            "instagram",
            "linkedin",
            "pinterest"]
        return any(social in self.url.lower() for social in social_networks)

    def url_is_random(self):
        entropy = self.get_domain_entropy()
        return entropy > 3.5

    def path_suspicious_keywords(self): 
        path = self.parsed_url.path.lower()
        return sum(path.count(keyword) for keyword in self.suspicious_keywords)

    def query_suspicious_keywords(self):
        query = self.parsed_url.query.lower()
        return sum(query.count(keyword) for keyword in self.suspicious_keywords)

    def title_is_random(self):
        title = self.get_title()
        if title:
            # Simple placeholder heuristic for randomness
            return len(set(title)) < len(title) * 0.5
        return False

    def description_is_random(self):
        description = self.get_description()
        if description:
            # Simple placeholder heuristic for randomness
            return len(set(description)) < len(description) * 0.5
        return False

    def load_reference_urls(self, ref_urls_csv: str) -> list:
        """ Load reference URLs from CSV file. """
        try:
            ref_urls_df = pd.read_csv(ref_urls_csv)
            return ref_urls_df['url'].tolist()
        except Exception as e:
            print(f"Error loading reference URLs from {ref_urls_csv}: {e}")
            return []

    def url_title_match_score(self) -> float:
        """Calculate a match score between the full URL and the title."""
        title = self.get_title()
        if not title:
            return 0.0

        try:
            # Convert URL and title to lowercase for consistency
            url = self.url.lower()
            title = title.lower()

            # Remove non-alphanumeric characters for simpler matching
            url_words = re.findall(r'\b\w+\b', url)
            title_words = re.findall(r'\b\w+\b', title)

            # Calculate word overlap ratio
            common_words = set(url_words) & set(title_words)
            word_overlap_score = len(common_words) / \
                max(len(url_words), len(title_words), 1)

            # Calculate string similarity using SequenceMatcher
            string_similarity_score = SequenceMatcher(None, url, title).ratio()

            # Combine both scores into an overall match score
            match_score = 0.5 * word_overlap_score + 0.5 * string_similarity_score

            return match_score
        except Exception as e:
            print(
                f"Error calculating URL-title match score for URL '{url}' and title '{title}': {e}")
            return 0.0



    def url_similarity_score(self, top_n: int = 2) -> float:
        """ Calculate URL similarity score using precomputed reference URLs. """
        if not self.ref_urls or not self.vectorizer or self.cluster_centers is None:
            return 0

        try:
            # Transform the current URL using the fitted vectorizer
            url_vector = self.vectorizer.transform([self.url])
            url_vec_1d = url_vector.toarray().flatten()  # Convert to 1D array

            # Calculate similarity scores with cluster centers
            scores = [1 - cosine(url_vec_1d, center.flatten()) for center in self.cluster_centers]
            top_scores = sorted(scores, reverse=True)[:top_n]
            avg_top_score = np.mean(top_scores)

            return avg_top_score

        except NotFittedError as e:
            print(f"Error calculating URL similarity: {e}")
            return 0

    

    async def calculate_expiration_risk(self):
        days_to_expiry = await self.get_days_to_expiry()
        if days_to_expiry is None:
            return "unknown"
        elif days_to_expiry < 30:
            return "high"
        elif days_to_expiry < 90:
            return "medium"
        else:
            return "low"

    def get_similarity_bin(self, score):
        # Define thresholds for bins
        bins = [0, 0.25, 0.5, 0.75, 1.0]
        labels = ["Unlikely", "Less Likely", "Likely", "Very Likely"]
        for i in range(len(bins) - 1):
            if bins[i] <= score < bins[i + 1]:
                return labels[i]
        return labels[-1]

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
        
    

    async def is_expired(self):
        days_to_expiry = await self.get_days_to_expiry()
        if days_to_expiry is None or days_to_expiry > 0:
            return False
        return True

    async def get_registration_duration(self):
        domain_age = await self.get_domain_age()
        days_to_expiry = await self.get_days_to_expiry()
        print(f"Domain Age: {domain_age}, Days to Expiry: {days_to_expiry}")
        if domain_age is not None and days_to_expiry is not None:
            return domain_age + days_to_expiry
        return None
    
    def validate_features(self, features):
        missing_features = [key for key, value in features.items() if value is None]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")




    # Extract All Features
    async def extract_all_features(self):
        print(f"Extracting features for: {self.url}")
        # Fetch the page content asynchronously before using content-based
        # features
        await self.determine_best_scheme()
        await self.fetch_page_content()
        

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
            'domain_age': await self.get_domain_age() or 0,
            'days_to_expiry': await self.get_days_to_expiry() or 0,
            'registration_type': self.get_registration_type() or "Unknown",
            'title': self.get_title() or "No Title",
            'description': self.get_description() or "No Description",
            'total_links': self.get_total_links(),
            'external_links': self.get_external_links(),
            'digit_ratio_in_url': self.get_digit_ratio_in_url(),
            'title_is_random': self.title_is_random(),
            'description_is_random': self.description_is_random(),
            'has_brand_name_in_domain': self.has_brand_name_in_domain(),
            'tld': self.get_tld(),
            'domain': self.get_domain(),
            'common_phishing_words': self.common_phishing_words(),
            'typosquatting_distance': self.typosquatting_distance() or 0,
            'path_suspicious_keywords': self.path_suspicious_keywords(),
            'query_suspicious_keywords': self.query_suspicious_keywords(),
            'title_description_similarity': self.title_description_similarity(),
            'url_title_match_score': self.url_title_match_score(),
            'url_similarity_score': self.url_similarity_score(),
            'title_similarity_bin': self.title_similarity_bin(),
            'description_similarity_bin': self.description_similarity_bin(),
            'similarity_bin': self.similarity_bin(),
            'expiration_risk': await self.calculate_expiration_risk(),
            'is_expired': await self.is_expired(),
            'registration_duration': await self.get_registration_duration(),
        }

        if self.perform_live_check:
            features['is_website_live'] = await self.is_website_live()
            features['has_redirect'] = await self.has_redirect()
           
        self.validate_features(features)     
        await self.close_session()
        return features
'''
Example Usage:
import pandas as pd
import asyncio
import aiohttp
import nest_asyncio

# Apply nest_asyncio to allow for re-entrance in the event loop in Jupyter/Colab
nest_asyncio.apply()

# Async function to extract features concurrently with batch handling
async def extract_features_in_batches(urls, ref_urls, batch_size=500, max_concurrent_requests=300, max_retries=0, request_timeout=10):
    extracted_features = []

    # Create a shared session for all requests to avoid creating a new session each time
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(urls), batch_size):
            # Get the current batch of URLs
            batch_urls = urls[i:i + batch_size]

            # Semaphore to limit concurrency for this batch
            semaphore = asyncio.Semaphore(max_concurrent_requests)

            # Create tasks for the current batch
            tasks = []
            for url in batch_urls:
                extractor = URLFeatureExtractor(
                    url=url,
                    ref_urls_csv=ref_urls,
                    session=session,
                    perform_live_check=True,
                    max_concurrent_requests=max_concurrent_requests,
                    max_retries=max_retries,
                    request_timeout=request_timeout
                )
                task = extract_features_for_url(extractor, semaphore)
                tasks.append(task)

            # Gather all tasks in the current batch
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results for the current batch
            for features in results:
                if isinstance(features, Exception):
                    print(f"Error during feature extraction: {features}")
                else:
                    extracted_features.append(features)

    # Convert the list of feature dictionaries into a DataFrame
    features_df = pd.DataFrame(extracted_features)
    return features_df

# Helper function to extract features for a single URL using the semaphore
async def extract_features_for_url(extractor, semaphore):
    # Semaphore to limit the number of concurrent requests
    async with semaphore:
        try:
            # Extract features using the asynchronous method
            features = await extractor.extract_all_features()
        except Exception as e:
            print(f"Error extracting features for {extractor.url}: {e}")
            features = None
        return features

# Example to run the async feature extraction
if __name__ == "__main__":
    # Assuming `data_sample` is your DataFrame with URLs
    urls = data_sample['url'].tolist()  # List of URLs
    ref_urls = ref_urls  # Replace with the actual list or path to CSV containing reference URLs

    # Run the async feature extraction using asyncio.run() for compatibility
    features_df = asyncio.run(
        extract_features_in_batches(
            urls=urls,
            ref_urls=ref_urls,
            batch_size=1000,  # Number of URLs to process in each batch
            max_concurrent_requests=300,  # Max number of concurrent requests
            max_retries=1,  # Max retries for failed requests
            request_timeout=10  # Timeout for requests
        )
    )

    # Save to CSV or perform any other operations
    features_df.to_csv('extracted_features.csv', index=False)
    features_df.head()
'''