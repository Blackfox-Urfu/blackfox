import praw
import requests
import os
import time
import dotenv
import html
from tqdm import tqdm
import sys 

# --- Конфигурация ---
dotenv.load_dotenv()
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
USER_AGENT = os.getenv('USER_AGENT')

SFW_SUBREDDITS = []
NSFW_SUBREDDITS = ['SEXY_PICS_NSFW','Selfies_NSFW','HighResNSFW','nsfwpicsmenwomen']

SFW_SAVE_PATH = "data/reddit/sfw_images"
NSFW_SAVE_PATH = "data/reddit/nsfw_images"
os.makedirs(SFW_SAVE_PATH, exist_ok=True)
os.makedirs(NSFW_SAVE_PATH, exist_ok=True)

DOWNLOAD_LIMIT_PER_SUBREDDIT = 5000
DOWNLOAD_TIMEOUT = 25

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT,
)

def sanitize_filename(filename_part):
    return "".join(c for c in filename_part if c.isalnum() or c in (' ', '.', '_')).rstrip()

def download_image(url, target_filepath, pbar_instance=None):
    try:
        response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()

        content_type = response.headers.get('content-type')
        if not content_type or not content_type.startswith('image/'):
            if any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                message = f"[Warning] URL {url} looks like an image but Content-Type is {content_type}. Proceeding anyway."
                if pbar_instance: pbar_instance.write(f"  {message}")
                else: print(f"  {message}")
            else:
                message = f"[Skipping] URL {url} is not an image (Content-Type: {content_type})"
                if pbar_instance: pbar_instance.write(f"  {message}")
                else: print(f"  {message}")
                return False

        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)

        base_name = os.path.basename(target_filepath)
        if len(base_name) > 200:
            name_part, ext_part = os.path.splitext(base_name)
            max_name_len = 200 - len(ext_part)
            new_name_part = name_part[:max_name_len]
            target_filepath = os.path.join(os.path.dirname(target_filepath), f"{new_name_part}{ext_part}")
            message = f"[Warning] Original filename too long, shortened to: {os.path.basename(target_filepath)}"
            if pbar_instance: pbar_instance.write(f"  {message}")
            else: print(f"  {message}")

        with open(target_filepath, 'wb') as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)
        return True
    except requests.exceptions.Timeout:
        message = f"[Error] Timeout downloading {url}"
        if pbar_instance: pbar_instance.write(f"  {message}")
        else: print(f"  {message}")
        return False
    except requests.exceptions.RequestException as e:
        message = f"[Error] Could not download {url}: {e}"
        if pbar_instance: pbar_instance.write(f"  {message}")
        else: print(f"  {message}")
        return False
    except Exception as e:
        message = f"[Error] An unexpected error occurred with {url} to {target_filepath}: {e}"
        if pbar_instance: pbar_instance.write(f"  {message}")
        else: print(f"  {message}")
        return False

def fetch_and_download(subreddit_names, save_path_base, is_nsfw_collection):
    for sub_name in tqdm(subreddit_names, desc="Overall Subreddits Progress", unit="sub", position=0, leave=True):
        tqdm.write(f"\nFetching from r/{sub_name}...")
        subreddit_save_path = os.path.join(save_path_base, sub_name)
        os.makedirs(subreddit_save_path, exist_ok=True)

        downloaded_count_for_subreddit = 0
        processed_posts_count = 0
        estimated_posts_to_scan = DOWNLOAD_LIMIT_PER_SUBREDDIT * 10

        with tqdm(total=DOWNLOAD_LIMIT_PER_SUBREDDIT, desc=f"r/{sub_name}", unit="img", position=1, leave=False) as pbar_images:
            try:
                for submission in reddit.subreddit(sub_name).hot(limit=None):
                    if downloaded_count_for_subreddit >= DOWNLOAD_LIMIT_PER_SUBREDDIT:
                        break

                    processed_posts_count += 1
                    if processed_posts_count > estimated_posts_to_scan and downloaded_count_for_subreddit < DOWNLOAD_LIMIT_PER_SUBREDDIT:
                        pbar_images.write(f"  [INFO] r/{sub_name}: Reached scan limit ({estimated_posts_to_scan} posts) "
                                        f"but only found {downloaded_count_for_subreddit} images. Moving to next subreddit.")
                        break

                    if not is_nsfw_collection and submission.over_18:
                        continue

                    images_downloaded_this_submission = 0

                    # Проверяем наличие атрибута is_gallery безопасным способом
                    if hasattr(submission, 'is_gallery') and submission.is_gallery and hasattr(submission, 'media_metadata') and submission.media_metadata:
                        for media_id, media_item in submission.media_metadata.items():
                            if downloaded_count_for_subreddit >= DOWNLOAD_LIMIT_PER_SUBREDDIT: 
                                break

                            if media_item.get('e') == 'Image':
                                image_url = None
                                if 'u' in media_item.get('s', {}):
                                    image_url = html.unescape(media_item['s']['u'])
                                elif 'gif' in media_item.get('s', {}):
                                    image_url = html.unescape(media_item['s']['gif'])
                                else:
                                    largest_res = 0
                                    best_url = None
                                    for res_info in media_item.get('p', []):
                                        if res_info.get('u') and (res_info.get('x', 0) * res_info.get('y', 0) > largest_res):
                                            largest_res = res_info.get('x', 0) * res_info.get('y', 0)
                                            best_url = html.unescape(res_info['u'])
                                    if best_url: 
                                        image_url = best_url
                                    else: 
                                        continue

                                mime_type = media_item.get('m', '')
                                extension = "." + mime_type.split('/')[-1] if '/' in mime_type else os.path.splitext(requests.utils.urlparse(image_url).path)[1]
                                if not extension or len(extension) > 5 or len(extension) < 3: 
                                    extension = ".jpg"

                                filename_base = f"{submission.id}_{media_id}"
                                target_filename = f"{filename_base}{extension}"
                                full_target_path = os.path.join(subreddit_save_path, target_filename)

                                if download_image(image_url, full_target_path, pbar_images):
                                    pbar_images.update(1)
                                    downloaded_count_for_subreddit += 1
                                    images_downloaded_this_submission += 1
                                    time.sleep(0.1)

                    elif hasattr(submission, "url") and submission.url:
                        image_url = submission.url
                        parsed_url = requests.utils.urlparse(image_url)
                        path_part = parsed_url.path
                        _, extension = os.path.splitext(path_part)
                        is_direct_link_domain = any(domain in parsed_url.hostname for domain in ['i.redd.it', 'i.imgur.com'])

                        if extension.lower() in ['.jpg', '.jpeg', '.png', '.gif'] or is_direct_link_domain:
                            if not extension and is_direct_link_domain: 
                                extension = ".jpg"
                            elif not extension: 
                                continue

                            original_filename_part = sanitize_filename(os.path.basename(path_part))
                            if original_filename_part and '.' in original_filename_part and len(original_filename_part) <= 100:
                                target_filename = f"{submission.id}_{original_filename_part}"
                            else:
                                target_filename = f"{submission.id}{extension if extension else '.jpg'}"
                            full_target_path = os.path.join(subreddit_save_path, target_filename)

                            if download_image(image_url, full_target_path, pbar_images):
                                pbar_images.update(1)
                                downloaded_count_for_subreddit += 1
                                images_downloaded_this_submission += 1

                    if images_downloaded_this_submission > 0:
                        time.sleep(0.2)

            except praw.exceptions.PRAWException as e:
                tqdm.write(f"  [PRAW Error] Could not fetch from r/{sub_name}: {e}")
            except Exception as e:
                tqdm.write(f"  [Error] An unexpected error occurred with r/{sub_name}: {e}")
                import traceback
                tqdm.write(traceback.format_exc())

        tqdm.write(f"Finished r/{sub_name}, downloaded {downloaded_count_for_subreddit} images from {processed_posts_count} processed posts.\n")

if __name__ == "__main__":
    print("--- Starting SFW Image Collection ---")
    fetch_and_download(SFW_SUBREDDITS, SFW_SAVE_PATH, is_nsfw_collection=False)

    print("\n\n--- Starting NSFW Image Collection (USE RESPONSIBLY!) ---")
    fetch_and_download(NSFW_SUBREDDITS, NSFW_SAVE_PATH, is_nsfw_collection=True)

    print("\n--- Collection Finished ---")