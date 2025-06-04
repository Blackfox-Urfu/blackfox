import praw
import os
import time
import dotenv
from tqdm import tqdm
import json
from datetime import datetime
import sys

# --- Конфигурация ---
dotenv.load_dotenv()
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
USER_AGENT = os.getenv('USER_AGENT')

# Субреддиты для парсинга
RUSSIAN_SUBREDDITS = [
    'Pikabu',
    'liberta',
    'epicentr',
    'RUbook',
    'Moscow',
    'SPb',
    'Kafka'
]

# Настройки сохранения
SAVE_PATH = "data/reddit/text_posts"
os.makedirs(SAVE_PATH, exist_ok=True)

# Лимиты
POST_LIMIT_PER_SUBREDDIT = 100000000
COMMENT_LIMIT_PER_POST = 1000
REQUEST_DELAY = 2  # Задержка между запросами

# Инициализация Reddit API
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT,
)

def sanitize_text(text):
    """Очистка текста от спецсимволов и лишних пробелов"""
    if not text:
        return ""
    return ' '.join(text.replace('\n', ' ').replace('\t', ' ').split())

def process_comments(comments, limit=COMMENT_LIMIT_PER_POST):
    """Обработка комментариев к посту"""
    processed_comments = []
    comments.replace_more(limit=0)
    
    for comment in comments.list()[:limit]:
        if not comment.body or comment.body == '[deleted]':
            continue
            
        processed_comments.append({
            'id': comment.id,
            'author': str(comment.author) if comment.author else '[deleted]',
            'created_utc': comment.created_utc,
            'score': comment.score,
            'text': sanitize_text(comment.body),
            'is_submitter': comment.is_submitter,
            'replies_count': len(comment.replies)
        })
    
    return processed_comments

def fetch_posts_and_write(subreddit_name):
    """Сбор постов сабреддита с немедленной записью в один JSON-файл"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{subreddit_name}_{timestamp}.json"
    filepath = os.path.join(SAVE_PATH, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('[\n')  # Открытие JSON-массива
        first_post = True

        subreddit = reddit.subreddit(subreddit_name)
        try:
            posts = subreddit.hot(limit=POST_LIMIT_PER_SUBREDDIT)
            
            for post in tqdm(posts, desc=f"r/{subreddit_name}", unit="post"):
                if post.stickied:
                    continue
                
                post_data = {
                    'subreddit': subreddit_name,
                    'post_id': post.id,
                    'title': sanitize_text(post.title),
                    'author': str(post.author) if post.author else '[deleted]',
                    'created_utc': post.created_utc,
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments,
                    'url': post.url,
                    'is_self': post.is_self,
                    'text': sanitize_text(post.selftext),
                    'flair': post.link_flair_text,
                    'nsfw': post.over_18,
                    'comments': []
                }
                
                if post.num_comments > 0:
                    try:
                        post.comments.replace_more(limit=0)
                        post_data['comments'] = process_comments(post.comments)
                    except Exception as e:
                        tqdm.write(f"  [Error] Ошибка при получении комментариев к посту {post.id}: {e}")
                
                # Запись поста в JSON-файл
                if not first_post:
                    f.write(',\n')
                else:
                    first_post = False
                json.dump(post_data, f, ensure_ascii=False, indent=2)
                f.flush()  # Мгновенная запись на диск
                
                tqdm.write(f"  [Saved] Пост {post.id} сохранен")
                time.sleep(REQUEST_DELAY)

        except Exception as e:
            tqdm.write(f"  [Critical Error] Ошибка при обработке r/{subreddit_name}: {e}")

        f.write('\n]')  # Закрытие JSON-массива

    tqdm.write(f"  [Done] Все посты сохранены в {filepath}")

def main():
    print("=== Парсер русскоязычных субреддитов ===")
    print(f"Всего сабреддитов для обработки: {len(RUSSIAN_SUBREDDITS)}\n")
    
    for subreddit in RUSSIAN_SUBREDDITS:
        print(f"\nНачинаем сбор данных с r/{subreddit}...")
        fetch_posts_and_write(subreddit)
        print(f"  Завершен сбор данных с r/{subreddit}")
        time.sleep(5)
    
    print("\n=== Завершено ===")

if __name__ == "__main__":
    main()
