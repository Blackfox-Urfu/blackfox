import queue
import threading
import praw
import requests
import os
import time
import dotenv
import html
from tqdm import tqdm
import sys 
from concurrent.futures import ThreadPoolExecutor
from prawcore.exceptions import TooManyRequests

# --- Конфигурация ---
dotenv.load_dotenv()
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
USER_AGENT = os.getenv('USER_AGENT')


SFW_SUBREDDITS = []
NSFW_SUBREDDITS = [
    'lactation',
    'tinytits',
    'aa_cups',
    'braceface',
    'earspokingout',
    'GirlswithNeonHair',
    'shorthairchicks',
    'stockings',
    'legs',
    'tightshorts',
    'buttsandbarefeet',
    'datgap',
    'thighhighs',
    'thickthighs',
    'rearpussy',
    'innie',
    'simps',
    'pelfie',
    'godpussy',
    'presenting',
    'hairypussy',
    'lipsthatgrip',
    'fucklicking',
    'moundofvenus',
    'pussymound',
    'Hotchickswithtattoos',
    'sexyfrex',
    'tanlines',
    'ComplexionExcellence',
    'SexyTummies',
    'theratio',
    'fitgirls',
    'bodyperfection',
    'samespecies',
    'athleticgirls',
    'fitgirlsfucking',
    'juicyasians',
    'voluptuous',
    'jigglefuck',
    'SlimThick',
    'massivetitsnass',
    'thicker',
    'tightsqueeze',
    'casualjiggles',
    'dirtysmall',
    'xsmallgirls',
    'funsized',
    'Athlete',
    'volleyballgirls',
    'Ohlympics',
    'celebnsfw',
    'WatchItForThePlot',
    'extramile',
    'onoffcelebs',
    'GirlsFinishingTheJob',
    'cumfetish',
    'cumcoveredfucking',
    'cumhaters',
    'thickloads',
    'before_after_cumsluts',
    'pulsatingcumshots',
    'impressedbycum',
    'creampies',
    'throatpies',
    'FacialFun',
    'cumonclothes',
    'oralcreampie',
    'HappyEmbarrassedGirls',
    'borednignored',
    'annoyedtobenude',
    'damngoodinterracial',
    'AsianHotties',
    'realasians',
    'AsianNSFW',
    'asianporn',
    'bustyasians',
    'IndianBabes',
    'NSFW_Japan',
    'kpopfap',
    'WomenOfColor',
    'darkangels',
    'blackchickswhitedicks',
    'Afrodisiac',
    'ginger',
    'latinacuties',
    'palegirls',
    'snowwhites',
    'NSFW_GIF',
    'nsfw_gifs',
    'porn_gifs',
    'porninfifteenseconds',
    'NSFW_HTML5',
    'the_best_nsfw_gifs',
    'twingirls',
    'groupofnudegirls',
    'Ifyouhadtopickone',
    'nsfwhardcore',
    'SheLikesItRough',
    'freeuse',
    'whenitgoesin',
    'outercourse',
    'gangbang',
    'breeding',
    'passionx',
    'amateurgirlsbigcocks',
    'facesitting',
    'nsfw_plowcam',
    'pronebone',
    'facefuck',
    'highresNSFW',
    'incestporn',
    'sarah_xxx',
    'remylacroix',
    'Anjelica_Ebbi',
    'BlancNoir',
    'rileyreid',
    'dollywinks',
    'tessafowler',
    'lilyivy',
    'funsizedasian',
    'mycherrycrush',
    'gillianbarnes',
    'kawaiikitten',
    'emilybloom',
    'legendarylootz',
    'sexyflowerwater',
    'miamalkova',
    'sashagrey',
    'keriberry_420',
    'justpeachyy',
    'angelawhite',
    'miakhalifa',
    'alexapearl',
    'missalice_18',
    'evalovia',
    'GiannaMichaels',
    'arianamarie',
    'StraightGirlsPlaying',
    'girlskissing',
    'mmgirls',
    'facesittinglesbians',
    'holdthemoan',
    'O_faces',
    'jilling',
    'gettingherselfoff',
    'quiver',
    'GirlsHumpingThings',
    'forcedorgasms',
    'ruinedorgasms',
    'suctiondildos',
    'baddragon',
    'grool',
    'ladybonersgw',
    'massivecock',
    'chickflixxx',
    'gaybrosgonewild',
    'sissies',
    'selffuck',
    'furryporn',
    'ZootopiaPorn',
    'yiffgif',
    'FurryPornSubreddit',
    'gfur',
    'femyiff',
    'gayfurryporn',
    'yiffcomics',
    'Sharktits',
    'ArousingAvians',
    'anthroids',
    'anthropokeporn',
    'DragonPenis',
    'DragonsFuckingDragons',
    'FeralPokePorn',
    'FurryFrot',
    'GayPokePorn',
    'HorsecocksMasterRace',
    'scalieporn',
    'WholesomeYiff',
    'OnOff',
    'nsfwoutfits',
    'girlswithglasses',
    'collared',
    'seethru',
    'sweatermeat',
    'cfnm',
    'nsfwfashion',
    'leotards',
    'bikinis',
    'bikinibridge',
    'nsfwcosplay',
    'nsfwcostumes',
    'girlsinschooluniforms',
    'WtSSTaDaMiT',
    'tightdresses',
    'lingerie',
    'garterbelts',
    'ChangingRooms',
    'trashyboners',
    'FlashingGirls',
    'publicflashing',
    'sexinfrontofothers',
    'NotSafeForNature',
    'realpublicnudity',
    'socialmediasluts',
    'flashingandflaunting',
    'Tgirls',
    'traps',
    'tgifs',
    'gaysex',
    'topsandbottoms',
    'lgbtsex',
    'gaykink',
    'gayBDSMcommunity',
    'gaymersgonewild',
    'gaybears',
    'LGBTgonewild',
    'bigonewild',
    'gayNSFW',
    'damselsindistress',
    'cuffed',
    'gagged',
    'femaleorgasmdenial',
    'girlscontrolled',
    'pornvids',
    'nsfw_videos',
    'dirtysnapchat',
    'randomactsofblowjob',
    'NSFWFunny',
    'pornhubcomments',
    'dirtykikpals',
    'randomactsofmuffdive',
    'stupidslutsclub',
    'sluttyconfessions',
    'sextrophies',
    'Quarantinegonewild',
    'celebrityarmpits',
    'armpitfetish',
    'dragonsfuckingcars',
    'SCPORN',
    'fedlegs',
    'cummingonfigurines',
    'sextsnap',
    'AdorableBoobs',
    'CamGirlsSlutty',
    'cockdoula',
    'wetspotguy',
    'Koreanhottiesreal'
]
# --- ГЛОБАЛЬНЫЕ НАСТРОЙКИ (будут переопределены в main) ---
DOWNLOAD_LIMIT_PER_SUBREDDIT = 100000
MAX_WORKERS = 15
DOWNLOAD_TIMEOUT = 25
EARLY_SKIP_THRESHOLD = 50

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT,
)

def sanitize_filename(filename_part):
    return "".join(c for c in filename_part if c.isalnum() or c in (' ', '.', '_')).rstrip()

def download_image(url, target_filepath, pbar_images):
    try:
        response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        if 'image' not in response.headers.get('content-type', ''): return
        with open(target_filepath, 'wb') as f: f.write(response.content)
        pbar_images.update(1)
    except requests.exceptions.RequestException: pass

def submission_producer(subreddit_name, limit, submission_queue, stop_event):
    try:
        for submission in reddit.subreddit(subreddit_name).hot(limit=limit):
            if stop_event.is_set():
                break
            
            while True:
                try:
                    submission._fetch()
                    break 
                except TooManyRequests as e:
                    wait_time_str = e.response.headers.get('retry-after', "60")
                    wait_time = int(float(wait_time_str)) + 1
                    tqdm.write(f"  [Rate Limit on fetch] r/{subreddit_name}. Waiting for {wait_time} s...")
                    time.sleep(wait_time)
                except Exception as e:
                    tqdm.write(f"  [Error fetching submission details] ID: {submission.id}, Error: {e}")
                    submission = None 
                    break
            
            if submission:
                submission_queue.put(submission)
                
    except TooManyRequests as e:
        wait_time_str = e.response.headers.get('retry-after', "60")
        wait_time = int(float(wait_time_str)) + 1
        tqdm.write(f"  [Rate Limit on listing] r/{subreddit_name}. Waiting for {wait_time} s and stopping producer.")
        time.sleep(wait_time) 
    except Exception as e:
        tqdm.write(f"  [PRAW Error in Producer] r/{subreddit_name}: {e}")
    finally:
        submission_queue.put(None)

def fetch_and_download(subreddit_names, save_path_base, is_nsfw_collection):
    # Создаем базовую директорию, если её нет
    os.makedirs(save_path_base, exist_ok=True)

    for sub_name in tqdm(subreddit_names, desc="Overall Subreddits Progress", unit="sub", position=0, leave=True):
        tqdm.write(f"\nFetching from r/{sub_name}...")
        subreddit_save_path = os.path.join(save_path_base, sub_name)
        os.makedirs(subreddit_save_path, exist_ok=True)

        estimated_posts_to_scan = DOWNLOAD_LIMIT_PER_SUBREDDIT * 5
        submission_queue = queue.Queue(maxsize=200)
        stop_event = threading.Event()

        producer_thread = threading.Thread(
            target=submission_producer,
            args=(sub_name, estimated_posts_to_scan, submission_queue, stop_event)
        )
        producer_thread.start()
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            with tqdm(total=estimated_posts_to_scan, desc=f"Scanning r/{sub_name}", unit="posts", position=1, leave=False, dynamic_ncols=True) as pbar_posts, \
                 tqdm(total=DOWNLOAD_LIMIT_PER_SUBREDDIT, desc=f"Downloaded r/{sub_name}", unit="img", position=2, leave=False, dynamic_ncols=True) as pbar_images:

                while pbar_posts.n < estimated_posts_to_scan:
                    try:
                        submission = submission_queue.get(timeout=20)
                        if submission is None: break
                        
                        pbar_posts.update(1)

                        if pbar_posts.n >= EARLY_SKIP_THRESHOLD and pbar_images.n == 0:
                            tqdm.write(f"  [Info] Scanned {pbar_posts.n} posts in r/{sub_name} with 0 images found. Skipping subreddit.")
                            break

                        if pbar_images.n >= DOWNLOAD_LIMIT_PER_SUBREDDIT: break

                        is_image = (hasattr(submission, 'post_hint') and submission.post_hint == 'image') or \
                                   (hasattr(submission, 'is_gallery') and submission.is_gallery)
                        if not is_image: continue
                        if not is_nsfw_collection and submission.over_18: continue

                        if hasattr(submission, 'is_gallery') and submission.is_gallery and hasattr(submission, 'media_metadata') and submission.media_metadata:
                            for media_id, item in submission.media_metadata.items():
                                if item.get('e') == 'Image':
                                    url = html.unescape(item.get('s', {}).get('u', ''))
                                    if not url: continue
                                    ext = "." + item.get('m', 'image/jpeg').split('/')[-1]
                                    path = os.path.join(subreddit_save_path, f"{submission.id}_{media_id}{ext}")
                                    executor.submit(download_image, url, path, pbar_images)
                        elif hasattr(submission, "url") and submission.url.lower().endswith(('.jpg', '.jpeg', '.png')):
                            filename = sanitize_filename(os.path.basename(submission.url))
                            path = os.path.join(subreddit_save_path, f"{submission.id}_{filename}")
                            executor.submit(download_image, submission.url, path, pbar_images)

                    except queue.Empty:
                        tqdm.write(f"  [Info] No new posts from producer for 20 seconds. Finalizing.")
                        break

                stop_event.set()
                while not submission_queue.empty():
                    try: submission_queue.get_nowait()
                    except queue.Empty: continue
                
                executor.shutdown(wait=False, cancel_futures=True)
                tqdm.write(f"Finished r/{sub_name}, downloaded {pbar_images.n} images from {pbar_posts.n} scanned posts.\n")

# --- ФУНКЦИЯ ВЫБОРА ДИСКА ---
def select_storage_path():
    print("\n--- ВЫБОР МЕСТА СОХРАНЕНИЯ / STORAGE SELECTION ---")
    print("1. Текущая папка (./data/reddit)")
    print("2. Диск sdb1 (ожидается в /mnt/sdb1)")
    print("3. Ввести свой путь вручную (Custom path)")
    
    choice = input("Выберите вариант (1-3): ").strip()
    
    base_path = ""
    
    if choice == '1':
        base_path = "data/reddit"
    elif choice == '2':
        # Проверяем популярные точки монтирования для sdb1
        potential_mounts = ["/mnt/sdb1", "/media/sdb1", "/mnt/data", "/media/data"]
        found = False
        for mount in potential_mounts:
            if os.path.exists(mount) and os.path.isdir(mount):
                base_path = os.path.join(mount, "data")
                print(f"--> Найден диск: {mount}. Сохраняем в {base_path}")
                found = True
                break
        
        if not found:
            print("(!) Не удалось автоматически найти точку монтирования для sdb1.")
            print("Убедитесь, что диск смонтирован (команда: mount /dev/sdb1 /mnt/sdb1).")
            base_path = input("Введите путь к точке монтирования вручную (например /mnt/sdb1): ").strip()
            base_path = os.path.join(base_path, "data")
            
    elif choice == '3':
        base_path = input("Введите полный путь к папке (например D:\\Images или /mnt/disk2): ").strip()
    else:
        print("Неверный выбор, используется путь по умолчанию.")
        base_path = "data/reddit"
        
    # Подтверждение
    print(f"\n--> Файлы будут сохранены в: {os.path.abspath(base_path)}")
    try:
        os.makedirs(base_path, exist_ok=True)
        # Пробная запись, чтобы проверить права доступа
        test_file = os.path.join(base_path, ".write_test")
        with open(test_file, 'w') as f: f.write('ok')
        os.remove(test_file)
    except PermissionError:
        print(f"\n[ОШИБКА] Нет прав на запись в {base_path}!")
        print("Попробуйте запустить скрипт через sudo или измените права на папку (chmod 777 ...)")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ОШИБКА] Не удалось создать директорию: {e}")
        sys.exit(1)
        
    return base_path

if __name__ == "__main__":
    # Сначала спрашиваем, куда сохранять
    ROOT_SAVE_PATH = select_storage_path()
    
    sfw_path = os.path.join(ROOT_SAVE_PATH, "sfw_images")
    nsfw_path = os.path.join(ROOT_SAVE_PATH, "nsfw_images")

    print("\n--- Starting SFW Image Collection ---")
    fetch_and_download(SFW_SUBREDDITS, sfw_path, is_nsfw_collection=False)

    print("\n\n--- Starting NSFW Image Collection (USE RESPONSIBLY!) ---")
    fetch_and_download(NSFW_SUBREDDITS, nsfw_path, is_nsfw_collection=True)

    print("\n--- Collection Finished ---")