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
    'gonwild',
    'ratemynudebody',
    'onmww',
    'GWCouples',
    'gonewildcouples',
    'WouldYouFuckMyWife',
    'gonewildcurvy',
    'GoneWildplus',
    'BigBoobsGW',
    'bigboobsgonewild',
    'mycleavage',
    'AsiansGoneWild',
    'gonewildcolor',
    'indiansgonewild',
    'latinasgw',
    'pawgtastic',
    'workgonewild',
    'GoneWildScrubs',
    'swingersgw',
    'militarygonewild',
    'NSFW_Snapchat',
    'snapleaks',
    'wifesharing',
    'hotwife',
    'slutwife',
    'futanari',
    'doujinshi',
    'yiff',
    'monstergirl',
    'mechanicalsluts',
    'rule34_comics',
    'sex_comics',
    'overwatch_porn',
    'pokeporn',
    'bowsette',
    'rule34lol',
    'rule34overwatch',
    'nintendowaifus',
    '34honor',
    'fivefapsatfreddys',
    'breathofthegonewild',
    'animalcrossingr34',
    'apexlegends_porn',
    'tflewd',
    'thelostwoods',
    'hentai_gif',
    'WesternHentai',
    'hentai_irl',
    'artistic_hentai',
    'hentaibeast',
    'hentaihumiliation',
    'traphentai',
    'ahegao',
    'ahegao_irl',
    'hypnohentai',
    'tentai',
    'handholding',
    'honeyfuckers',
    'itshiptofuckbees',
    'guro',
    'hentaibondage',
    'animeshorts',
    'kuroihada',
    '2dtittytouching',
    'buttfangs',
    'yuri',
    'ZettaiRyouiki',
    'hentaifemdom',
    'thighhighhentai',
    'animebooty',
    'swimsuithentai',
    'animelegs',
    'animearmpits',
    '2dsuccubi',
    'animemidriff',
    'skindentation',
    'thighdeology',
    'chiisaihentai',
    'bokunoeroacademia',
    'waifusgonewild',
    'sideoppai',
    'BDSMcommunity',
    'onherknees',
    'blowjobsandwich',
    'asstastic',
    'facedownassup',
    'assinthong',
    'bigasses',
    'buttplug',
    'TheUnderbun',
    'booty',
    'pawg',
    'paag',
    'cutelittlebutts',
    'HungryButts',
    'celebritybutts',
    'cosplaybutts',
    'mooning',
    'painal',
    'masterofanal',
    'buttsharpies',
    'AssholeBehindThong',
    'spreadem',
    'bendover',
    'girlsinyogapants',
    'yogapants',
    'boobies',
    'TittyDrop',
    'boltedontits',
    'boobbounce',
    'homegrowntits',
    'breastenvy',
    'youtubetitties',
    'torpedotits',
    'thehangingboobs',
    'page3glamour',
    'biggerthanyouthought',
    'BustyPetite',
    'hugeboobs',
    'stacked',
    'burstingout',
    '2busty2hide',
    'bigtiddygothgf',
    'engorgedveinybreasts',
    'pokies',
    'ghostnipples',
    'nipples',
    'puffies',
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


SFW_SAVE_PATH = "data/reddit/sfw_images"
NSFW_SAVE_PATH = "data/reddit/nsfw_images"
os.makedirs(SFW_SAVE_PATH, exist_ok=True)
os.makedirs(NSFW_SAVE_PATH, exist_ok=True)

DOWNLOAD_LIMIT_PER_SUBREDDIT = 5000
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

# --- ФИНАЛЬНАЯ ВЕРСИЯ ПРОДЮСЕРА ---
def submission_producer(subreddit_name, limit, submission_queue, stop_event):
    """
    'Продюсер', который получает посты от Reddit, ПОЛНОСТЬЮ ИХ ЗАГРУЖАЕТ
    и умеет обрабатывать Rate Limit.
    """
    try:
        for submission in reddit.subreddit(subreddit_name).hot(limit=limit):
            if stop_event.is_set():
                break
            
            # --- НОВЫЙ БЛОК: Принудительная загрузка данных ---
            while True:
                try:
                    # Эта строка заставляет PRAW загрузить все "ленивые" атрибуты.
                    # Она сама может вызвать ошибку TooManyRequests.
                    submission._fetch()
                    break # Если успешно, выходим из внутреннего цикла
                except TooManyRequests as e:
                    # Перехватываем ошибку Rate Limit здесь, внутри цикла
                    wait_time_str = e.response.headers.get('retry-after', "60")
                    wait_time = int(float(wait_time_str)) + 1
                    tqdm.write(f"  [Rate Limit on fetch] r/{subreddit_name}. Waiting for {wait_time} s...")
                    time.sleep(wait_time)
                except Exception as e:
                    tqdm.write(f"  [Error fetching submission details] ID: {submission.id}, Error: {e}")
                    # Помечаем пост как невалидный, чтобы пропустить его
                    submission = None 
                    break
            # --- КОНЕЦ НОВОГО БЛОКА ---
            
            if submission: # Кладем в очередь только если загрузка прошла успешно
                submission_queue.put(submission)
                
    except TooManyRequests as e:
        # Эта ошибка может возникнуть на самом первом запросе (получение списка)
        wait_time_str = e.response.headers.get('retry-after', "60")
        wait_time = int(float(wait_time_str)) + 1
        tqdm.write(f"  [Rate Limit on listing] r/{subreddit_name}. Waiting for {wait_time} s and stopping producer.")
        time.sleep(wait_time) # Просто ждем и завершаем продюсер для этого саба
    except Exception as e:
        tqdm.write(f"  [PRAW Error in Producer] r/{subreddit_name}: {e}")
    finally:
        submission_queue.put(None)

def fetch_and_download(subreddit_names, save_path_base, is_nsfw_collection):
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
                        submission = submission_queue.get(timeout=20) # Увеличим таймаут
                        if submission is None: break
                        
                        pbar_posts.update(1)

                        if pbar_posts.n >= EARLY_SKIP_THRESHOLD and pbar_images.n == 0:
                            tqdm.write(f"  [Info] Scanned {pbar_posts.n} posts in r/{sub_name} with 0 images found. Skipping subreddit.")
                            break

                        if pbar_images.n >= DOWNLOAD_LIMIT_PER_SUBREDDIT: break

                        # Теперь эти проверки не вызывают сетевых запросов
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

if __name__ == "__main__":
    print("--- Starting SFW Image Collection ---")
    fetch_and_download(SFW_SUBREDDITS, SFW_SAVE_PATH, is_nsfw_collection=False)

    print("\n\n--- Starting NSFW Image Collection (USE RESPONSIBLY!) ---")
    fetch_and_download(NSFW_SUBREDDITS, NSFW_SAVE_PATH, is_nsfw_collection=True)

    print("\n--- Collection Finished ---")