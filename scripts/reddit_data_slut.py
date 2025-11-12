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

code
Python
download
content_copy
expand_less
SFW_SUBREDDITS = [
    'PrettyOlderWomen',
    'PrettyGirls',
    'teenagers'
    'GilmoreGirls',
    'fashion',
    'popculturechat',
    'carspotting',
    'EmoFashion',
    'pics',
    'photography',
    'itookapicture',
    'streetphotography',
    'portraits',
    'humanporn',
    'OldSchoolCool',
    'TheWayWeWere',
    'AccidentalRenaissance',
    'prettygirls',
    'malehairadvice',
    'femalefashionadvice',
    'malefashionadvice',
    'OUTFITS',
    'streetwear',
    'lookatmydog',
    'cats',
    'aww',
    'mildlyinteresting',
    'interestingasfuck',
    'MadeMeSmile',
    'HumansBeingBros',
    'wholesomememes',
    'CozyPlaces',
    'EarthPorn',
    'waterporn',
    'SkyPorn',
    'BotanicalPorn',
    'AnimalPorn',
    'ArchitecturePorn',
    'CityPorn',
    'VillagePorn',
    'AbandonedPorn',
    'InfrastructurePorn',
    'MachinePorn',
    'MilitaryPorn',
    'PolicePorn',
    'FirefighterPorn',
    'DoctorPorn',
    'LawyerPorn',
    'TeacherPorn',
    'StudentPorn',
    'AthletePorn',
    'MusicianPorn',
    'ArtistPorn',
    'DancerPorn',
    'CosplayGirls',
    'cosplay',
    'tattoos',
    'Art',
    'drawing',
    'painting',
    'sculpture',
    'Graffiti',
    'streetart',
    'Museum',
    'HistoryPorn',
    'OldPhotosInRealLife',
    'ColorizedHistory',
    '90s',
    '80s',
    '70s',
    '60s',
    '50s',
    '40s',
    '30s',
    '20s',
    'history',
    'travel',
    'backpacking',
    'hiking',
    'camping',
    'climbing',
    'sailing',
    'surfing',
    'skateboarding',
    'snowboarding',
    'skiing',
    'bicycling',
    'motorcycles',
    'cars',
    'formula1',
    'sports',
    'nba',
    'soccer',
    'baseball',
    'hockey',
    'tennis',
    'Fitness',
    'bodybuilding',
    'yoga',
    'meditation',
    'happy',
    'sad',
    'funny',
    'creepy',
    'creepyPMs',
    'creepyasterisks',
    'facepalm',
    'mildlyinfuriating',
    'iamverysmart',
    'insanepeoplefacebook',
    'oldpeoplefacebook',
    'tifu',
    'AITA',
    'relationships',
    'relationship_advice',
    'weddingplanning',
    'BabyBumps',
    'daddit',
    'mommit',
    'Parenting',
    'lifehacks',
    'DIY',
    'Frugal',
    'personalfinance',
    'investing',
    'wallstreetbets',
    'science',
    'space',
    'nasa',
    'technology',
    'gadgets',
    'programming',
    'webdev',
    'gamedev',
    'gaming',
    'movies',
    'television',
    'Music',
    'books',
    'writing',
    'comics',
    'anime',
    'manga',
    'food',
    'FoodPorn',
    'Cooking',
    'baking',
    'beer',
    'wine',
    'cocktails',
    'Coffee',
    'tea',
    'gardening',
    'homestead',
    'sustainability',
    'ZeroWaste',
    'minimalism'
]
NSFW_SUBREDDITS = [
    'nsfw',
    'porn',
    'realgirls',
    'Amateur',
    'gonewild',
    'rule34',
    'ass',
    'boobs',
    'pussy',
    'tits',
    'milf',
    'cuckold',
    'BDSM',
    'bondage',
    'lesbians',
    'gay',
    'bisexual',
    'trans',
    'hentai',
    'ecchi',
    'feet',
    'anal',
    'oral',
    'threesome',
    'orgy',
    'cumsluts',
    'creampie',
    'deepthroat',
    'blowjobs',
    'handjobs',
    'joi',
    'pegging',
    'femdom',
    'maledom',
    'submissive',
    'dominant',
    'petplay',
    'watersports',
    'scat',
    'gore',
    'necrophilia',
    'incest',
    'beastiality',
    'public',
    'exhibition',
    'voyeur',
    'upskirt',
    'downblouse',
    'squirting',
    'fingering',
    'fisting',
    'sounding',
    'peeing',
    'shitting',
    'vomit',
    'suicidegirls',
    'altgonewild',
    'punkgw',
    'gothsluts',
    'emogirls',
    'scene',
    'cosplayboobs',
    'gamergirls',
    'nerdgirls',
    'geeks',
    'asians',
    'latinas',
    'ebony',
    'indian',
    'arab',
    'white',
    'redheads',
    'blondes',
    'brunettes',
    'chubby',
    'bbw',
    'ssbbw',
    'thick',
    'curvy',
    'petite',
    'tall',
    'short',
    'fit',
    'muscle',
    'hairy',
    'shaved',
    'natural',
    'pierced',
    'tattooed',
    'pregnant',
    'lactating',
    'couples',
    'swingers',
    'cheating',
    'dirtyr4r',
    'sex',
    'sexting',
    'camwhores',
    'onlyfans',
    'patreon',
    'TipOfMyPenis',
    'nsfw411',
    'iWantToFuckHer',
    'distension',
    'bimbofetish',
    'christiangirls',
    'dirtygaming',
    'sexybutnotporn',
    'femalepov',
    'omgbeckylookathiscock',
    'sexygirls',
    'breedingmaterial',
    'toocuteforporn',
    'justhotwomen',
    'realsexyselfies',
    'stripgirls',
    'uncommonposes',
    'gifsofremoval',
    'nostalgiafapping',
    'oilporn',
    'bisexy',
    'riskyporn',
    'gonewild30plus',
    'preggoporn',
    'realmoms',
    'legalteens',
    'collegesluts',
    'adorableporn',
    'legalteensXXX',
    'gonewild18',
    '18_19',
    'PornStarletHQ',
    'fauxbait',
    'homemadexxx',
    'dirtypenpals',
    'FestivalSluts',
    'CollegeAmateurs',
    'amateurcumsluts',
    'nsfw_amateurs',
    'funwithfriends',
    'randomsexiness',
    'amateurporn',
    'normalnudes',
    'camsluts',
    'tiktokliveslip',
    'PetiteGoneWild',
    'gonewildstories',
    'treesgonewild',
    'gonewildaudio',
    'GWNerdy',
    'gonemild',
    'gifsgonewild',
    'analgw',
    'gonewildsmiles',
    'onstageGW',
    'RepressedGoneWild',
    'bdsmgw',
    'UnderwearGW',
    'LabiaGW',
    'TributeMe',
    'WeddingsGoneWild',
    'gwpublic',
    'assholegonewild',
    'leggingsgonewild',
    'dykesgonewild',
    'goneerotic',
    'gonewildhairy',
    'gonewildtrans',
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

DOWNLOAD_LIMIT_PER_SUBREDDIT = 1000
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
            if any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
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
