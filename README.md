 ❯ curl -X POST "https://blackfoxus.ru:8000/api/classify_text/" \
     -H "Content-Type: application/json" \
     -d '{"text":"your text here"}'

{"prediction":0.19568155451220104,"is_ad":false}%                                                                                                                                                                 ❯ curl -X POST "https://blackfoxus.ru:8000/api/classify_image/" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@photo_2025-03-08_22-42-12.jpg"

{"prediction":0.6427165534846647,"is_nsfw":true,"error":null}%                                                                                                                                                    
  ~/Изображения ❯          