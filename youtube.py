from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

video_id = input("Podaj ID filmu (samo ID, nie link): ")
video_title = input("Podaj tytuł filmu: ")

ytt_api = YouTubeTranscriptApi()
transkrypcja = ytt_api.fetch(video_id, languages=['en'])

chunks = [] 
text = ""
start = None

for snippet in transkrypcja.snippets:
    if start is None:
        start = snippet.start
        text = snippet.text + " "

    elif snippet.start - start <= 45:
        text += snippet.text + " "

    else:
        chunks.append({"text": text.strip(), "start": start})
        start = snippet.start
        text = snippet.text + " "

if text != "":
    chunks.append({"text": text.strip(), "start": start})

print(f"\nUdało się! Podzieliłem film na {len(chunks)} bloków.")

teksty_do_oceny = [chunk['text'] for chunk in chunks]
embeddings = model.encode(teksty_do_oceny)
embeddings_title = model.encode([video_title])
wyniki = cosine_similarity(embeddings_title, embeddings)[0]

najlepszy_indeks = np.argmax(wyniki)
najlepszy_wynik = wyniki[najlepszy_indeks]
najlepszy_chunk = chunks[najlepszy_indeks]
procenty = round(najlepszy_wynik * 100, 1)

# ---  PARAMETRY ANALITYCZNE ---

# 1. Czas do tematu (w którym procencie trwania filmu pada najlepszy fragment) //idk czy nie lepiej brac wystarczająco dobry????
# Zakładamy, że koniec filmu to start ostatniego bloku + 45 sekund
calkowity_czas = chunks[-1]['start'] + 45.0 
czas_do_tematu = round((najlepszy_chunk['start'] / calkowity_czas) * 100, 1)

# 2. Gęstość tematu (ile procent bloków przekroczyło próg np. 30% powiązania z tytułem) //idk czy taki akurat prog on jest to sprawdzenia ekperymentalnie
prog_przyzwoitosci = 0.30
ilosc_sensownych_blokow = sum(1 for w in wyniki if w >= prog_przyzwoitosci)
gestosc_tematu = round((ilosc_sensownych_blokow / len(chunks)) * 100, 1)

# 3. Wariancja / Skoki tematyczne (odchylenie standardowe wyników)
# Pokazuje, jak bardzo temat "skacze" (wyższe odchylenie = większy chaos)
skoki_tematyczne = round(np.std(wyniki) * 100, 1)


# --- WYSWIETLANIE WYNIKÓW ---
print("\n" + "="*50)
print("="*50)

print(f" Największe powiązanie z tytułem: {procenty}%")
print(f"Czas do 'tematu z tytulu': Najlepszy fragment pojawia się w {czas_do_miesa}% trwania filmu.")
print(f"Gęstość tematu: {gestosc_tematu}% bloków wideo ma sensowne powiązanie z tytułem.")
print(f"Skoki tematyczne (odchylenie): {skoki_tematyczne} (im wyżej, tym większy chaos).")

print("\n--- Najlepszy fragment zaczyna się w", najlepszy_chunk['start'], "sekundzie ---")
print(f"> {najlepszy_chunk['text']}")
print("="*50)