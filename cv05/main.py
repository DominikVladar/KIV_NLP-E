import argparse
import json
import re
import random
import asyncio
import aiohttp
import os
import sys
from pathlib import Path
from typing import List, Dict, Union, Optional

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass


DATA_FILES = {
    "csfd": { 
        "train": Path("data/csfd-train.tsv"), 
        "test": Path("data/csfd-test-llm.tsv") 
    },
    "sts": { 
        "train": Path("data/anlp01-sts-free-train.tsv"), 
        "test": Path("data/anlp01-sts-free-test-llm.tsv") 
    },
    "ner": { 
        "train": Path("data/ner-train.txt"), 
        "test": Path("data/ner-dev-llm.txt") 
    }
}

# OpenRouter API nastavení
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not API_KEY:
    print("FATAL ERROR: Proměnná prostředí 'OPENROUTER_API_KEY' není nastavena.")
    print("Vytvořte soubor .env nebo nastavte proměnnou v terminálu.")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Limit souběžných dotazů (aby nedošlo k Rate Limitingu)
CONCURRENT_LIMIT = 5 


def load_tsv(path: Path, header: Optional[List[str]] = None) -> List[Dict]:
    """Načte TSV soubor a vrátí seznam slovníků."""
    data = []
    if not path.exists():
        print(f"Warning: Soubor {path} nebyl nalezen.")
        return []
    
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            toks = line.split("\t")
            
            if not header:
                header = toks
                continue

            if len(toks) < len(header):
                toks += [""] * (len(header) - len(toks))
            
            row = {k: v for k, v in zip(header, toks[:len(header)])}
            data.append(row)
    return data

def load_ner_txt(path: Path, has_tags: bool = False) -> List[Dict]:
    """Načte NER data ve formátu CoNLL (slovo na řádek, prázdný řádek odděluje věty)."""
    data = []
    toks, tags = [], []
    
    if not path.exists():
        print(f"Warning: Soubor {path} nebyl nalezen.")
        return []

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            # Prázdný řádek = konec věty
            if not line:
                if toks:
                    entry = {"tokens": toks}
                    if has_tags:
                        entry["tags"] = tags
                    data.append(entry)
                    toks, tags = [], []
                continue
            
            vals = line.split()
            # První sloupec je token
            toks.append(vals[0])
            
            # Pokud očekáváme tagy, načteme je (poslední sloupec), jinak fallback na 'O'
            if has_tags:
                tags.append(vals[-1] if len(vals) > 1 else "O")
    
    # Přidání poslední věty, pokud soubor nekončí prázdným řádkem
    if toks:
        entry = {"tokens": toks}
        if has_tags:
            entry["tags"] = tags
        data.append(entry)
        
    return data


def format_few_shot_csfd(examples: List[Dict]) -> str:
    prompt = ""
    for ex in examples:
        label = ex.get('label', '1')
        text = ex.get('text', '')
        prompt += f"Recenze: \"{text}\"\nSentiment: {label}\n\n"
    return prompt

def format_few_shot_sts(examples: List[Dict]) -> str:
    prompt = ""
    for ex in examples:
        s1 = ex.get('a', '')
        s2 = ex.get('b', '')
        score = ex.get('sts', '3.0')
        prompt += f"Věta 1: {s1}\nVěta 2: {s2}\nShoda: {score}\n\n"
    return prompt

def format_few_shot_ner(examples: List[Dict]) -> str:
    prompt = ""
    for ex in examples:
        sent = " ".join(ex['tokens'])
        tags = " ".join(ex['tags'])
        prompt += f"Věta: {sent}\nTagy: {tags}\n\n"
    return prompt


def get_csfd_prompt(text: str, fs_text: str = "") -> str:
    base = (
        "Vyhodnoť sentiment filmové recenze pro klasifikaci. "
        "Výsledek musí být jedno číslo bez dalšího textu.\n\n"
        "Definice tříd:\n"
        "0 = Negativní (recenze obsahuje silnou nespokojenost, kritiku, výtky nebo zklamání)\n"
        "1 = Neutrální (informativní komentář nebo smíšené hodnocení bez jasného postoje)\n"
        "2 = Pozitivní (převážně spokojenost, chvála, jednoznačně kladné hodnocení)\n\n"
        "Pokyny:\n"
        "- Ignoruj popis hereckého obsazení, režie či produkce, pokud neobsahují hodnotící soud.\n"
        "- Ironii vyhodnoť podle skutečného sentimentu, ne povrchově.\n"
        "- Pokud jsou v textu pozitivní i negativní části, rozhodni podle celkového tónu.\n"
        "- Pokud není přítomné hodnotící vyjádření, zvol 1.\n\n"
    )

    if fs_text:
        base += "Příklady:\n" + fs_text + "Tvoje úloha:\n"

    return base + f"Recenze: \"{text}\"\nSentiment:"

def get_sts_prompt(s1: str, s2: str, fs_text: str = "") -> str:
    base = "Ohodnoť sémantickou podobnost dvou vět na stupnici od 0.0 do 6.0. Odpověz POUZE číslem.\n\n"
    if fs_text:
        base += "Příklady:\n" + fs_text + "Tvoje úloha:\n"
    return base + f"Věta 1: {s1}\nVěta 2: {s2}\nShoda:"

def get_ner_prompt(tokens: List[str], fs_text: str = "") -> str:
    sentence = " ".join(tokens)
    base = (
        "Jsi přesný NER systém. Pro každé slovo ve větě přiřaď přesně jeden tag "
        "podle schématu CNEC. Počet tagů musí přesně odpovídat počtu slov. "
        "Výstup je pouze seznam tagů oddělený mezerami, bez dalšího textu.\n\n"
        "Stručný popis tagů:\n"
        "B-P / I-P = osobní jména (lidé)\n"
        "B-G / I-G = geografické názvy (města, státy, hory, řeky)\n"
        "B-I / I-I = instituce (organizace, firmy, úřady)\n"
        "B-O / I-O = artefakty (díla, produkty, objekty)\n"
        "B-M / I-M = mediální názvy (filmy, seriály, knihy, alba)\n"
        "B-T / I-T = časové výrazy (datum, rok, čas)\n"
        "B-A / I-A = čísla v adresách (ulice, popisná čísla)\n"
        "O = mimo jakoukoli pojmenovanou entitu\n\n"
        "Pokyny:\n"
        "- U víceslovných entit začni tagem B- a pokračuj I-.\n"
        "- Pokud si nejsi jistý, použij O.\n"
        "- Nikdy neprodukuj více nebo méně tagů než je slov.\n\n"
    )

    if fs_text:
        base += "Příklady:\n" + fs_text + "Tvoje úloha:\n"

    return base + f"Věta: {sentence}\nTagy:"


def parse_csfd(output: str, _=None) -> int:
    # Hledá číslo 0, 1, nebo 2
    match = re.search(r'\b[0-2]\b', output)
    return int(match.group(0)) if match else 1

def parse_sts(output: str, _=None) -> float:
    # Hledá float nebo int
    match = re.search(r"[-+]?\d*\.\d+|\d+", output)
    if match:
        val = float(match.group(0))
        return max(0.0, min(6.0, val))
    return 3.0

def parse_ner(output: str, input_len: int) -> List[str]:
    clean_out = output.replace("Tagy:", "").strip()
    tags = clean_out.split()
    
    # Zarovnání délky
    if len(tags) > input_len:
        return tags[:input_len]
    elif len(tags) < input_len:
        return tags + ["O"] * (input_len - len(tags))
    return tags


async def call_api(session, model: str, prompt: str, max_tokens: int) -> str:
    """Odeslání jednoho dotazu na API s ošetřením chyb a retry logikou."""
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, # Chci deterministický výstup
        "max_tokens": max_tokens
    }

    retries = 5
    for attempt in range(retries):
        try:
            async with session.post(BASE_URL, headers=HEADERS, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'choices' in data and len(data['choices']) > 0:
                        return data['choices'][0]['message']['content']
                    return ""
                
                elif response.status == 429: # Rate Limit
                    wait_time = 2 ** attempt # "Exponential backoff"
                    print(f"Rate Limit (429). Čekám {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                
                else:
                    text = await response.text()
                    print(f"API Error {response.status}: {text}")
                    return ""

        except Exception as e:
            print(f"Network/Client Error: {e}. Zkouším znovu...")
            await asyncio.sleep(2 ** attempt)
            continue

    print("FATAL: Nepodařilo se získat odpověď ani po opakovaných pokusech.")
    return ""

async def process_task(
    task_name: str, 
    model: str, 
    few_shot_count: int, 
    prompt_func, 
    fs_format_func, 
    parse_func, 
    max_tokens: int
):
    print(f"\n--- Zpracovávám úlohu: {task_name.upper()} ---")
    
    # Načtení testovacích dat
    if task_name == "ner":
        test_data = load_ner_txt(DATA_FILES[task_name]["test"], has_tags=False)
    else:
        test_data = load_tsv(DATA_FILES[task_name]["test"], header=["label", "text"] if task_name == "csfd" else ["a", "b", "sts"])

    if task_name == "csfd":
        test_data = test_data[1:]
    
    if not test_data:
        print(f"Konec: Žádná testovací data pro {task_name}.")
        return

    # Příprava Few-Shot kontextu
    fs_text = ""
    if few_shot_count > 0:
        if task_name == "ner":
            train_data = load_ner_txt(DATA_FILES[task_name]["train"], has_tags=True)
        else:
            train_data = load_tsv(DATA_FILES[task_name]["train"], header=["label", "text"] if task_name == "csfd" else ["a", "b", "sts"])
        
        if train_data:
            # Náhodný výběr vzorků
            samples = random.sample(train_data, min(len(train_data), few_shot_count))
            fs_text = fs_format_func(samples)
            print(f"Používám {len(samples)} few-shot příkladů.")

    # Generování promptů
    tasks_data = []
    for idx, row in enumerate(test_data):
        if task_name == "csfd":
            prompt = prompt_func(row['text'], fs_text)
            input_len = None
        elif task_name == "sts":
            prompt = prompt_func(row['a'], row['b'], fs_text)
            input_len = None
        elif task_name == "ner":
            prompt = prompt_func(row['tokens'], fs_text)
            input_len = len(row['tokens'])
        
        tasks_data.append((idx, prompt, input_len))

    # Spuštění asynchronního zpracování
    results = []
    sem = asyncio.Semaphore(CONCURRENT_LIMIT)

    async def worker(idx, prompt, input_len):
        async with sem:
            raw_output = await call_api(session, model, prompt, max_tokens)
            parsed = parse_func(raw_output, input_len)
            results.append({"testset_id": idx, "prediction": parsed})
            
            if len(results) % 10 == 0:
                print(f"{task_name}: {len(results)}/{len(test_data)} hotovo")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        coroutines = [worker(i, p, l) for i, p, l in tasks_data]
        await asyncio.gather(*coroutines)

    # Seřazení a uložení
    results.sort(key=lambda x: x['testset_id'])
    
    out_dir = Path("submissions")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"submission_{model.replace(".", "_").replace("-", "_").replace("/", "_")}_{few_shot_count}{"_with_tags" if task_name=="ner" else "_new_query"}{"_fixed" if task_name=="csfd" else ""}.{task_name}.json"
    
    with out_file.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Uloženo {len(results)} výsledků do {out_file}")

async def main(args):
    if args.task in ['all', 'csfd']:
        await process_task("csfd", args.model, args.few_shot, get_csfd_prompt, format_few_shot_csfd, parse_csfd, max_tokens=5)
        
    if args.task in ['all', 'sts']:
        await process_task("sts", args.model, args.few_shot, get_sts_prompt, format_few_shot_sts, parse_sts, max_tokens=5)
        
    if args.task in ['all', 'ner']:
        await process_task("ner", args.model, args.few_shot, get_ner_prompt, format_few_shot_ner, parse_ner, max_tokens=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Solver using OpenRouter API")
    parser.add_argument("--model", type=str, default="meta-llama/llama-3.1-8b-instruct", help="OpenRouter model ID")
    parser.add_argument("--task", type=str, default="all", choices=["all", "csfd", "sts", "ner"])
    parser.add_argument("--few_shot", type=int, default=3, help="Počet few-shot příkladů")
    
    args = parser.parse_args()
    
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main(args))