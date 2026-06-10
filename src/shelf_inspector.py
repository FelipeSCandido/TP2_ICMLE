import os
import json
import hashlib
import time
import base64
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
_root = Path(__file__).resolve().parent.parent
if (_root / ".env").exists():
    load_dotenv(_root / ".env", override=True)

CACHE_DIR = Path(os.getenv("CACHE_DIR", "./cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_SRC_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = _SRC_DIR.parent / "prompts"
if not PROMPTS_DIR.exists():
    PROMPTS_DIR = _SRC_DIR.parent.parent / "prompts" 

ZONE_DESCRIPTIONS = {
    "Z_S1": "Secção de frescos e lacticínios",
    "Z_S2": "Secção de padaria e pastelaria",
    "Z_S3": "Secção de talho e charcutaria",
    "Z_S4": "Secção de produtos de higiene e limpeza",
    "Z_S5": "Secção de bebidas e conservas",
    "Z_S6": "Secção de vinhos e destilados",
    "Z_S7": "Secção de produtos congelados",
}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path(image_path: Path, strategy: str) -> Path:
    md5 = _md5(image_path)
    return CACHE_DIR / f"{md5}_{strategy}.json"


def _load_from_cache(cache_file: Path) -> Optional[dict]:
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return None


def _save_to_cache(cache_file: Path, result: dict):
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def _load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def _image_to_base64(image_path: Path) -> tuple[str, str]:
    suffix = image_path.suffix.lower()
    media_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".png": "image/png", ".webp": "image/webp"}
    media_type = media_map.get(suffix, "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, media_type


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Não foi possível extrair JSON válido da resposta:\n{text[:500]}")


def _call_gemini_with_backoff(client, model_name: str, prompt: str,
                               image_data: str, media_type: str,
                               max_retries: int = 4) -> str:
    from google import genai
    from google.genai import types

    if client is None:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    contents = [
        types.Part.from_bytes(
            data=base64.b64decode(image_data),
            mime_type=media_type
        ),
        prompt
    ]

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents
            )
            return response.text
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower():
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        "Quota diária do Gemini esgotada. "
                        "O sistema continua a funcionar para imagens em cache."
                    ) from e
                wait = 2 ** (attempt + 1)
                print(f"[Rate limit] Aguardando {wait}s (tentativa {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise


def inspect_image(
    image_path: str,
    zone_id: str = "Z_S1",
    strategy: str = "cot",  
    model_name: str = "gemini-1.5-flash",
    force_refresh: bool = False,
) -> dict:
    """Analisa uma imagem de prateleira."""
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY não definida. Copia .env.example para .env e preenche.")

    client = genai.Client(api_key=api_key)

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    cache_file = _cache_path(img_path, strategy)
    if not force_refresh:
        cached = _load_from_cache(cache_file)
        if cached:
            print(f"[Cache] Resultado carregado do cache para {img_path.name} ({strategy})")
            return cached

    prompt_files = {
        "zero_shot": "inspector_zero_shot.txt",
        "cot": "inspector_cot.txt",
        "few_shot": "inspector_few_shot.txt",
    }
    if strategy not in prompt_files:
        raise ValueError(f"Estratégia desconhecida: {strategy}. Use: {list(prompt_files.keys())}")

    prompt_template = _load_prompt(prompt_files[strategy])
    now = datetime.now(timezone.utc)
    inspection_id = f"INS_{now.strftime('%Y%m%d_%H%M%S')}_001"
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    zone_desc = ZONE_DESCRIPTIONS.get(zone_id, "Zona de produto")

    prompt = (prompt_template
              .replace("{{IMAGE_PATH}}", str(image_path))
              .replace("{{ZONE_ID}}", zone_id)
              .replace("{{ZONE_DESCRIPTION}}", zone_desc)
              .replace("{{DATETIME}}", now.strftime("%Y%m%d_%H%M%S"))
              .replace("{{TIMESTAMP}}", timestamp))

    image_data, media_type = _image_to_base64(img_path)

    print(f"[API] Chamando Gemini ({strategy}) para {img_path.name}...")
    raw_response = _call_gemini_with_backoff(
        client, model_name, prompt, image_data, media_type
    )

    result = _extract_json(raw_response)

    result.setdefault("inspection_id", inspection_id)
    result.setdefault("timestamp", timestamp)
    result.setdefault("image_path", str(image_path))
    result.setdefault("zone_id", zone_id)
    result.setdefault("overall_status", "ok")
    result.setdefault("issues", [])
    result.setdefault("shelf_fill_rate", 1.0)
    result.setdefault("products_detected", [])
    result.setdefault("model_reasoning", "")
    result["_strategy"] = strategy
    result["_model"] = model_name

    _save_to_cache(cache_file, result)

    inspections_dir = Path(os.getenv("INSPECTIONS_DIR", "./data/inspections"))
    inspections_dir.mkdir(parents=True, exist_ok=True)
    out_file = inspections_dir / f"{result['inspection_id']}_{strategy}.json"
    _save_to_cache(out_file, result)
    print(f"[OK] Inspeção guardada: {out_file.name}")

    return result


def inspect_batch(
    images_dir: str,
    zone_id: str = "Z_S1",
    strategy: str = "cot",
    extensions: tuple = (".jpg", ".jpeg", ".png", ".webp"),
    delay_s: float = 4.5,  
) -> list[dict]:
    img_dir = Path(images_dir)
    images = [p for p in img_dir.iterdir() if p.suffix.lower() in extensions]
    images.sort()

    results = []
    print(f"[Batch] {len(images)} imagens encontradas em {images_dir}")

    for i, img_path in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] {img_path.name}")
        try:
            result = inspect_image(str(img_path), zone_id=zone_id, strategy=strategy)
            results.append(result)
        except RuntimeError as e:
            print(f"[ERRO] {e}")
            break
        except Exception as e:
            print(f"[AVISO] Erro em {img_path.name}: {e}")
            results.append({"image_path": str(img_path), "error": str(e)})

        if i < len(images) - 1:
            time.sleep(delay_s)

    return results


def compare_strategies(
    image_path: str,
    zone_id: str = "Z_S1",
) -> dict:
    strategies = ["zero_shot", "cot", "few_shot"]
    results = {}

    for s in strategies:
        print(f"\n=== Estratégia: {s.upper()} ===")
        try:
            results[s] = inspect_image(image_path, zone_id=zone_id, strategy=s)
        except Exception as e:
            results[s] = {"error": str(e)}

    comparison = {
        "image": image_path,
        "zone_id": zone_id,
        "strategies": {}
    }
    for s, r in results.items():
        if "error" not in r:
            comparison["strategies"][s] = {
                "overall_status": r.get("overall_status"),
                "n_issues": len(r.get("issues", [])),
                "shelf_fill_rate": r.get("shelf_fill_rate"),
                "reasoning_length": len(r.get("model_reasoning", "")),
            }
        else:
            comparison["strategies"][s] = {"error": r["error"]}

    print("\n=== COMPARAÇÃO ===")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    return comparison


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Uso: python shelf_inspector.py <imagem> <zona> [estrategia]")
        print("Estrategias: zero_shot | cot | few_shot (default: cot)")
        sys.exit(1)

    img = sys.argv[1]
    zone = sys.argv[2]
    strat = sys.argv[3] if len(sys.argv) > 3 else "cot"

    result = inspect_image(img, zone_id=zone, strategy=strat)
    print(json.dumps(result, indent=2, ensure_ascii=False))