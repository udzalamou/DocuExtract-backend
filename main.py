import os
from pypdf import PdfReader
import pandas as pd
import io
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from openai import OpenAI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import json
import re
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

app = FastAPI(title="DocuExtract API")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Templates prédéfinis
TEMPLATES = {
    "facture": {
        "nom": "Facture",
        "description": "Extraire les données de factures fournisseurs",
        "champs": [
            {"nom": "nom_fournisseur", "label": "Nom du fournisseur", "type": "text"},
            {"nom": "numero_facture", "label": "Numéro de facture", "type": "text"},
            {"nom": "date_facture", "label": "Date de facture", "type": "date"},
            {"nom": "date_echeance", "label": "Date d'échéance", "type": "date"},
            {"nom": "montant_ht", "label": "Montant HT", "type": "number"},
            {"nom": "montant_ttc", "label": "Montant TTC", "type": "number"},
            {"nom": "tva", "label": "TVA", "type": "number"},
            {"nom": "devise", "label": "Devise", "type": "text"}
        ]
    },
    "releve_bancaire": {
        "nom": "Relevé Bancaire / Écritures",
        "description": "Extraire les opérations bancaires (date, libellé, débit, crédit)",
        "champs": [
            {"nom": "date", "label": "📅 Date", "type": "date"},
            {"nom": "date_valeur", "label": "📅 Date de valeur", "type": "date"},
            {"nom": "libelle", "label": "📝 Libellé", "type": "text"},
            {"nom": "debit", "label": "💸 Débit", "type": "number"},
            {"nom": "credit", "label": "💰 Crédit", "type": "number"},
            {"nom": "solde", "label": "💳 Solde", "type": "number"},
            {"nom": "reference_client", "label": "🔖 Réf. Client", "type": "text"},
            {"nom": "reference_mandat", "label": "🔖 Réf. Mandat", "type": "text"}
        ]
    },
    "contrat": {
        "nom": "Contrat",
        "description": "Extraire les informations de contrats",
        "champs": [
            {"nom": "partie_1", "label": "Partie 1", "type": "text"},
            {"nom": "partie_2", "label": "Partie 2", "type": "text"},
            {"nom": "date_signature", "label": "Date de signature", "type": "date"},
            {"nom": "date_debut", "label": "Date de début", "type": "date"},
            {"nom": "date_fin", "label": "Date de fin", "type": "date"},
            {"nom": "montant", "label": "Montant", "type": "number"},
            {"nom": "duree", "label": "Durée", "type": "text"}
        ]
    }
}

def normalize_date(date_str: str, year_hint: int = 2025) -> Optional[str]:
    if not date_str or str(date_str).lower() in ['null', 'none', '']:
        return None
    
    date_str = str(date_str).strip()
    
    match = re.match(r'^(\d{1,2})\.(\d{1,2})$', date_str)
    if match:
        day, month = match.groups()
        return f"{year_hint}-{int(month):02d}-{int(day):02d}"
    
    match = re.match(r'^(\d{1,2})/(\d{1,2})$', date_str)
    if match:
        day, month = match.groups()
        return f"{year_hint}-{int(month):02d}-{int(day):02d}"
    
    date_formats = [
        "%d.%m.%y", "%d/%m/%y", "%d.%m.%Y", "%d/%m/%Y",
        "%Y-%m-%d", "%d-%m-%Y", "%d %B %Y", "%d %b %Y", "%B %d, %Y"
    ]
    
    for fmt in date_formats:
        try:
            date_obj = datetime.strptime(date_str, fmt)
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    return date_str

def parse_french_number(value) -> Optional[float]:
    if value is None or value == '':
        return None
    
    try:
        value_str = str(value).replace(',', '.').replace(' ', '')
        return float(value_str)
    except (ValueError, TypeError):
        return None

def map_field_names(data: dict, all_fields: list) -> dict:
    field_mapping = {
        'valeur': 'date_valeur',
        'date_valeur': 'date_valeur',
        'date': 'date',
        'libelle': 'libelle',
        'libellé': 'libelle',
        'debit': 'debit',
        'débit': 'debit',
        'credit': 'credit',
        'crédit': 'credit',
        'solde': 'solde',
        'reference_client': 'reference_client',
        'ref_client': 'reference_client',
        'reference_mandat': 'reference_mandat',
        'ref_mandat': 'reference_mandat',
    }
    
    mapped_data = {}
    for key, value in data.items():
        official_name = field_mapping.get(key.lower(), key)
        if official_name in all_fields:
            mapped_data[official_name] = value
    
    return mapped_data

@app.get("/")
async def root():
    return {
        "message": "DocuExtract API is running",
        "version": "2.0",
        "status": "online"
    }

@app.get("/templates")
async def get_templates():
    return {"templates": TEMPLATES}

@app.post("/extract-custom")
async def extract_custom(
    files: List[UploadFile] = File(...),
    fields: str = Form(...),
    custom_fields: str = Form("[]"),
    template_type: str = Form("facture")
):
    try:
        fields_list = json.loads(fields)
        custom_fields_list = json.loads(custom_fields) if custom_fields else []
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Erreur de format JSON: {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"📥 Extraction personnalisée")
    print(f"📁 Template : {template_type}")
    print(f"📋 Champs demandés : {fields_list}")
    print(f"📄 Fichiers : {len(files)}")
    print(f"{'='*60}\n")
    
    all_fields = fields_list + custom_fields_list
    
    if not all_fields:
        raise HTTPException(status_code=400, detail="Aucun champ sélectionné")
    
    if template_type == "releve_bancaire":
        system_prompt = """Tu es un expert en extraction de relevés bancaires français.

RÈGLES IMPORTANTES:
1. Chaque ligne = UNE opération bancaire
2. Combine les libellés multi-lignes
3. Montants avec virgule (ex: 1 462,50)
4. Utilise EXACTEMENT ces noms de champs: date, date_valeur, libelle, debit, credit

Réponds avec un JSON valide."""
    else:
        system_prompt = f"""Tu es un expert en extraction de données.

Extrais: {', '.join(all_fields)}

Réponds avec un JSON valide."""
    
    results = []
    
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue
        
        print(f"\n📄 Traitement: {file.filename}")
        
        try:
            contents = await file.read()
            pdf_file = io.BytesIO(contents)
            reader = PdfReader(pdf_file)
            text = ""
            for i in range(min(5, len(reader.pages))):
                text += reader.pages[i].extract_text()
            
            print(f"   Texte extrait: {len(text)} caractères")
            
            if len(text.strip()) < 50:
                continue
            
            user_message = f"""DOCUMENT:
{text}

Extrais: {', '.join(all_fields)}"""
            
            completion = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            extracted_data = json.loads(completion.choices[0].message.content)
            print(f"   ✅ Données extraites")
            
            if template_type == "releve_bancaire":
                operations = []
                
                if isinstance(extracted_data, dict):
                    for key, value in extracted_data.items():
                        if isinstance(value, list):
                            operations.extend(value)
                        elif isinstance(value, dict):
                            operations.append(value)
                elif isinstance(extracted_data, list):
                    operations = extracted_data
                
                for op in operations:
                    op = map_field_names(op, all_fields)
                    
                    if 'date' in op:
                        op['date'] = normalize_date(op.get('date'))
                    if 'date_valeur' in op:
                        op['date_valeur'] = normalize_date(op.get('date_valeur'))
                    if 'debit' in op:
                        op['debit'] = parse_french_number(op.get('debit'))
                    if 'credit' in op:
                        op['credit'] = parse_french_number(op.get('credit'))
                    
                    op['fichier'] = file.filename
                    results.append(op)
                
                print(f"   ✅ {len(operations)} opération(s)")
            else:
                extracted_data = map_field_names(extracted_data, all_fields)
                
                for field in all_fields:
                    if field in extracted_data:
                        if 'date' in field.lower():
                            extracted_data[field] = normalize_date(extracted_data[field])
                        elif field in ['debit', 'credit', 'montant_ht', 'montant_ttc']:
                            extracted_data[field] = parse_french_number(extracted_data[field])
                
                extracted_data['fichier'] = file.filename
                results.append(extracted_data)
            
        except Exception as e:
            print(f"   ❌ ERREUR: {str(e)}")
            continue
    
    if not results:
        raise HTTPException(status_code=400, detail="Aucun document traité")
    
    print(f"\n📊 Création Excel... {len(results)} enregistrement(s)")
    
    df = pd.DataFrame(results)
    column_order = ['fichier'] + all_fields
    available_columns = [col for col in column_order if col in df.columns]
    df = df[available_columns]
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Données')
    
    output.seek(0)
    
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=extraction.xlsx"}
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Démarrage de DocuExtract API v2.0")
    print("📍 Serveur: http://0.0.0.0:8000")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)