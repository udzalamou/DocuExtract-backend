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
    allow_origins=["*"],  # En production, remplace par ton domaine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration OpenRouter avec variable d'environnement
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Vérifier que la clé API est configurée
if not os.getenv("OPENROUTER_API_KEY"):
    print("⚠️  WARNING: OPENROUTER_API_KEY not set in environment variables!")

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
    """Normalise les dates au format YYYY-MM-DD, même les formats courts comme '02.01'"""
    if not date_str or str(date_str).lower() in ['null', 'none', '']:
        return None
    
    date_str = str(date_str).strip()
    
    # Format court JJ.MM (ex: "02.01" → "2025-01-02")
    match = re.match(r'^(\d{1,2})\.(\d{1,2})$', date_str)
    if match:
        day, month = match.groups()
        return f"{year_hint}-{int(month):02d}-{int(day):02d}"
    
    # Format court JJ/MM (ex: "02/01" → "2025-01-02")
    match = re.match(r'^(\d{1,2})/(\d{1,2})$', date_str)
    if match:
        day, month = match.groups()
        return f"{year_hint}-{int(month):02d}-{int(day):02d}"
    
    # Formats complets
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
    """Convertit un nombre français (virgule) en float"""
    if value is None or value == '':
        return None
    
    try:
        value_str = str(value).replace(',', '.').replace(' ', '')
        return float(value_str)
    except (ValueError, TypeError):
        return None

def map_field_names(data: dict, all_fields: list) -> dict:
    """Map les noms de champs retournés par l'IA vers les noms attendus"""
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
    """Récupérer tous les templates disponibles"""
    return {"templates": TEMPLATES}

@app.post("/extract-custom")
async def extract_custom(
    files: List[UploadFile] = File(...),
    fields: str = Form(...),
    custom_fields: str = Form("[]"),
    template_type: str = Form("facture")
):
    """Extraction personnalisée selon les champs choisis"""
    
    # Parser les champs depuis les strings JSON
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
    
    # Prompt système adapté au type de document
    if template_type == "releve_bancaire":
        system_prompt = """Tu es un expert en extraction de relevés bancaires français.

Le document contient un tableau d'écritures bancaires avec les colonnes:
- DATE: Date de l'opération (format JJ.MM ou JJ.MM.AA)
- LIBELLE: Description détaillée de l'opération
- VALEUR: Date de valeur de l'opération
- DEBIT: Montant débité (colonne de gauche)
- CREDIT: Montant crédité (colonne de droite)

RÈGLES TRÈS IMPORTANTES:
1. Chaque ligne du tableau = UNE opération bancaire distincte
2. Les libellés peuvent être sur plusieurs lignes - COMBINE-LES en une seule chaîne
3. Les montants utilisent la VIRGULE comme séparateur décimal (ex: 1 462,50)
4. Si DEBIT est vide, mets null. Si CREDIT est vide, mets null.
5. Capture TOUTES les opérations, y compris le solde initial
6. Les références client/mandat sont dans le libellé (REF.CLIENT:, REF.MANDAT:)

NOMS DE CHAMPS EXACTS À UTILISER (très important):
- "date" pour la date d'opération
- "date_valeur" pour la date de valeur (PAS "valeur")
- "libelle" pour le libellé
- "debit" pour le débit
- "credit" pour le crédit
- "solde" pour le solde
- "reference_client" pour la référence client
- "reference_mandat" pour la référence mandat

Format de réponse: JSON avec une liste d'opérations.

Structure attendue pour CHAQUE opération:
{
  "date": "JJ.MM",
  "date_valeur": "JJ.MM.AA ou null",
  "libelle": "texte complet du libellé",
  "debit": nombre ou null,
  "credit": nombre ou null
}

Sois PRÉCIS et COMPLET dans l'extraction."""
    else:
        system_prompt = f"""Tu es un expert en extraction de données de documents.

Extrais UNIQUEMENT les champs suivants:
{chr(10).join([f'- {field}' for field in all_fields])}

Règles:
1. Dates: format YYYY-MM-DD
2. Montants: nombres décimaux (utilise le point)
3. Si un champ n'est pas trouvé: null
4. Réponds UNIQUEMENT avec un JSON valide."""
    
    results = []
    
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            print(f"⚠️  Fichier ignoré (pas un PDF): {file.filename}")
            continue
        
        print(f"\n📄 Traitement: {file.filename}")
        
        try:
            # Lecture du PDF avec pypdf
            contents = await file.read()
            pdf_file = io.BytesIO(contents)
            reader = PdfReader(pdf_file)
            text = ""
            for i in range(min(5, len(reader.pages))):
                text += reader.pages[i].extract_text()
            
            print(f"   Texte extrait: {len(text)} caractères")
            
            if len(text.strip()) < 50:
                print(f"   ⚠️  Trop peu de texte, ignoré")
                continue
            
            # Construction du message utilisateur
            user_message = f"""DOCUMENT À ANALYSER:
{text}

EXTRAIS les champs suivants au format JSON:
{chr(10).join([f'- {field}' for field in all_fields])}

IMPORTANT: Utilise EXACTEMENT ces noms de champs. Réponds UNIQUEMENT avec un JSON valide."""
            
            # Appel à l'API OpenAI
            print(f"   🤖 Appel à l'IA...")
            completion = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            # Parsing du JSON retourné
            extracted_data = json.loads(completion.choices[0].message.content)
            
            print(f"   ✅ Données extraites par l'IA")
            
            # Traitement selon le type de document
            if template_type == "releve_bancaire":
                operations = []
                
                if isinstance(extracted_data, dict):
                    for key, value in extracted_data.items():
                        if isinstance(value, list):
                            operations.extend(value)
                        elif isinstance(value, dict) and 'libelle' in str(value).lower():
                            operations.append(value)
                    if not operations and 'libelle' in str(extracted_data).lower():
                        operations.append(extracted_data)
                elif isinstance(extracted_data, list):
                    operations = extracted_data
                
                for op in operations:
                    op = map_field_names(op, all_fields)
                    
                    if 'date' in op and op['date']:
                        op['date'] = normalize_date(op['date'])
                    if 'date_valeur' in op and op['date_valeur']:
                        op['date_valeur'] = normalize_date(op['date_valeur'])
                    
                    if 'debit' in op:
                        op['debit'] = parse_french_number(op['debit'])
                    if 'credit' in op:
                        op['credit'] = parse_french_number(op['credit'])
                    if 'solde' in op:
                        op['solde'] = parse_french_number(op['solde'])
                    
                    op['fichier'] = file.filename
                    results.append(op)
                
                print(f"   ✅ {len(operations)} opération(s) extraites")
            else:
                extracted_data = map_field_names(extracted_data, all_fields)
                
                for field in all_fields:
                    if field in extracted_data and extracted_data[field]:
                        if 'date' in field.lower():
                            extracted_data[field] = normalize_date(extracted_data[field])
                        elif field in ['debit', 'credit', 'solde', 'montant_ht', 'montant_ttc', 'montant']:
                            extracted_data[field] = parse_french_number(extracted_data[field])
                
                extracted_data['fichier'] = file.filename
                results.append(extracted_data)
                print(f"   ✅ 1 document extrait")
            
        except Exception as e:
            print(f"   ❌ ERREUR: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        raise HTTPException(status_code=400, detail="Aucun document traité avec succès")
    
    print(f"\n{'='*60}")
    print(f"📊 Création du fichier Excel...")
    print(f"📈 Total: {len(results)} enregistrement(s)")
    print(f"{'='*60}\n")
    
    df = pd.DataFrame(results)
    
    column_order = ['fichier'] + all_fields
    available_columns = [col for col in column_order if col in df.columns]
    df = df[available_columns]
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Données extraites')
    
    output.seek(0)
    
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"}
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Démarrage de DocuExtract API v2.0")
    print("📍 Serveur: http://0.0.0.0:8000")
    print("📖 Docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)