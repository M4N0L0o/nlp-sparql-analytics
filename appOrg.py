from flask import Flask, request, render_template, jsonify
import pandas as pd
import rdflib
from rdflib import Graph, Namespace, URIRef
import json
from rank_bm25 import BM25Okapi
import spacy
import os
import re
import math
import urllib.parse
import numpy as np

app = Flask(__name__)

# Variables globales para los datos cargados y modelos
df_reviews = None
g = None
ex = Namespace("http://example.org/review-ontology#")
schema1 = Namespace("http://schema.org/")
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
bm25 = None
nlp = None

# Definir la ruta base de la carpeta data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, 'data')
CSV_FILE = os.path.join(DATA_FOLDER, 'df_cleaned.csv')
TTL_FILE = os.path.join(DATA_FOLDER, 'reviews_knowledge_graph.ttl')

known_product_names = []
# Mapeo de términos de consulta a URIs de predicados/clases en tu grafo

query_to_rdf_mapping = {
    "buy": ex.buy,
    "bought": ex.bought,
    "purchase": ex.purchase,
    "use": ex.use,
    "problem": ex.has_problem,
    "problem": ex.has_negative_sentiment_towards,
    "issue": ex.has_problem,
    "issue": ex.has_negative_sentiment_towards,
    "positive sentiment": ex.has_positive_sentiment_towards,
    "negative sentiment": ex.has_negative_sentiment_towards,
    "neutral sentiment": ex.has_neutral_sentiment_towards,
    "positive": ex.has_positive_sentiment_towards,
    "negative": ex.has_negative_sentiment_towards, 
    "neutral": ex.has_neutral_sentiment_towards,
    "feature": schema1.Feature,
    "users": schema1.Person,
    "person": schema1.Person,
    "love": ex.love,
    "get": ex.get,
    "like": ex.like,
    "recommend": ex.recommends,
    "recommends": ex.recommends,
    "not recommend": ex.does_not_recommend,
    "don't recommend": ex.does_not_recommend,
    "do not recommend": ex.does_not_recommend,
    "does not recommend": ex.does_not_recommend,
    "doesn't recommend": ex.does_not_recommend,
    "recommended": ex.recommends,
    "not recommended": ex.does_not_recommend,
    "don't recommended": ex.does_not_recommend,
    "do not recommended": ex.does_not_recommend,
    "does not recommended": ex.does_not_recommend,
    "doesn't recommended": ex.does_not_recommend,
    "not suggested": ex.does_not_recommend,
    "don't suggested": ex.does_not_recommend,
    "do not suggested": ex.does_not_recommend,
    "avoid": ex.does_not_recommend,
}


def load_data():
    global df_reviews, g, bm25, nlp

    print("Intentando cargar datos desde:", DATA_FOLDER)

    try:
        df_reviews = pd.read_csv(CSV_FILE, dtype={
            'reviews.title': str, 'reviews.text': str, 'name': str, 'base_product_name': str,
            'reviews.date': str, 'reviews.dateAdded': str, 'reviews.dateSeen': str,
            'categories': str, 'reviews.username': str,
            'extracted_events': str, 'extracted_relations_all': str
        })
        print(f"{CSV_FILE} cargado. Shape: {df_reviews.shape}.")
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} no encontrado en {DATA_FOLDER}. Asegúrate de que el archivo está ahí.")
        df_reviews = pd.DataFrame(columns=['reviews.text_lower', 'reviews.text', 'reviews.title', 'reviews.rating', 'reviews.doRecommend', 'name', 'base_product_name', 'reviews.username', 'extracted_events', 'extracted_relations_all'])
    except Exception as e:
        print(f"Error al cargar el archivo CSV: {e}")
        df_reviews = pd.DataFrame(columns=['reviews.text_lower', 'reviews.text', 'reviews.title', 'reviews.rating', 'reviews.doRecommend', 'name', 'base_product_name', 'reviews.username', 'extracted_events', 'extracted_relations_all'])


    required_csv_cols = ['reviews.text', 'reviews.title', 'reviews.rating', 'reviews.doRecommend', 'name', 'base_product_name', 'reviews.username', 'extracted_events', 'extracted_relations_all']
    for col in required_csv_cols:
        if col not in df_reviews.columns:
            print(f"Advertencia: Columna '{col}' no encontrada en CSV. Creando columna vacía.")
            df_reviews[col] = ''
        else:
            if df_reviews[col].dtype == 'object':
                 df_reviews[col] = df_reviews[col].astype(str).fillna('').str.strip()

    optional_csv_cols = ['reviews.date', 'reviews.dateAdded', 'reviews.dateSeen', 'categories']
    for col in optional_csv_cols:
        if col not in df_reviews.columns:
             df_reviews[col] = ''
        else:
             if df_reviews[col].dtype == 'object':
                  df_reviews[col] = df_reviews[col].astype(str).fillna('').str.strip()


    if 'reviews.text_lower' not in df_reviews.columns or df_reviews['reviews.text_lower'].isnull().all():
         df_reviews['reviews.text_lower'] = df_reviews['reviews.text'].str.lower()
    df_reviews['reviews.text_lower'] = df_reviews['reviews.text_lower'].astype(str).fillna('').str.strip()


    if 'reviews.rating' in df_reviews.columns:
         df_reviews['reviews.rating'] = pd.to_numeric(df_reviews['reviews.rating'], errors='coerce').fillna(np.nan)
    else:
         print("Advertencia: Columna 'reviews.rating' no encontrada. Creando columna con np.nan.")
         df_reviews['reviews.rating'] = np.nan

    if 'reviews.doRecommend' in df_reviews.columns:
         df_reviews['reviews.doRecommend'] = df_reviews['reviews.doRecommend'].astype(str).str.lower().map({'true': True, 'false': False}).fillna(False)
    else:
        print("Advertencia: Columna 'reviews.doRecommend' no encontrada. Creando columna con False.")
        df_reviews['reviews.doRecommend'] = False


    print(f"DataFrame final cargado. Shape: {df_reviews.shape}. Columnas: {df_reviews.columns.tolist()}")


    try:
        g = Graph()
        g.parse(TTL_FILE, format='turtle')
        print(f"Grafo de conocimiento desde {TTL_FILE} cargado correctamente. Tiene {len(g)} triples.")
    except FileNotFoundError:
        print(f"Error: {TTL_FILE} no encontrado en {DATA_FOLDER}. Asegúrate de que el archivo está ahí.")
        g = None
    except Exception as e:
        print(f"Error al cargar el archivo TTL: {e}")
        g = None


    if df_reviews is not None and not df_reviews.empty and 'reviews.text_lower' in df_reviews.columns:
        corpus = df_reviews['reviews.text_lower'].tolist()
        if any(corpus):
            tokenized_corpus = [doc.split() for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            print("Modelo BM25 inicializado.")
        else:
             print("La columna 'reviews.text_lower' está vacía o contiene solo strings vacíos.")
             bm25 = None
    else:
        print("No se pudo inicializar BM25. Verifica el DataFrame, su contenido y la columna 'reviews.text_lower'.")
        bm25 = None


    try:
        nlp = spacy.load("en_core_web_lg")
        
        # Crear lista de nombres de productos conocidos desde base_product_name
        global known_product_names
        if 'base_product_name' in df_reviews.columns:
            known_product_names = df_reviews['base_product_name'] \
                .dropna().astype(str).str.lower().str.strip().unique().tolist()
            print(f"{len(known_product_names)} nombres de productos cargados para detección.")
        else:
            known_product_names = []
            print("Advertencia: Columna 'base_product_name' no encontrada para extracción de nombres de producto.")

        print("Modelo spaCy 'en_core_web_lg' cargado.")
    except OSError:
        print("Modelo spaCy 'en_core_web_lg' no encontrado.")
        print("Por favor, descarga el modelo 'en_core_web_lg' manualmente ejecutando: !python -m spacy download en_core_web_lg")
        nlp = None
    except Exception as e:
        print(f"Otro error al intentar cargar el modelo de spaCy: {e}")
        nlp = None


# --- Funciones de Procesamiento de la Lógica ---

def process_query(query):
    """
    Procesa la consulta extrayendo entidades, verbos y frases negativas completas
    """
    if nlp is None:
        print("Modelo spaCy no cargado. La extracción de entidades/verbos no funcionará.")
        return [], []
    
    doc = nlp(query.lower())
    entities = [ent.text for ent in doc.ents]
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
    
    # NUEVO: Detectar frases negativas completas antes de la lematización
    negative_recommendation_phrases = []
    query_lower = query.lower()
    
    # Lista de frases negativas a detectar
    negative_phrases_to_detect = [
        'not recommend', 'not recommended', 'don\'t recommend', 'do not recommend',
        'does not recommend', 'doesn\'t recommend', 'not suggest', 'don\'t suggest',
        'do not suggest', 'avoid'
    ]
    
    for phrase in negative_phrases_to_detect:
        if phrase in query_lower:
            negative_recommendation_phrases.append(phrase)
            print(f"Frase negativa detectada: '{phrase}'")
    
    # Agregar las frases negativas como "entidades" especiales
    all_extracted_terms = entities + nouns + adjectives + verbs + negative_recommendation_phrases
    
    return all_extracted_terms, verbs

"""
def process_query(query):
    if nlp is None:
        print("Modelo spaCy no cargado. La extracción de entidades/verbos no funcionará.")
        return [], []
    doc = nlp(query.lower())
    entities = [ent.text for ent in doc.ents]
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
    return entities + nouns + adjectives, verbs
"""
def extract_known_products_from_query(query, product_name_list):
    """
    Extrae nombres de productos conocidos de una consulta usando coincidencias de palabras completas
    para evitar detecciones incorrectas como 'eco' dentro de 'recommended'
    """
    query_lower = query.lower()
    matched_products = []
    
    for product in product_name_list:
        if len(product) > 1:  # Mantener la validación de longitud mínima
            # Crear un patrón regex que busque la palabra completa
            # \b indica límite de palabra
            pattern = r'\b' + re.escape(product) + r'\b'
            
            try:
                if re.search(pattern, query_lower):
                    matched_products.append(product)
            except re.error as e:
                # En caso de error en la regex, hacer fallback a búsqueda simple
                # pero solo si el producto no está contenido en otra palabra
                if product in query_lower:
                    # Verificación adicional: asegurarse de que no esté dentro de otra palabra
                    words_in_query = query_lower.split()
                    if product in words_in_query:
                        matched_products.append(product)
    
    return matched_products



# Función para limpiar términos extraídos de URIs para comparación
def clean_uri_term(uri_str):
    if '#' in uri_str:
        term_id = uri_str.split('#')[-1]
        cleaned_term = re.sub(r'_+', ' ', term_id)
        cleaned_term = urllib.parse.unquote(cleaned_term)
        return cleaned_term.lower().strip()
    return None

# Función mejorada para consultar el grafo de conocimiento
def query_knowledge_graph_enhanced(entities, verbs):
    if g is None:
        print("Grafo de conocimiento no cargado.")
        return [], set()

    sparql_results = set()
    potential_queries = set()

    mapped_predicates_classes = []
    for term in verbs + entities:
        term_lower = term.lower().strip()
        if term_lower in query_to_rdf_mapping:
            mapping = query_to_rdf_mapping[term_lower]
            if isinstance(mapping, list):
                mapped_predicates_classes.extend(mapping)
            else:
                mapped_predicates_classes.append(mapping)

    mapped_entities = [
         ex[re.sub(r'\s+', '_', entity).replace("'", "%27").replace("(", "%28").replace(")", "%29").replace(",", "").replace("!", "%21")]
         for entity in entities
    ]

    print(f"Predicados/Clases mapeados de la consulta: {mapped_predicates_classes}")
    print(f"Entidades mapeadas de la consulta: {mapped_entities}")

    if mapped_predicates_classes and mapped_entities:
        for pred_cls_uri in mapped_predicates_classes:
            for ent_uri in mapped_entities:
                 potential_queries.add(f"SELECT ?s ?p ?o WHERE {{ ?s <{pred_cls_uri}> <{ent_uri}> . OPTIONAL {{ ?s ?p ?o }} }}")
                 potential_queries.add(f"SELECT ?s ?p ?o WHERE {{ <{ent_uri}> <{pred_cls_uri}> ?o . OPTIONAL {{ <{ent_uri}> ?p ?o }} }}")
                 potential_queries.add(f"SELECT ?s ?p ?o WHERE {{ {{ ?s <{pred_cls_uri}> <{ent_uri}> }} UNION {{ <{ent_uri}> <{pred_cls_uri}> ?o }} }}")


    if not potential_queries and mapped_entities:
         for ent_uri in mapped_entities:
              potential_queries.add(f"SELECT ?s ?p ?o WHERE {{ {{ ?s ?p <{ent_uri}> }} UNION {{ <{ent_uri}> ?p ?o }} }}")

    if not potential_queries and mapped_predicates_classes:
         for pc_uri in mapped_predicates_classes:
             potential_queries.add(f"SELECT ?s ?p ?o WHERE {{ {{ ?s <{pc_uri}> ?o }} UNION {{ ?s ?p <{pc_uri}> }} UNION {{ <{pc_uri}> ?p ?o }} }}")


    print(f"Consultas SPARQL potenciales generadas ({len(potential_queries)}): {list(potential_queries)[:5]}...")


    for query_str in potential_queries:
        try:
            for row in g.query(query_str):
                if len(row) > 0:
                    sparql_results.add(tuple(row))

        except Exception as e:
            print(f"Error al ejecutar SPARQL: {query_str[:100]}... Error: {e}")


    final_graph_terms_uris = set()
    for item in sparql_results:
         if isinstance(item, tuple):
              for element in item:
                   if isinstance(element, URIRef):
                        final_graph_terms_uris.add(str(element))
         elif isinstance(item, URIRef):
              final_graph_terms_uris.add(str(item))

    related_terms_from_graph_for_filtering = set()
    for uri_str in final_graph_terms_uris:
         cleaned = clean_uri_term(uri_str)
         if cleaned: related_terms_from_graph_for_filtering.add(cleaned)


    print(f"Términos limpios derivados del grafo para filtrar reseñas ({len(related_terms_from_graph_for_filtering)}): {list(related_terms_from_graph_for_filtering)[:10]}...")


    return list(sparql_results), related_terms_from_graph_for_filtering

# Función para convertir ratings a sentimiento categórico
def get_sentiment_from_rating(rating):
    if pd.notna(rating) and not isinstance(rating, str):
        try:
            rating_float = float(rating) if not (isinstance(rating, float) and np.isnan(rating)) else rating
            if pd.notna(rating_float):
                if rating_float >= 4.0:
                    return 'positive'
                elif rating_float == 3.0:
                    return 'neutral'
                elif rating_float >= 1.0 and rating_float <= 2.0:
                    return 'negative'
        except (ValueError, TypeError):
            pass
    return None

def detect_sentiment_intent(all_terms_combined_lower, query_to_rdf_mapping, original_query):
    """
    Detecta la intención de sentimiento de forma más precisa, priorizando términos de la consulta original
    """
    search_for_sentiment = None
    
    # Términos exactos para cada tipo de sentimiento
    positive_terms = {'positive', 'positive sentiment', 'good', 'great', 'excellent', 'love', 'amazing', 'fantastic', 'wonderful'}
    negative_terms = {'negative', 'negative sentiment', 'bad', 'poor', 'hate', 'terrible', 'awful', 'horrible', 'disappointing'}
    neutral_terms = {'neutral', 'neutral sentiment', 'okay', 'average', 'mediocre'}
    
    # Procesar la consulta original para detectar términos de sentimiento directamente
    original_query_lower = original_query.lower()
    original_query_words = set(re.findall(r'\b\w+\b', original_query_lower))
    
    print(f"DEBUG: Consulta original: '{original_query}'")
    print(f"DEBUG: Palabras en consulta original: {original_query_words}")
    print(f"DEBUG: Términos combinados del grafo: {all_terms_combined_lower}")
    
    # 1. PRIMERA PRIORIDAD: Verificar términos de sentimiento directamente en la consulta original
    original_positive_terms = positive_terms.intersection(original_query_words)
    original_negative_terms = negative_terms.intersection(original_query_words)
    original_neutral_terms = neutral_terms.intersection(original_query_words)
    
    if original_negative_terms:
        search_for_sentiment = 'negative'
        print(f"Consulta indica búsqueda de sentimiento negativo (términos en consulta original: {original_negative_terms}).")
        return search_for_sentiment
    elif original_positive_terms:
        search_for_sentiment = 'positive'
        print(f"Consulta indica búsqueda de sentimiento positivo (términos en consulta original: {original_positive_terms}).")
        return search_for_sentiment
    elif original_neutral_terms:
        search_for_sentiment = 'neutral'
        print(f"Consulta indica búsqueda de sentimiento neutral (términos en consulta original: {original_neutral_terms}).")
        return search_for_sentiment
    
    # 2. SEGUNDA PRIORIDAD: Verificar predicados RDF específicos que aparecen en el mapeo
    sentiment_predicates_in_query = set()
    for term_lower, mapping in query_to_rdf_mapping.items():
        if term_lower in all_terms_combined_lower:
            if hasattr(mapping, '__iter__') and not isinstance(mapping, str):
                # Si mapping es una lista o conjunto
                for m in mapping:
                    if m in {ex.has_positive_sentiment_towards, ex.has_negative_sentiment_towards, ex.has_neutral_sentiment_towards}:
                        sentiment_predicates_in_query.add(m)
            else:
                # Si mapping es un solo elemento
                if mapping in {ex.has_positive_sentiment_towards, ex.has_negative_sentiment_towards, ex.has_neutral_sentiment_towards}:
                    sentiment_predicates_in_query.add(mapping)
    
    if ex.has_negative_sentiment_towards in sentiment_predicates_in_query:
        search_for_sentiment = 'negative'
        print("Consulta indica búsqueda de sentimiento negativo (predicado RDF).")
        return search_for_sentiment
    elif ex.has_positive_sentiment_towards in sentiment_predicates_in_query:
        search_for_sentiment = 'positive'
        print("Consulta indica búsqueda de sentimiento positivo (predicado RDF).")
        return search_for_sentiment
    elif ex.has_neutral_sentiment_towards in sentiment_predicates_in_query:
        search_for_sentiment = 'neutral'
        print("Consulta indica búsqueda de sentimiento neutral (predicado RDF).")
        return search_for_sentiment
    
    # 3. TERCERA PRIORIDAD: Verificar términos que vienen del grafo (pero con precaución)
    # Solo si no hay términos de sentimiento en la consulta original
    graph_positive_terms = positive_terms.intersection(all_terms_combined_lower) - original_query_words
    graph_negative_terms = negative_terms.intersection(all_terms_combined_lower) - original_query_words
    graph_neutral_terms = neutral_terms.intersection(all_terms_combined_lower) - original_query_words
    
    if graph_negative_terms:
        # Verificar que realmente sea relevante para la consulta
        if len(graph_negative_terms) <= 2:  # Evitar demasiados términos del grafo
            search_for_sentiment = 'negative'
            print(f"Consulta indica búsqueda de sentimiento negativo (términos del grafo: {graph_negative_terms}).")
    elif graph_positive_terms:
        # Verificar que realmente sea relevante para la consulta
        if len(graph_positive_terms) <= 2:  # Evitar demasiados términos del grafo
            search_for_sentiment = 'positive'
            print(f"Consulta indica búsqueda de sentimiento positivo (términos del grafo: {graph_positive_terms}).")
    elif graph_neutral_terms:
        if len(graph_neutral_terms) <= 2:
            search_for_sentiment = 'neutral'
            print(f"Consulta indica búsqueda de sentimiento neutral (términos del grafo: {graph_neutral_terms}).")
    
    # Si no se detectó sentimiento, indicarlo claramente
    if search_for_sentiment is None:
        print("No se detectó intención de sentimiento específica en la consulta.")
    
    return search_for_sentiment

def detect_sentiment_intent(all_terms_combined_lower, query_to_rdf_mapping,original_query):
    """
    Versión estricta que SOLO considera términos de la consulta original
    """
    search_for_sentiment = None
    
    # Términos exactos para cada tipo de sentimiento
    positive_terms = {'positive', 'good', 'great', 'excellent', 'love', 'amazing', 'fantastic', 'wonderful'}
    negative_terms = {'negative', 'bad', 'poor', 'hate', 'terrible', 'awful', 'horrible', 'disappointing'}
    neutral_terms = {'neutral', 'okay', 'average', 'mediocre'}
    
    # Procesar solo la consulta original
    original_query_lower = original_query.lower()
    original_query_words = set(re.findall(r'\b\w+\b', original_query_lower))
    
    print(f"DEBUG: Detección estricta - Consulta: '{original_query}'")
    print(f"DEBUG: Palabras en consulta: {original_query_words}")
    
    # Verificar términos de sentimiento directamente en la consulta
    if negative_terms.intersection(original_query_words):
        search_for_sentiment = 'negative'
        matched_terms = negative_terms.intersection(original_query_words)
        print(f"Sentimiento negativo detectado (términos: {matched_terms}).")
    elif positive_terms.intersection(original_query_words):
        search_for_sentiment = 'positive'
        matched_terms = positive_terms.intersection(original_query_words)
        print(f"Sentimiento positivo detectado (términos: {matched_terms}).")
    elif neutral_terms.intersection(original_query_words):
        search_for_sentiment = 'neutral'
        matched_terms = neutral_terms.intersection(original_query_words)
        print(f"Sentimiento neutral detectado (términos: {matched_terms}).")
    else:
        print("No se detectó intención de sentimiento en la consulta original.")
    
    return search_for_sentiment

def detect_recommendation_intent(all_terms_combined_lower, query_to_rdf_mapping, original_query):
    """
    Versión mejorada para detectar intención de recomendación que prioriza frases completas
    """
    search_for_recommendation = None
    
    # Términos que indican recomendación negativa (frases completas primero)
    negative_recommendation_phrases = [
        'not recommend', 'not recommended', 'don\'t recommend', 'do not recommend',
        'does not recommend', 'doesn\'t recommend', 'not suggest', 'don\'t suggest',
        'do not suggest', 'avoid'
    ]
    
    # Términos que indican recomendación positiva
    positive_recommendation_terms = {
        'recommend', 'recommends', 'recommended', 'suggest', 'suggests', 
        'suggested', 'advise', 'endorse', 'endorses'
    }
    
    original_query_lower = original_query.lower()
    
    print(f"DEBUG: Detección de recomendación mejorada - Consulta: '{original_query}'")
    
    # 1. PRIMERA PRIORIDAD: Verificar frases negativas completas en la consulta original
    for negative_phrase in negative_recommendation_phrases:
        if negative_phrase in original_query_lower:
            search_for_recommendation = 'negative'
            print(f"Recomendación negativa detectada (frase completa: '{negative_phrase}').")
            return search_for_recommendation
    
    # 2. SEGUNDA PRIORIDAD: Verificar si hay términos negativos cerca de palabras de recomendación
    import re
    # Buscar patrones como "not ... recommend", "don't ... recommend", etc.
    negative_patterns = [
        r'\bnot\s+\w*\s*recommend\w*',
        r'\bdon\'?t\s+\w*\s*recommend\w*', 
        r'\bdo\s+not\s+\w*\s*recommend\w*',
        r'\bdoes\s+not\s+\w*\s*recommend\w*',
        r'\bdoesn\'?t\s+\w*\s*recommend\w*'
    ]
    
    for pattern in negative_patterns:
        if re.search(pattern, original_query_lower):
            search_for_recommendation = 'negative'
            matched = re.search(pattern, original_query_lower).group()
            print(f"Recomendación negativa detectada (patrón: '{matched}').")
            return search_for_recommendation
    
    # 3. TERCERA PRIORIDAD: Verificar términos positivos de recomendación (solo si no hay negación)
    original_query_words = set(re.findall(r'\b\w+\b', original_query_lower))
    if positive_recommendation_terms.intersection(original_query_words):
        # Verificar que no haya palabras negativas cerca
        negation_words = {'not', 'don', 'dont', 'doesn', 'doesnt', 'do', 'never', 'avoid'}
        if not negation_words.intersection(original_query_words):
            search_for_recommendation = 'positive'
            matched_terms = positive_recommendation_terms.intersection(original_query_words)
            print(f"Recomendación positiva detectada (términos: {matched_terms}).")
            return search_for_recommendation
    
    # 4. CUARTA PRIORIDAD: Verificar predicados RDF específicos
    recommendation_predicates_in_query = set()
    for term_lower, mapping in query_to_rdf_mapping.items():
        if term_lower in all_terms_combined_lower:
            if hasattr(mapping, '__iter__') and not isinstance(mapping, str):
                for m in mapping:
                    if m in {ex.recommends, ex.does_not_recommend}:
                        recommendation_predicates_in_query.add(m)
            else:
                if mapping in {ex.recommends, ex.does_not_recommend}:
                    recommendation_predicates_in_query.add(mapping)
    
    if ex.does_not_recommend in recommendation_predicates_in_query:
        search_for_recommendation = 'negative'
        print("Recomendación negativa detectada (predicado RDF).")
    elif ex.recommends in recommendation_predicates_in_query:
        search_for_recommendation = 'positive'
        print("Recomendación positiva detectada (predicado RDF).")
    
    if search_for_recommendation is None:
        print("No se detectó intención de recomendación específica en la consulta.")
    
    return search_for_recommendation

# Función para encontrar reseñas relevantes (prioriza datos extraídos y atributos)
def find_relevant_reviews_prioritizing_extracted(graph_raw_results, related_terms_from_graph, df_reviews, query_entities, query_verbs,original_query):
    if df_reviews is None or df_reviews.empty:
        print("DataFrame de reseñas no cargado o vacío.")
        return pd.DataFrame()

    required_cols_for_filter = ['reviews.text_lower', 'name', 'base_product_name', 'reviews.username', 'reviews.rating', 'reviews.doRecommend', 'extracted_events', 'extracted_relations_all']
    for col in required_cols_for_filter:
        if col not in df_reviews.columns:
            print(f"Columna '{col}' necesaria no encontrada en el DataFrame. Esta lógica la requiere.")
            return pd.DataFrame(columns=required_cols_for_filter + ['bm25_score'])


    relevant_reviews_indices = set()

    query_terms_lower = {term.lower().strip() for term in query_entities + query_verbs if term and term.strip()}

    # --- Paso de Identificación de Atributos Relevantes en la Consulta/Grafo ---
    all_terms_combined_lower = related_terms_from_graph.union(query_terms_lower)

    # Detectar intención de recomendación (positiva/negativa)
    search_for_recommendation = detect_recommendation_intent(all_terms_combined_lower, query_to_rdf_mapping, original_query)

    # Detectar intención de sentimiento de forma más precisa
    search_for_sentiment = detect_sentiment_intent(all_terms_combined_lower, query_to_rdf_mapping, original_query)

    # --- Paso de Filtrado por Producto o Usuario Específico ---
    all_potential_names_for_identification = related_terms_from_graph.union(query_terms_lower)

    specific_product_df = pd.DataFrame()
    identified_product_name = None
    specific_user_df = pd.DataFrame()
    identified_user_name = None

    df_reviews_subset = df_reviews

    if all_potential_names_for_identification and not df_reviews.empty:
        regex_names = [re.escape(name) for name in all_potential_names_for_identification if name]
        if regex_names:
            regex_pattern_names = '|'.join(regex_names)
            try:
                # match_in_name = df_reviews['name'].str.lower().str.contains(regex_pattern_names, na=False)
                match_in_base_name = df_reviews['base_product_name'].str.lower().str.contains(regex_pattern_names, na=False)
               # match = df_reviews[match_in_name | match_in_base_name]
                match = df_reviews[ match_in_base_name]

                if not match.empty:
                    specific_product_df = match.copy()
                    identified_product_name = match['base_product_name'].iloc[0]
                    print(f"Producto específico identificado: '{identified_product_name}'")

            except re.error as e:
                print(f"Error en la expresión regular para nombres de productos: {e}")

        if specific_product_df.empty and 'reviews.username' in df_reviews.columns:
             for potential_name in all_potential_names_for_identification:
                  match_user = df_reviews[df_reviews['reviews.username'].str.lower() == potential_name.lower()]
                  if not match_user.empty:
                       specific_user_df = match_user.copy()
                       identified_user_name = match_user['reviews.username'].iloc[0]
                       print(f"Usuario específico identificado: '{identified_user_name}'")
                       break


    if not specific_product_df.empty:
        df_reviews_subset = specific_product_df
        print(f"Limitando búsqueda a reseñas de '{identified_product_name}' ({len(df_reviews_subset)} reseñas).")
    elif not specific_user_df.empty:
         df_reviews_subset = specific_user_df
         print(f"Limitando búsqueda a reseñas de usuario '{identified_user_name}' ({len(df_reviews_subset)} reseñas).")

    print(f"Subconjunto de reseñas inicial para evaluación ({len(df_reviews_subset)}).")


    # --- Iterar sobre el subconjunto de reseñas y determinar su relevancia ---
    for index, row in df_reviews_subset.iterrows():
        is_relevant = False

        # Intentar cargar los datos extraídos
        events = []
        events_raw = row.get("extracted_events", "")
        if pd.notna(events_raw) and isinstance(events_raw, str) and events_raw.strip():
            try:
                events = json.loads(events_raw)
                if not isinstance(events, list):
                    events = []
            except (json.JSONDecodeError, Exception):
                pass

        relations = []
        relations_raw = row.get("extracted_relations_all", "")
        if pd.notna(relations_raw) and isinstance(relations_raw, str) and relations_raw.strip():
            try:
                relations = json.loads(relations_raw)
                if not isinstance(relations, list):
                    relations = []
            except (json.JSONDecodeError, Exception):
                pass

        # Si no hay datos extraídos, esta reseña no es relevante en este enfoque
        if not events and not relations:
             continue


        # --- 1. Verificación Explícita del Nombre del Producto/Usuario Identificado en DATOS EXTRAÍDOS ---
        identified_entity_in_extracted_data = False
        if identified_product_name:
            potential_names_to_check = all_potential_names_for_identification
            if potential_names_to_check:
                 # Ejemplo genérico: buscar si algún término potencial aparece en valores string dentro de eventos o relaciones
                if any(
                     (isinstance(rel.get('subject'), str) and rel.get('subject').lower() in potential_names_to_check) or
                     (isinstance(rel.get('object'), str) and rel.get('object').lower() in potential_names_to_check)
                     for rel in relations if isinstance(rel, dict) # Asegurarse de que es un diccionario de relación
                 ):
                      identified_entity_in_extracted_data = True
                 # === Lógica más robusta buscaría en claves/elementos específicos, ej: ===
                 # if any(isinstance(evt, dict) and str(evt.get('product_name', '')).lower() in potential_names_to_check for evt in events):
                 #      identified_entity_in_extracted_data = True
                 # if any(isinstance(rel, (list,tuple)) and len(rel) > 2 and isinstance(rel[2], str) and str(rel[2]).lower() in potential_names_to_check for rel in relations): # Asumiendo objeto es producto
                 #      identified_entity_in_extracted_data = True


        elif identified_user_name:
             # === ADAPTA ESTA LÓGICA para buscar nombres/términos de usuario en TUS eventos/relaciones ===
             # Ejemplo genérico:
             if any(
                 isinstance(rel.get('subject'), str) and rel.get('subject').lower() == identified_user_name.lower()
                 for rel in relations if isinstance(rel, dict)
             ):
                  identified_entity_in_extracted_data = True
             # === Lógica más robusta buscaría en claves/elementos específicos, ej: ===
             # if any(isinstance(evt, dict) and str(evt.get('user_id', '')).lower() == identified_user_name.lower() for evt in events):
             #      identified_entity_in_extracted_data = True
             # if any(isinstance(rel, (list,tuple)) and len(rel) > 0 and isinstance(rel[0], str) and str(rel[0]).lower() == identified_user_name.lower() for rel in relations): # Asumiendo sujeto es usuario
             #      identified_entity_in_extracted_data = True

        # Si se identificó una entidad pero NO se encuentra en los datos extraídos, no es relevante en este enfoque.
        if (identified_product_name or identified_user_name) and not identified_entity_in_extracted_data:
             continue


        # --- 2. Emparejamiento por Atributos del CSV (Mantenemos como criterio) ---
        attribute_match = False
        review_sentiment = get_sentiment_from_rating(row.get('reviews.rating'))

        if search_for_recommendation and row.get('reviews.doRecommend') is True:
             attribute_match = True

        if search_for_sentiment and review_sentiment == search_for_sentiment:
             attribute_match = True

        # --- 3. Emparejamiento por Coincidencia en OTROS Términos en DATOS EXTRAÍDOS ---
        extracted_data_other_terms_match = False
        other_terms_for_extracted_search = all_terms_combined_lower.copy() # Copiar para no modificar el original

        if identified_product_name: other_terms_for_extracted_search.discard(identified_product_name.lower())
        if identified_user_name: other_terms_for_extracted_search.discard(identified_user_name.lower())
        other_terms_for_extracted_search = {term for term in other_terms_for_extracted_search if len(term) > 1 and term.strip()}

        if other_terms_for_extracted_search:
             # === ADAPTA ESTA LÓGICA para buscar otros términos en TUS eventos/relaciones ===
             # Ejemplo genérico: buscar si algún otro término aparece en valores string
              if any(
                 (isinstance(rel.get('predicate'), str) and rel.get('predicate').lower() in other_terms_for_extracted_search) or
                 (isinstance(rel.get('subject'), str) and rel.get('subject').lower() in other_terms_for_extracted_search and not identified_product_name and not identified_user_name) or # Solo buscar subject/object si no es la entidad principal que ya validamos
                 (isinstance(rel.get('object'), str) and rel.get('object').lower() in other_terms_for_extracted_search and not identified_product_name and not identified_user_name)
                 for rel in relations if isinstance(rel, dict)
             ) or any(
                 (isinstance(evt.get('type'), str) and evt.get('type').lower() in other_terms_for_extracted_search) or
                 (isinstance(evt.get('verb'), str) and evt.get('verb').lower() in other_terms_for_extracted_search) or
                 (isinstance(evt.get('lemma'), str) and evt.get('lemma').lower() in other_terms_for_extracted_search)
                 for evt in events if isinstance(evt, dict)
             ):
                  extracted_data_other_terms_match = True
             # === Lógica más robusta buscaría en claves/elementos específicos, ej: ===
             # if any(isinstance(evt, dict) and any(str(value).lower() in other_terms_for_extracted_search for key, value in evt.items() if isinstance(value, str)) for evt in events):
             #      extracted_data_other_terms_match = True
             # if any(isinstance(rel, (list,tuple)) and any(isinstance(elem, str) and str(elem).lower() in other_terms_for_extracted_search for elem in rel) for rel in relations):
             #      extracted_data_other_terms_match = True


        # --- Determinación Final de Relevancia (Priorizando Datos Extraídos) ---

        # Criterio Principal: Si se busca SENTIMIENTO O RECOMENDACIÓN específicos
        if search_for_sentiment or search_for_recommendation:
            # Debe haber coincidencia de atributos Y
            # (Hay coincidencia de otros términos en datos extraídos O (No hay otros términos para buscar Y la entidad identificada está en datos extraídos))
            if attribute_match:
                 if extracted_data_other_terms_match:
                      is_relevant = True
                 elif not other_terms_for_extracted_search and (identified_product_name or identified_user_name) and identified_entity_in_extracted_data:
                      is_relevant = True


        # Criterio Secundario: Si NO se busca sentimiento ni recomendación específicos
        else:
             # Relevancia por coincidencia de otros términos en datos extraídos
             if extracted_data_other_terms_match:
                  is_relevant = True
             # Caso especial: No hay otros términos para buscar Y se filtró por producto/usuario Y la entidad está en los datos extraídos
             elif not other_terms_for_extracted_search and (identified_product_name or identified_user_name) and identified_entity_in_extracted_data:
                  is_relevant = True


        # --- Fallback a búsqueda en Texto Libre (Opcional pero útil) ---
        # Solo si no se consideró relevante basándose en datos extraídos y atributos
        if not is_relevant:
             # Usamos todos los términos combinados (incluido producto/usuario) para el fallback en texto libre
             all_terms_for_fallback_text_search = all_terms_combined_lower
             all_terms_for_fallback_text_search = {term for term in all_terms_for_fallback_text_search if len(term) > 1 and term.strip()}

             if all_terms_for_fallback_text_search:
                 regex_terms = [re.escape(term) for term in all_terms_for_fallback_text_search if term]
                 if regex_terms:
                      regex_pattern = '|'.join(regex_terms)
                      try:
                          if re.search(regex_pattern, str(row.get('reviews.text_lower', ''))):
                               is_relevant = True
                      except re.error: pass


        # Debugging
        # if index == ID_DE_UNA_RESEÑA_ESPECIFICA_PARA_DEBUG:
        #      print(f"DEBUG Reseña {index}: Relevant={is_relevant}, AttrMatch={attribute_match}, ExtractedOtherTermsMatch={extracted_data_other_terms_match}, EntityInExtracted={identified_entity_in_extracted_data}, SearchSentiment={search_for_sentiment}, SearchRecommend={search_for_recommendation}, IdentifiedEntity={identified_product_name or identified_user_name}, OtherTermsExtractedCount={len(other_terms_for_extracted_search)}")


        if is_relevant:
            relevant_reviews_indices.add(index)

    relevant_reviews_df = df_reviews.loc[list(relevant_reviews_indices)].copy()

    print(f"Reseñas finales consideradas relevantes para ranking ({len(relevant_reviews_df)}).")

    return relevant_reviews_df


def rank_reviews_with_bm25(query, relevant_reviews_df):
    if relevant_reviews_df.empty or bm25 is None:
        print("DataFrame de reseñas relevantes vacío o modelo BM25 no inicializado. No se puede rankear.")
        return relevant_reviews_df.assign(bm25_score=[])

    reviews_to_rank = relevant_reviews_df['reviews.text_lower'].astype(str).fillna('').tolist()

    if not any(reviews_to_rank):
        print("No hay textos de reseñas válidos para rankear.")
        return relevant_reviews_df.assign(bm25_score=0.0).sort_values(by='bm25_score', ascending=False)

    tokenized_query = query.lower().split()

    tokenized_relevant_reviews = [doc.split() for doc in reviews_to_rank]
    bm25_relevant = BM25Okapi(tokenized_relevant_reviews)

    scores = bm25_relevant.get_scores(tokenized_query)

    if len(scores) == len(relevant_reviews_df):
        ranked_reviews = relevant_reviews_df.copy()
        ranked_reviews['bm25_score'] = scores
        ranked_reviews = ranked_reviews.sort_values(by='bm25_score', ascending=False)
        return ranked_reviews
    else:
        print(f"Error: El número de scores BM25 ({len(scores)}) no coincide con el número de reseñas relevantes ({len(relevant_reviews_df)}).")
        return relevant_reviews_df.assign(bm25_score=0.0).sort_values(by='bm25_score', ascending=False)
    
# --- Rutas de Flask ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search_ajax', methods=['POST'])
def search_ajax():
    query = request.form.get('query', '')

    # 1. Procesar la consulta
    entities, verbs = process_query(query)

    # 1.1 Detectar nombres de productos conocidos desde la consulta
    detected_products = extract_known_products_from_query(query, known_product_names)
    if detected_products:
        print(f"Nombres de productos detectados en la consulta: {detected_products}")
        entities += detected_products  # agregamos como entidades extra

    print(f"Consulta recibida: '{query}'")
    print(f"Entidades/Sustantivos detectados: {entities}")
    print(f"Verbos detectados: {verbs}")


    # 2. Consultar el grafo de conocimiento
    graph_raw_results, related_terms_from_graph = query_knowledge_graph_enhanced(entities, verbs)
    print(f"Términos derivados del grafo para búsqueda ({len(related_terms_from_graph)}): {list(related_terms_from_graph)[:10]}...")


    # 3. Encontrar reseñas relevantes (prioriza datos extraídos y atributos)
    relevant_reviews = find_relevant_reviews_prioritizing_extracted(graph_raw_results, related_terms_from_graph, df_reviews, entities, verbs, query)


    print(f"Total de reseñas relevantes encontradas: {len(relevant_reviews)}")

    # 4. Rankear las reseñas
    ranked_reviews = rank_reviews_with_bm25(query, relevant_reviews)

    # 5. Preparar los resultados en formato JSON para el frontend
    results = []
    if not ranked_reviews.empty:
        max_results = 50
        for index, row in ranked_reviews.head(max_results).iterrows():
            score = row.get('bm25_score')
            score_str = f"{score:.4f}" if pd.notna(score) else 'N/A'

            rating = row.get('reviews.rating')
            rating_str = str(rating) if pd.isna(rating) or not isinstance(rating, (int, float)) or (isinstance(rating, float) and math.isnan(rating)) else (f"{rating:.1f}" if isinstance(rating, float) else str(rating))


            results.append({
                'score': score_str,
                'title': str(row.get('reviews.title', 'Sin título')).strip() or 'Sin título',
                'text': str(row.get('reviews.text', 'Sin texto')).strip() or 'Sin texto',
                'product': str(row.get('base_product_name', 'Producto desconocido')).strip() or 'Producto desconocido',
                'author': str(row.get('reviews.username', 'Autor desconocido')).strip() or 'Autor desconocido',
                'rating': rating_str
            })

    return jsonify(results)

# MODIFICACIONES PRINCIPALES PARA TU CÓDIGO

# 1. MODIFICAR LA FUNCIÓN convert_graph_results_to_d3 (REEMPLAZAR COMPLETAMENTE)
def convert_graph_results_to_d3(graph_raw_results, entities, verbs, query):
    """
    Convierte los resultados del grafo RDF y datos extraídos de reviews a formato D3.js
    con enfoque en relaciones semánticas
    """
    nodes = {}
    links = []
    
    try:
        # 1. PROCESAR RESULTADOS DEL GRAFO RDF SPARQL
        processed_sparql_relations = set()
        
        for result in graph_raw_results:
            if isinstance(result, tuple) and len(result) >= 3:
                subject, predicate, obj = result[0], result[1], result[2]
                
                subject_name = clean_uri_term(str(subject)) if subject else None
                predicate_name = clean_uri_term(str(predicate)) if predicate else None
                object_name = clean_uri_term(str(obj)) if obj else None
                
                if subject_name and object_name and predicate_name:
                    # Evitar duplicados
                    relation_key = (subject_name, predicate_name, object_name)
                    if relation_key in processed_sparql_relations:
                        continue
                    processed_sparql_relations.add(relation_key)
                    
                    # Agregar nodos con tipos semánticos
                    if subject_name not in nodes:
                        nodes[subject_name] = {
                            'id': subject_name,
                            'label': subject_name.title(),
                            'type': determine_node_type(subject_name, predicate_name, 'subject'),
                            'count': 1,
                            'size': 10,
                            'sentiment': None
                        }
                    else:
                        nodes[subject_name]['count'] += 1
                    
                    if object_name not in nodes:
                        nodes[object_name] = {
                            'id': object_name,
                            'label': object_name.title(),
                            'type': determine_node_type(object_name, predicate_name, 'object'),
                            'count': 1,
                            'size': 10,
                            'sentiment': extract_sentiment_from_name(object_name)
                        }
                    else:
                        nodes[object_name]['count'] += 1
                    
                    # Agregar enlace con información semántica
                    links.append({
                        'source': subject_name,
                        'target': object_name,
                        'relation': predicate_name,
                        'label': format_relation_label(predicate_name),
                        'type': determine_relation_type(predicate_name),
                        'weight': 1,
                        'sentiment': extract_sentiment_from_relation(predicate_name)
                    })
        
        # 2. ENRIQUECER CON DATOS EXTRAÍDOS DE REVIEWS
        semantic_data = extract_semantic_data_from_reviews(entities, verbs, query)
        
        # Agregar nodos de productos, usuarios y conceptos
        for node_id, node_info in semantic_data['nodes'].items():
            if node_id not in nodes:
                nodes[node_id] = node_info
            else:
                # Combinar información existente
                nodes[node_id]['count'] += node_info.get('count', 1)
                nodes[node_id]['size'] = max(nodes[node_id].get('size', 10), node_info.get('size', 10))
                if node_info.get('sentiment') and not nodes[node_id].get('sentiment'):
                    nodes[node_id]['sentiment'] = node_info['sentiment']
        
        # Agregar enlaces semánticos de reviews
        links.extend(semantic_data['links'])
        
        # 3. PROCESAR TÉRMINOS DE LA CONSULTA ORIGINAL
        query_terms = set([term.lower().strip() for term in entities + verbs if term and len(term.strip()) > 1])
        query_lower = query.lower()
        
        for node_id, node_data in nodes.items():
            if node_id in query_lower or node_id in query_terms:
                node_data['type'] = 'query_match'
                node_data['count'] += 3  # Mayor peso
                node_data['size'] = max(node_data.get('size', 10), 15)
        
        # 4. CALCULAR MÉTRICAS DE CENTRALIDAD
        calculate_node_importance(nodes, links)
        
        # 5. FILTRAR Y LIMITAR RESULTADOS
        # Mantener solo los nodos más relevantes
        filtered_nodes, filtered_links = filter_most_relevant_nodes(nodes, links, max_nodes=50)
        
        return {
            'nodes': list(filtered_nodes.values()),
            'links': filtered_links,
            'query': query,
            'total_nodes': len(filtered_nodes),
            'total_links': len(filtered_links),
            'semantic_summary': generate_semantic_summary(filtered_nodes, filtered_links),
            'node_types': get_unique_node_types(filtered_nodes),
            'relation_types': get_unique_relation_types(filtered_links)
        }
        
    except Exception as e:
        print(f"Error al convertir datos del grafo: {e}")
        return create_fallback_graph(entities, verbs, query)

# 2. NUEVAS FUNCIONES AUXILIARES PARA AGREGAR

def extract_semantic_data_from_reviews(entities, verbs, query):
    """
    Extrae datos semánticos directamente de las reviews procesadas
    """
    global df_reviews
    nodes = {}
    links = []
    
    if df_reviews is None or df_reviews.empty:
        return {'nodes': nodes, 'links': links}
    
    # Buscar reviews relevantes (versión simplificada)
    relevant_reviews = find_reviews_for_semantic_graph(entities, verbs, query)
    
    for _, row in relevant_reviews.iterrows():
        # Extraer datos de eventos y relaciones JSON
        events = extract_events_from_row(row)
        relations = extract_relations_from_row(row)
        
        # Procesar eventos
        for event in events:
            process_event_for_graph(event, nodes, links, row)
        
        # Procesar relaciones
        for relation in relations:
            process_relation_for_graph(relation, nodes, links, row)
        
        # Agregar nodo de producto
        product_name = str(row.get('base_product_name', '')).strip().lower()
        if product_name and product_name != 'nan':
            if product_name not in nodes:
                nodes[product_name] = {
                    'id': product_name,
                    'label': product_name.title(),
                    'type': 'product',
                    'count': 1,
                    'size': 12,
                    'sentiment': get_sentiment_from_rating(row.get('reviews.rating')),
                    'rating': row.get('reviews.rating'),
                    'recommend': row.get('reviews.doRecommend')
                }
            else:
                nodes[product_name]['count'] += 1
                nodes[product_name]['size'] += 2
        
        # Agregar nodo de usuario
        username = str(row.get('reviews.username', '')).strip().lower()
        if username and username != 'nan' and len(username) > 2:
            if username not in nodes:
                nodes[username] = {
                    'id': username,
                    'label': f"User: {username}",
                    'type': 'user',
                    'count': 1,
                    'size': 8,
                    'sentiment': get_sentiment_from_rating(row.get('reviews.rating'))
                }
            else:
                nodes[username]['count'] += 1
        
        # Crear enlaces usuario-producto
        if product_name and username and product_name != 'nan' and username != 'nan':
            sentiment = get_sentiment_from_rating(row.get('reviews.rating'))
            recommend = row.get('reviews.doRecommend')
            
            relation_type = 'reviewed'
            if recommend:
                relation_type = 'recommends'
            elif recommend is False:
                relation_type = 'does_not_recommend'
            
            links.append({
                'source': username,
                'target': product_name,
                'relation': relation_type,
                'label': relation_type.replace('_', ' ').title(),
                'type': 'user_product',
                'weight': 2,
                'sentiment': sentiment,
                'rating': row.get('reviews.rating')
            })
    
    return {'nodes': nodes, 'links': links}

def find_reviews_for_semantic_graph(entities, verbs, query, max_reviews=20):
    """
    Encuentra reviews relevantes para construir el grafo semántico
    """
    global df_reviews
    
    if df_reviews is None or df_reviews.empty:
        return pd.DataFrame()
    
    # Usar la función existente pero con límite
    relevant_reviews = find_relevant_reviews_prioritizing_extracted(
        [], set(), df_reviews, entities, verbs, query
    )
    
    if relevant_reviews.empty:
        # Fallback: buscar por términos en texto
        all_terms = set([term.lower() for term in entities + verbs if term])
        if all_terms:
            mask = df_reviews['reviews.text_lower'].str.contains('|'.join(all_terms), na=False)
            relevant_reviews = df_reviews[mask]
    
    return relevant_reviews.head(max_reviews)

def extract_events_from_row(row):
    """Extrae eventos del JSON almacenado en la fila"""
    events = []
    events_raw = row.get("extracted_events", "")
    if pd.notna(events_raw) and isinstance(events_raw, str) and events_raw.strip():
        try:
            events = json.loads(events_raw)
            if not isinstance(events, list):
                events = []
        except (json.JSONDecodeError, Exception):
            pass
    return events

def extract_relations_from_row(row):
    """Extrae relaciones del JSON almacenado en la fila"""
    relations = []
    relations_raw = row.get("extracted_relations_all", "")
    if pd.notna(relations_raw) and isinstance(relations_raw, str) and relations_raw.strip():
        try:
            relations = json.loads(relations_raw)
            if not isinstance(relations, list):
                relations = []
        except (json.JSONDecodeError, Exception):
            pass
    return relations

def process_event_for_graph(event, nodes, links, row):
    """Procesa un evento individual para el grafo"""
    if not isinstance(event, dict):
        return
    
    # Extraer información del evento (adaptar según tu estructura)
    event_type = str(event.get('type', '')).lower().strip()
    verb = str(event.get('verb', '')).lower().strip()
    lemma = str(event.get('lemma', '')).lower().strip()
    
    # Agregar nodo de acción/evento
    action = verb or lemma or event_type
    if action and len(action) > 2:
        if action not in nodes:
            nodes[action] = {
                'id': action,
                'label': action.title(),
                'type': 'action',
                'count': 1,
                'size': 8,
                'sentiment': None
            }
        else:
            nodes[action]['count'] += 1

def process_relation_for_graph(relation, nodes, links, row):
    """Procesa una relación individual para el grafo"""
    if not isinstance(relation, dict):
        return
    
    # Extraer sujeto, predicado, objeto (adaptar según tu estructura)
    subject = str(relation.get('subject', '')).lower().strip()
    predicate = str(relation.get('predicate', '')).lower().strip()
    obj = str(relation.get('object', '')).lower().strip()
    
    if subject and predicate and obj and len(subject) > 1 and len(obj) > 1:
        # Agregar nodos
        for entity, role in [(subject, 'subject'), (obj, 'object')]:
            if entity not in nodes:
                nodes[entity] = {
                    'id': entity,
                    'label': entity.title(),
                    'type': determine_node_type(entity, predicate, role),
                    'count': 1,
                    'size': 8,
                    'sentiment': extract_sentiment_from_name(entity) if role == 'object' else None
                }
            else:
                nodes[entity]['count'] += 1
        
        # Agregar enlace
        links.append({
            'source': subject,
            'target': obj,
            'relation': predicate,
            'label': format_relation_label(predicate),
            'type': determine_relation_type(predicate),
            'weight': 1,
            'sentiment': extract_sentiment_from_relation(predicate)
        })

def determine_node_type(name, predicate=None, role=None):
    """Determina el tipo semántico de un nodo"""
    name_lower = name.lower()
    
    # Productos conocidos
    if name_lower in known_product_names:
        return 'product'
    
    # Sentimientos
    if any(sent in name_lower for sent in ['positive', 'negative', 'neutral', 'good', 'bad', 'excellent', 'poor']):
        return 'sentiment'
    
    # Acciones/verbos
    if any(action in name_lower for action in ['buy', 'use', 'recommend', 'love', 'hate', 'get']):
        return 'action'
    
    # Características/features
    if any(feat in name_lower for feat in ['quality', 'price', 'design', 'feature', 'performance']):
        return 'feature'
    
    # Personas/usuarios
    if 'user' in name_lower or (predicate and 'person' in str(predicate).lower()):
        return 'user'
    
    return 'concept'

def extract_sentiment_from_name(name):
    """Extrae sentimiento del nombre de una entidad"""
    name_lower = name.lower()
    if any(pos in name_lower for pos in ['positive', 'good', 'excellent', 'great', 'love', 'amazing']):
        return 'positive'
    elif any(neg in name_lower for neg in ['negative', 'bad', 'poor', 'hate', 'terrible', 'awful']):
        return 'negative'
    elif 'neutral' in name_lower:
        return 'neutral'
    return None

def extract_sentiment_from_relation(predicate):
    """Extrae sentimiento de una relación/predicado"""
    pred_lower = predicate.lower()
    if any(pos in pred_lower for pos in ['positive_sentiment', 'love', 'recommend']):
        return 'positive'
    elif any(neg in pred_lower for neg in ['negative_sentiment', 'hate', 'not_recommend', 'problem']):
        return 'negative'
    elif 'neutral_sentiment' in pred_lower:
        return 'neutral'
    return None

def determine_relation_type(predicate):
    """Determina el tipo de relación"""
    pred_lower = predicate.lower()
    if 'sentiment' in pred_lower:
        return 'sentiment'
    elif any(rec in pred_lower for rec in ['recommend', 'suggest']):
        return 'recommendation'
    elif any(act in pred_lower for act in ['buy', 'use', 'get']):
        return 'action'
    elif 'problem' in pred_lower:
        return 'issue'
    return 'relation'

def format_relation_label(predicate):
    """Formatea etiquetas de relaciones para visualización"""
    return predicate.replace('_', ' ').replace('has ', '').title()

def calculate_node_importance(nodes, links):
    """Calcula métricas de importancia para los nodos"""
    # Contar conexiones
    connection_count = {}
    for link in links:
        source = link['source']
        target = link['target']
        connection_count[source] = connection_count.get(source, 0) + 1
        connection_count[target] = connection_count.get(target, 0) + 1
    
    # Actualizar tamaño basado en conexiones
    for node_id, node_data in nodes.items():
        connections = connection_count.get(node_id, 0)
        node_data['connections'] = connections
        node_data['size'] = max(node_data.get('size', 8), 8 + connections * 2)
        node_data['importance'] = node_data.get('count', 1) * (1 + connections * 0.5)

def filter_most_relevant_nodes(nodes, links, max_nodes=50):
    """Filtra los nodos más relevantes"""
    # Ordenar por importancia
    sorted_nodes = sorted(nodes.items(), key=lambda x: x[1].get('importance', 0), reverse=True)
    
    if len(sorted_nodes) <= max_nodes:
        return nodes, links
    
    # Mantener los nodos más importantes
    kept_node_ids = set([node_id for node_id, _ in sorted_nodes[:max_nodes]])
    
    # Filtrar nodos
    filtered_nodes = {node_id: node_data for node_id, node_data in nodes.items() if node_id in kept_node_ids}
    
    # Filtrar enlaces
    filtered_links = [link for link in links if link['source'] in kept_node_ids and link['target'] in kept_node_ids]
    
    return filtered_nodes, filtered_links

def generate_semantic_summary(nodes, links):
    """Genera un resumen semántico del grafo"""
    node_types = {}
    relation_types = {}
    sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    for node in nodes.values():
        node_type = node.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
        
        sentiment = node.get('sentiment')
        if sentiment in sentiments:
            sentiments[sentiment] += 1
    
    for link in links:
        rel_type = link.get('type', 'unknown')
        relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
    
    return {
        'node_types': node_types,
        'relation_types': relation_types,
        'sentiments': sentiments,
        'total_entities': len(nodes),
        'total_relations': len(links)
    }

def get_unique_node_types(nodes):
    """Obtiene tipos únicos de nodos"""
    return list(set([node.get('type', 'unknown') for node in nodes.values()]))

def get_unique_relation_types(links):
    """Obtiene tipos únicos de relaciones"""
    return list(set([link.get('type', 'unknown') for link in links]))

def create_fallback_graph(entities, verbs, query):
    """Crea un grafo de fallback cuando hay errores"""
    nodes = []
    links = []
    
    all_terms = entities + verbs
    for i, term in enumerate(all_terms[:10]):  # Limitar a 10 términos
        if term and len(term.strip()) > 1:
            nodes.append({
                'id': term.lower(),
                'label': term.title(),
                'type': 'query_term',
                'count': 1,
                'size': 10
            })
    
    return {
        'nodes': nodes,
        'links': links,
        'query': query,
        'total_nodes': len(nodes),
        'total_links': 0,
        'error': 'Modo de fallback activado'
    }

# 3. MODIFICAR LA RUTA get_graph_data PARA INCLUIR MÁS INFORMACIÓN
@app.route('/graph_view')
def graph_view():
    return render_template('graph_view.html')

@app.route('/get_graph_data', methods=['POST'])
def get_graph_data():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'No se proporcionó consulta'})
        
        # Procesar la consulta
        entities, verbs = process_query(query)
        detected_products = extract_known_products_from_query(query, known_product_names)
        if detected_products:
            entities += detected_products
        
        print(f"Procesando consulta para grafo: '{query}'")
        print(f"Entidades: {entities}")
        print(f"Verbos: {verbs}")
        
        # Consultar el grafo de conocimiento
        graph_raw_results, related_terms_from_graph = query_knowledge_graph_enhanced(entities, verbs)
        
        print(f"Resultados del grafo RDF: {len(graph_raw_results)} triples")
        print(f"Términos relacionados: {len(related_terms_from_graph)} términos")
        
        # Convertir a formato D3.js con información semántica enriquecida
        graph_data = convert_graph_results_to_d3(graph_raw_results, entities, verbs, query)
        
        print(f"Grafo final: {graph_data['total_nodes']} nodos, {graph_data['total_links']} enlaces")
        
        return jsonify(graph_data)
        
    except Exception as e:
        print(f"Error en get_graph_data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error interno del servidor: {str(e)}'})

if __name__ == '__main__':
    print("Iniciando aplicación Flask...")
    print(f"Directorio actual: {os.getcwd()}")

    print("Cargando datos y modelos...")
    load_data()
    print("Carga completa.")

    app.run(debug=True)