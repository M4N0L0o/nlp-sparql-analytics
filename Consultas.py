from rdflib import Graph, Namespace, Literal, URIRef, BNode
import sys
from gemini_sparql_generator import get_sparql_from_gemini

# BM25
from rank_bm25 import BM25Okapi
import re

# Cargar el archivo .ttl
g = Graph()
g.parse("data/reviews_knowledge_graph.ttl", format="turtle")  # Asegúrate de que el archivo existe en la ruta correcta

def tokenize(text):
    # Tokenización simple, minúsculas y sin puntuación
    if not text:
        return []
    return re.findall(r'\w+', text.lower())

def main():
    print("Escribe tu pregunta en lenguaje natural (o 'salir' para terminar):")
    while True:
        user_question = input("\nPregunta: ")
        if user_question.lower() == 'salir':
            print("Hasta pronto.")
            break

        # Genera la consulta SPARQL usando Gemini
        sparql_query = get_sparql_from_gemini(user_question)
        if not sparql_query:
            print("No se pudo generar la consulta SPARQL.")
            continue

        print("\n--- Consulta SPARQL generada ---\n")
        print(sparql_query)
        print("\n--- Ejecutando consulta... ---\n")

        try:
            results = g.query(sparql_query)
        except Exception as e:
            print(f"Error al ejecutar la consulta SPARQL: {e}")
            continue

        print("\n--- Resultados de la Consulta ---")
        if len(results) == 0:
            print("No se encontraron resultados para esta consulta.")
        else:
            result_variables = [str(v) for v in results.vars]
            print(f"Columnas disponibles: {', '.join(result_variables)}")

            # --- BM25: Prepara documentos y calcula scores ---
            docs = []
            rows = []
            for row in results:
                # Obtiene el texto de la reseña o string vacío si no existe
                review_text = getattr(row, 'reviewText', '') or ''
                docs.append(tokenize(str(review_text)))
                rows.append(row)

            # Tokeniza la pregunta del usuario
            query_tokens = tokenize(user_question)
            bm25 = BM25Okapi(docs)
            scores = bm25.get_scores(query_tokens)

            # Ordena los resultados por score BM25 descendente
            ranked = sorted(zip(scores, rows), key=lambda x: x[0], reverse=True)

            for i, (score, row) in enumerate(ranked):
                print(f"\n--- Resultado {i+1} (BM25: {score:.2f}) ---")
                if 'reviewNode' in result_variables:
                    review_node = row.reviewNode
                    print(f"  Nodo Reseña (BNode): {review_node}")
                if 'reviewTitle' in result_variables:
                    review_title = row.reviewTitle
                    print(f"  Título: {review_title if review_title else 'N/A'}")
                if 'reviewText' in result_variables:
                    review_text = row.reviewText
                    print(f"  Texto: {review_text if review_text is not None else 'N/A'}")
                if 'user' in result_variables:
                    user_uri = row.user
                    print(f"  Autor (URI): {user_uri.split('#')[-1] if isinstance(user_uri, URIRef) else user_uri}")
                if 'rating' in result_variables:
                    rating = row.rating
                    print(f"  Calificación: {rating if rating is not None else 'N/A'}")
                if 'productLabel' in result_variables:
                    product_label = row.productLabel
                    print(f"  Producto: {product_label if product_label else 'N/A'}")
                elif 'product' in result_variables:
                    product_uri = row.product
                    print(f"  Producto (URI): {product_uri.split('#')[-1] if isinstance(product_uri, URIRef) else product_uri}")
                for var_name in result_variables:
                    if var_name not in ['reviewNode', 'reviewTitle', 'reviewText', 'userName', 'user', 'rating', 'productLabel', 'product']:
                        value = getattr(row, var_name)
                        print(f"  {var_name}: {value if value is not None else 'N/A'}")

if __name__ == "__main__":
    main()