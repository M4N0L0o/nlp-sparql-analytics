from flask import Flask, render_template, request, jsonify
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from gemini_sparql_generator import get_sparql_from_gemini

app = Flask(__name__)

# Cargar el grafo RDF una sola vez al iniciar la aplicación
g = Graph()
try:
    g.parse("data/reviews_knowledge_graph.ttl", format="turtle")
    print(f"Grafo cargado exitosamente con {len(g)} tripletas")
except Exception as e:
    print(f"Error cargando el grafo: {e}")
    g = None

def execute_sparql_query(sparql_query):
    """
    Ejecuta una consulta SPARQL sobre el grafo y retorna los resultados formateados
    Replica la lógica de presentación de Consultas.py
    """
    if not g:
        return {"error": "Grafo no disponible"}
    
    try:
        results = g.query(sparql_query)
        
        if len(results) == 0:
            return {"message": "No se encontraron resultados para esta consulta.", "results": []}
        
        # Obtener las variables seleccionadas (igual que en Consultas.py)
        result_variables = [str(v) for v in results.vars]
        formatted_results = []
        
        for i, row in enumerate(results):
            result_item = {
                "index": i + 1,
                "raw_data": {}  # Para datos adicionales si es necesario
            }
            
            # Procesar reviewNode si existe (pero no mostrarlo en resultados)
            if 'reviewNode' in result_variables:
                review_node = row.reviewNode
                result_item["raw_data"]["reviewNode"] = review_node

            # Procesar reviewTitle
            if 'reviewTitle' in result_variables:
                review_title = row.reviewTitle
                result_item["reviewTitle"] = str(review_title) if review_title else "N/A"

            # Procesar reviewText
            if 'reviewText' in result_variables:
                review_text = row.reviewText
                result_item["reviewText"] = str(review_text) if review_text is not None else "N/A"

            # Procesar autor - SOLO mostrar user (URI), no reviewAuthorName
            if 'user' in result_variables:
                user_uri = row.user
                if isinstance(user_uri, URIRef):
                    author_display = user_uri.split('#')[-1] if '#' in str(user_uri) else str(user_uri)
                else:
                    author_display = str(user_uri)
                result_item["user"] = f" {author_display}"
                result_item["raw_data"]["user"] = user_uri

            # Procesar rating
            if 'rating' in result_variables:
                rating = row.rating
                result_item["rating"] = str(rating) if rating is not None else "N/A"

            # Procesar producto
            if 'productLabel' in result_variables:
                product_label = row.productLabel
                result_item["productLabel"] = str(product_label) if product_label else "N/A"
            elif 'product' in result_variables:
                product_uri = row.product
                if isinstance(product_uri, URIRef):
                    product_display = product_uri.split('#')[-1] if '#' in str(product_uri) else str(product_uri)
                else:
                    product_display = str(product_uri)
                result_item["product"] = f"Producto (URI): {product_display}"
                result_item["raw_data"]["product"] = product_uri

            # Procesar otras variables que no son las estándar de reseña
            for var_name in result_variables:
                if var_name not in ['reviewNode', 'reviewTitle', 'reviewText', 'reviewAuthorName', 'user', 'rating', 'productLabel', 'product']:
                    value = getattr(row, var_name, None)
                    if value is not None:
                        if isinstance(value, URIRef):
                            formatted_value = str(value).split('#')[-1] if '#' in str(value) else str(value)
                        else:
                            formatted_value = str(value)
                        result_item[var_name] = formatted_value
                    else:
                        result_item[var_name] = "N/A"
            
            formatted_results.append(result_item)
        
        return {
            "variables": result_variables,
            "results": formatted_results,
            "total": len(formatted_results),
            "raw_query": sparql_query
        }
        
    except Exception as e:
        return {"error": f"Error ejecutando la consulta SPARQL: {str(e)}"}

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def process_query():
    """
    Endpoint para procesar consultas en lenguaje natural
    """
    data = request.get_json()
    user_query = data.get('query', '').strip()
    
    if not user_query:
        return jsonify({"error": "La consulta no puede estar vacía"})
    
    # Generar consulta SPARQL usando Gemini
    sparql_query = get_sparql_from_gemini(user_query)
    
    if not sparql_query:
        return jsonify({"error": "No se pudo generar la consulta SPARQL"})
    
    # Ejecutar la consulta SPARQL
    results = execute_sparql_query(sparql_query)
    
    return jsonify({
        "user_query": user_query,
        "sparql_query": sparql_query,
        "results": results
    })

@app.route('/direct-sparql', methods=['POST'])
def direct_sparql():
    """
    Endpoint para ejecutar consultas SPARQL directamente
    """
    data = request.get_json()
    sparql_query = data.get('sparql_query', '').strip()
    
    if not sparql_query:
        return jsonify({"error": "La consulta SPARQL no puede estar vacía"})
    
    # Ejecutar la consulta SPARQL directamente
    results = execute_sparql_query(sparql_query)
    
    return jsonify({
        "sparql_query": sparql_query,
        "results": results
    })

@app.route('/health')
def health_check():
    """Endpoint para verificar el estado de la aplicación"""
    status = {
        "status": "OK",
        "graph_loaded": g is not None,
        "graph_size": len(g) if g else 0
    }
    return jsonify(status)

# Agregar estas rutas a tu app.py existente

@app.route('/graph')
def graph_visualizer():
    """Página del visualizador de grafo"""
    # Necesitarás crear un template graph.html o servir el HTML directamente
    # Por simplicidad, puedes crear el archivo graph.html en tu carpeta templates
    return render_template('graph.html')

@app.route('/graph-data')
def get_graph_data():
    """
    Endpoint para obtener los datos del grafo en formato JSON para D3.js
    """
    if not g:
        return jsonify({"error": "Grafo no disponible"})
    
    try:
        # Obtener todos los nodos únicos (sujetos y objetos)
        nodes_set = set()
        links = []
        
        # Procesar todas las tripletas del grafo
        for subj, pred, obj in g:
            subj_str = str(subj)
            obj_str = str(obj)
            pred_str = str(pred)
            
            # Agregar nodos
            nodes_set.add(subj_str)
            if not isinstance(obj, Literal):  # Solo agregar objetos que son URIs, no literales
                nodes_set.add(obj_str)
            
            # Crear enlace
            if not isinstance(obj, Literal):
                # Extraer nombre del predicado para mostrar
                pred_label = pred_str.split('#')[-1] if '#' in pred_str else pred_str.split('/')[-1]
                
                links.append({
                    "source": subj_str,
                    "target": obj_str,
                    "predicate": pred_label,
                    "full_predicate": pred_str
                })
        
        # Convertir set de nodos a lista de objetos
        nodes = [{"id": node_id} for node_id in nodes_set]
        
        # Limitar el número de nodos para mejor rendimiento (opcional)
        max_nodes = 200
        if len(nodes) > max_nodes:
            # Tomar una muestra de los nodos más conectados
            node_connections = {}
            for link in links:
                node_connections[link["source"]] = node_connections.get(link["source"], 0) + 1
                node_connections[link["target"]] = node_connections.get(link["target"], 0) + 1
            
            # Ordenar nodos por número de conexiones
            sorted_nodes = sorted(nodes, key=lambda x: node_connections.get(x["id"], 0), reverse=True)
            selected_nodes = sorted_nodes[:max_nodes]
            selected_node_ids = {node["id"] for node in selected_nodes}
            
            # Filtrar enlaces para incluir solo los nodos seleccionados
            filtered_links = [
                link for link in links 
                if link["source"] in selected_node_ids and link["target"] in selected_node_ids
            ]
            
            return jsonify({
                "nodes": selected_nodes,
                "links": filtered_links,
                "total_nodes": len(nodes),
                "total_links": len(links),
                "limited": True
            })
        
        return jsonify({
            "nodes": nodes,
            "links": links,
            "total_nodes": len(nodes),
            "total_links": len(links),
            "limited": False
        })
        
    except Exception as e:
        return jsonify({"error": f"Error obteniendo datos del grafo: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)