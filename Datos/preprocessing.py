import pandas as pd
import unicodedata
from pathlib import Path
from typing import List, Dict, Set
import os
import csv
import io

BASE_DIR = Path(__file__).resolve().parent

def normalize_text(text):
    """Remove accents, convert to lowercase, and normalize whitespace for ML consistency."""
    if not isinstance(text, str):
        return text
    # Convert to lowercase for ML consistency
    text = text.lower()
    # Remove accents
    nfd = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
    # Normalize spaces: collapse multiple spaces into one
    text = ' '.join(text.split())
    return text

def normalize_mapping(category_mapping: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Normalize all values in the category mapping by removing accents and normalizing spaces.
    
    Args:
        category_mapping: Dictionary with categories as keys and lists of occupation values
    
    Returns:
        Dictionary with normalized values
    """
    return {
        category: [normalize_text(value) for value in values]
        for category, values in category_mapping.items()
    }

def map_ocu_migr(value, category_mapping: Dict[str, List[str]], unmapped_values: Set[str]):
    """
    Map ocu_migr values to categories using category-to-values mapping.
    Assumes category_mapping values are already normalized.
    
    Args:
        value: The value to map
        category_mapping: Dictionary with categories as keys and lists of normalized values
        unmapped_values: Set to collect unmapped values
    
    Returns:
        Category name for the value, or original value if unmapped
    """
    if pd.isna(value):
        return value
    
    value_str = normalize_text(str(value))
    
    # Search for the value in all categories (values are already normalized)
    for category, values in category_mapping.items():
        if value_str in values:
            return category
    
    # If value not found, add to unmapped set and return original value
    unmapped_values.add(value_str)
    return value_str

def process_csv_file(input_file: str, output_file: str = None, chunk_size: int = 100000, category_mapping: Dict[str, List[str]] = None):
    """
    Process a CSV file by removing accents and standardizing column names to CAPS.
    Adds a new column 'OCU_MIGR_CLASS' with the mapped categories.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file (defaults to input_file_PROCESSED.csv)
        chunk_size: Number of rows to process at a time
        category_mapping: Dictionary mapping categories to lists of ocu_migr values
    
    Returns:
        Set of unmapped values found in this file
    """
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_PROCESSED.csv"
    
    print(f"Processing {input_file}...")
    
    processed_chunks = []
    total_rows = 0
    unmapped_values = set()
    
    # Try different encodings if failed to read
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    separator = ','  # Default separator
    
    if 'esi_2021.csv' in input_file:
        separator = ';'
    
    if 'ESI_2020' in input_file:
        df = rowFixer(BASE_DIR / 'ESI_2020.csv')
        df.to_csv(BASE_DIR / 'ESI_2020_fixed.csv', index=False, encoding='utf-8', sep=',',header=False)
        esi_file = BASE_DIR / 'ESI_2020_fixed.csv'
    else:
        esi_file = input_file
    
    # Read file in chunks
    final_encoding = 'utf-8'
    for encoding in encodings:
        try:
            chunk_reader = pd.read_csv(
                esi_file,
                chunksize=chunk_size,
                encoding=encoding,
                sep=separator,
                dtype=str 
            )
            print(f"  Reading with encoding: {encoding}")
            final_encoding = encoding
            break
        except UnicodeDecodeError:
            print(f"  Failed to read with encoding: {encoding}")
            continue
    
    for chunk in chunk_reader:
        # Standardize column names to lowercase and remove accents
        chunk.columns = [normalize_text(col) for col in chunk.columns]
        
        # Remove accents, convert to lowercase, and normalize whitespace from all columns
        for col in chunk.columns:
            chunk[col] = chunk[col].apply(normalize_text)
        
        # Map ocu_migr column
        chunk['ocu_class'] = chunk['ocu_migr'].apply(
            lambda x: map_ocu_migr(x, category_mapping, unmapped_values)
        )
        
        processed_chunks.append(chunk)
        total_rows += len(chunk)
        print(f"  Processed {total_rows} rows...")

    # Merge all chunks and write to output file
    if processed_chunks:
        processed_df = pd.concat(processed_chunks, ignore_index=True)
        processed_df.to_csv(output_file, index=False, encoding=final_encoding)
        print(f"✓ Successfully saved to {output_file}")
        print(f"  Total rows processed: {total_rows}")
    else:
        print(f"✗ No data found in {input_file}")

    
    return unmapped_values

def rowFixer(file):

    
    good_rows = []
    bad_rows = []
    fixed_bad_rows = []

    with open(file, encoding='latin-1', newline='') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')

        for _, row in enumerate(reader, start=1):
            if len(row) == 24:
                good_rows.append(row)
            else:
                bad_rows.append(row)

    goodDF = pd.DataFrame(good_rows)
    
    # fix each row of the bad rows, parsing as string and restructuring to 24 columns
    
    for row in bad_rows:
        parsed = next(csv.reader(
            io.StringIO(row[0]),
            delimiter=',',
            quotechar='"'
        ))
        
        fixed_bad_rows.append(parsed)
    
    fixed_bad_rows = pd.DataFrame(fixed_bad_rows)
    
    finalDF = pd.concat([goodDF, fixed_bad_rows], ignore_index=True)   

    #finalDF.to_csv(file.split('.')[0] + "_fixed.csv", index=False, encoding='latin1', sep=',',header=False)
    
    return finalDF
    
    


if __name__ == "__main__":
    # Define the category mapping - each category maps to the values it contains
    
    #Profesionales son los que requieren formacion universitaria o tecnica avanzada
    
    #No profesionales son los que no requieren formacion universitaria o tecnica avanzada
    
    #Menores de edad son los que no han alcanzado la mayoria de edad segun la legislacion vigente
    
    #Estudiantes son los que se encuentran actualmente matriculados en una institucion educativa
    
    #Sin especificar son los que no han proporcionado informacion sobre su ocupacion
    
    #Jubilados son los que han dejado de trabajar debido a la edad o por pension
    
    #Artesanos son los que se dedican a oficios manuales o artesanales que no requieren formacion universitaria o tecnica avanzada
    
    category_mapping = {
        "Profesionales": [
            "Ingenieros no clasificados bajo otros epigrafes", "Profesionales de enfermeria", "Desarrolladores de software",
            "Atletas y deportistas", "Profesores de formacion profesional", "Capitanes, oficiales de cubierta y practicos",
            "Psicologos", "Profesionales de la ensenanza no clasificados bajo otros epigrafes",
            "Directores generales y gerentes generales", "Tecnicos en ingenieria mecanica", "Medicos generales",
            "Matematicos, actuarios y estadisticos", "Pilotos de aviacion y afines", "Artistas de artes plasticas",
            "Analistas de gestion y organizacion", "Periodistas", "Agronomos y afines", "Contables",
            "Tecnicos en ciencias fisicas y en ingenieria no clasificados bajo otros epigrafes",
            "Profesionales del trabajo social", "Ingenieros industriales y de produccion",
            "Traductores, interpretes y linguistas", "Bailarines y coreografos",
            "Biologos, botanicos, zoologos y afines", "Sociologos, antropologos y afines", "Abogados",
            "Ingenieros de minas, metalurgicos y afines", "Economistas", "Geologos y geofisicos",
            "Musicos, cantantes y compositores", "Profesionales de ventas tecnicas y medicas (excluyendo la tic)",
            "Dentistas", "Chefs", "Ingenieros mecanicos", "Profesionales de la publicidad y la comercializacion",
            "Otros miembros de las fuerzas armadas", "Miembros del poder ejecutivo y de los gobiernos locales o seccionales",
            "Jueces", "Ingenieros electricistas", "Analistas financieros", "Ingenieros civiles", "Arquitectos",
            "Cartografos y agrimensores", "Artistas creativos e interpretativos no clasificados bajo otros epigrafes",
            "Farmaceuticos", "Profesionales de relaciones publicas",
            "Filosofos, historiadores y especialistas en ciencias politicas", "Autores y otros escritores",
            "Fotografos", "Profesionales de la proteccion medio ambiental", "Asesores financieros y en inversiones",
            "Profesionales de medicina tradicional y alternativa", "Fisicos y astronomos",
            "Disenadores y decoradores de interior", "Veterinarios", "Disenadores graficos y multimedia", "Actores",
            "Bibliotecarios, documentalistas y afines", "Entrenadores instructores y arbitros de actividades deportivas",
            "Ingenieros quimicos", "Medicos especialistas", "Personal directivo de la administracion publica",
            "Profesionales en derecho no clasificados bajo otros epigrafes", "Maestros preescolares",
            "Especialistas en politicas de administracion", "Tecnicos de radiodifusion y grabacion audio visual",
            "Profesionales de la salud y la higiene laboral y ambiental", "Meteorologos", "Quimicos",
            "Inspectores de la salud laboral, medioambiental y afines", "Ingenieros en telecomunicaciones",
            # --- NUEVOS AGREGADOS ---
            "Archivistas y curadores de museos",
            "Delineantes y dibujantes Tecnicos",
            "Locutores de radio, television y otros medios de comunicacion",
            "Operadores de instalaciones de refinacion de petroleo y gas natural",
            "Otros profesores de artes",
            "Otros profesores de idiomas",
            "Practicantes paramedicos",
            "Tecnicos de protesis medicas y dentales"
        ],
        "No profesionales": [
            "Amas de casa", "Limpiadores y asistentes domesticos", "Cuidadores de ninos",
            "Auxiliares laicos de las religiones", "Camareros de mesas", "Profesionales religiosos",
            "Conductores de automoviles, taxis y camionetas", "Guardianes de proteccion",
            "Mensajeros, mandaderos, maleteros y repartidores", "Secretarios (generales)", "Cocineros",
            "Peones de carga", "Pintores y empapeladores", "Panaderos, pasteleros y confiteros",
            "Bomberos", "Policias", "Cajeros de bancos y afines",
            # --- MOVIDOS DE "PROFESIONALES" POR CRITERIO ---
            "Limpiadores y asistentes de oficinas, hoteles y otros establecimientos", "Comerciantes de tiendas",
            "Auxiliares de servicios de abordo", "Representantes comerciales", "Criadores de ganado",
            "Agricultores y trabajadores calificados de huertas, invernaderos, viveros y jardines",
            "Asistentes de venta de tiendas y almacenes", "Oficinistas generales", "Electricistas de obras y afines",
            "Empacadores manuales", "Peluqueros", "Mineros y operadores de instalaciones mineras",
            "Pescadores, cazadores y tramperos", "Albaniles", "Marineros de cubierta y afines",
            "Agricultores y trabajadores calificados de jardines y de cultivos para el mercado",
            "Guias de turismo", "Especialistas en tratamiento de belleza y afines",
            "Trabajadores de explotacion de acuicultura", "Inspectores de policia y detectives",
            "Cobradores y afines", "Agentes de servicios comerciales no clasificados bajo otros epigrafes",
            "Productores y trabajadores calificados de explotaciones agropecuarias mixtas cuya produccion se destina al mercado",
            "Oficiales y operarios de la construccion (obra gruesa) y afines no clasificados bajo otros epigrafes",
            # --- NUEVOS AGREGADOS ---
            "Apicultores y sericultores y trabajadores calificados de la apicultura y la sericultura",
            "Astrologos, adivinadores y afines",
            "Avicultores y trabajadores calificados de la avicultura",
            "Catadores y clasificadores de alimentos y bebidas",
            "Conserjes",
            "Economos y mayordomos domesticos",
            "Fontaneros e instaladores de tuberias",
            "Grabadores de datos",
            "Lavanderos y planchadores manuales",
            "Mecanicos y reparadores de vehiculos de motor",
            "Modelos de moda, arte y publicidad",
            "Operadores de maquinas para fabricar productos de caucho",
            "Peones de explotacion de cultivos mixtos y ganaderos",
            "Personal de apoyo administrativo no clasificado bajo otros epigrafes",
            "Pescadores, cazadores y tramperosPescadores, cazadores y tramperos",
        ],
        "Menores de edad": ["Menores de edad"],
        "Estudiantes": ["Estudiantes"],
        "Sin especificar": ["Sin especificar"],
        "Jubilados": ["Jubilados y pensionistas"],
        "Artesanos": [
            "Oficiales, operarios y artesanos de artes mecanicas y de otros oficios no clasificados bajo otros epigrafes",
            "Carpinteros de armar y de obra blanca", "Artesanos no clasificados bajo otros epigrafes",
            "Alfareros y afines (barro, arcilla y abrasivos)", "Ebanistas y afines",
            "Artesanos de los tejidos, el cuero y materiales similares", "Costureros, bordadores y afines",
            # --- NUEVOS AGREGADOS ---
            "Chapistas y caldereros",
            "Joyeros, orfebres y plateros",
            "Sastres, modistos, peleteros y sombrereros",
            "Tapiceros, colchoneros y afines",
            "Zapateros y afines",
        ]
    }
    
    category_mapping = normalize_mapping(category_mapping)

    
    # Define the files to process
    current_dir = Path(__file__).parent
    files_to_process = [
        current_dir / "esi_2018.csv",
        current_dir / "ESI_2019.csv",
        current_dir / "ESI_2020.csv",
        current_dir / "esi_2021.csv",
        current_dir / "esi2023.csv",
        current_dir / "esi_2024.csv",
    ]
    
    # Convert to string paths that exist
    valid_files = [str(f) for f in files_to_process if f.exists()]
    
    for file in valid_files:
        unmapped_values = process_csv_file(
            input_file=file,
            category_mapping=category_mapping
        )
        print("\nUnmapped values:")
        for value in unmapped_values:
            print(value)





