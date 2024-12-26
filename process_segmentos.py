# -*- coding: utf-8 -*-
import numpy as np
import pyproj
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, lit, udf
from pyspark.sql.types import ArrayType, StringType
from shapely import wkt
from shapely.ops import substring, transform

# Configurações estáticas
CONFIG = {
    "buffer_segmento_metros": 20,
    "comprimento_shape": 1000,
    "projecao_sirgas_2000": "EPSG:31983",
    "projecao_wgs_84": "EPSG:4326",
}

INPUT_TABLE = "ro-areatecnica.planejamento_staging.aux_shapes_geom_filtrada"
OUTPUT_TABLE = "ro-areatecnica.planejamento_staging.aux_segmento_shape_raw"


def transform_projection(shape, from_shapely=False):
    """
    Projeta um shape de uma CRS para outra CRS.
    """
    bq_projection = pyproj.CRS(CONFIG["projecao_wgs_84"])
    shapely_projection = pyproj.CRS(CONFIG["projecao_sirgas_2000"])

    project = pyproj.Transformer.from_crs(
        shapely_projection if from_shapely else bq_projection,
        bq_projection if from_shapely else shapely_projection,
        always_xy=True
    ).transform

    return transform(project, shape)


def cut(line, distance, buffer_size):
    """
    Corta a geometria em segmentos baseados na distância e adiciona buffers.
    """
    line_len = line.length
    dist_mod = line_len % distance
    dist_range = list(np.arange(0, line_len, distance))
    middle_index = (len(dist_range) // 2) + 1

    last_final_dist = 0
    lines = []

    for i, _ in enumerate(dist_range, start=1):
        if i == middle_index:
            cut_distance = dist_mod
        else:
            cut_distance = distance
        final_dist = last_final_dist + cut_distance
        segment = substring(line, last_final_dist, final_dist)
        lines.append(
            [
                str(i),
                transform_projection(segment, True).wkt,
                segment.length,
                transform_projection(segment.buffer(distance=buffer_size), True).wkt,
            ]
        )
        last_final_dist = final_dist

    return lines


def cut_udf(wkt_string, distance, buffer_size):
    """
    UDF para cortar geometria no PySpark.
    """
    line = transform_projection(wkt.loads(wkt_string))
    return cut(line, distance, buffer_size=buffer_size)


def main():
    # Inicializar sessão Spark
    spark = SparkSession.builder \
        .appName("BigQueryTest") \
        .config("spark.sql.catalogImplementation", "in-memory") \
        .config("spark.sql.catalog.bq", "com.google.cloud.spark.bigquery.v2.Spark35BigQueryTableProvider") \
        .getOrCreate()

    # Configurar UDF
    cut_udf_spark = udf(cut_udf, ArrayType(ArrayType(StringType())))

    # Ler dados da tabela de entrada no BigQuery
    df = spark.read.format("bigquery") \
        .option("table", INPUT_TABLE) \
        .load()

    # Aplicar transformações
    df_segments = df.withColumn(
        "shape_lists",
        cut_udf_spark(
            col("wkt_shape"),
            lit(CONFIG["comprimento_shape"]),
            lit(CONFIG["buffer_segmento_metros"]),
        ),
    )

    df_exploded = (
        df_segments.select(
            "feed_version",
            "feed_start_date",
            "feed_end_date",
            "shape_id",
            explode(col("shape_lists")).alias("shape_list"),
        )
        .withColumn("id_segmento", col("shape_list").getItem(0))
        .withColumn("wkt_segmento", col("shape_list").getItem(1))
        .withColumn("comprimento_segmento", col("shape_list").getItem(2))
        .withColumn("buffer_completo", col("shape_list").getItem(3))
        .drop("shape_list")
    )

    # Salvar resultado no BigQuery
    df_exploded.write.format("bigquery") \
        .option("table", OUTPUT_TABLE) \
        .option("writeMethod", "direct") \
        .mode("overwrite") \
        .save()

    print(f"Dados processados e salvos na tabela: {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()
