import os
import argparse
from typing import List, Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException

from pyspark.sql.functions import broadcast
from multiprocessing.pool import ThreadPool
import concurrent.futures

from dotenv import load_dotenv

load_dotenv()

# Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
BUCKET_NAME = os.getenv("BUCKET_NAME", "isic-parquet-p")
DELTA_PATH_PREFIX = os.getenv("DELTA_PATH_PREFIX", "delta")
DELTA_TABLES_PATH = f"s3a://{BUCKET_NAME}/{DELTA_PATH_PREFIX}/delta_tables"

def init_spark() -> SparkSession:
    """Initialize Spark session with MinIO and Delta Lake support"""
    spark = (
        SparkSession.builder.appName("ISICDeltaQuery")
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.connection.timeout", "10000")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.parquet.mergeSchema", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

def test_minio_connection(spark: SparkSession) -> bool:
    """Test connection to MinIO"""
    try:
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark._jvm.java.net.URI(f"s3a://{BUCKET_NAME}"),
            spark._jsc.hadoopConfiguration()
        )
        return fs.exists(spark._jvm.org.apache.hadoop.fs.Path(DELTA_TABLES_PATH))
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False

def list_collections(spark: SparkSession) -> List[str]:
    """List all available collections"""
    try:
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark._jvm.java.net.URI(DELTA_TABLES_PATH),
            spark._jsc.hadoopConfiguration()
        )
        collections = [
            f.getPath().getName().replace("collection_", "")
            for f in fs.listStatus(spark._jvm.org.apache.hadoop.fs.Path(DELTA_TABLES_PATH))
            if f.isDirectory() and f.getPath().getName().startswith("collection_")
        ]
        return sorted(collections, key=int)
    except Exception as e:
        print(f"Error listing collections: {e}")
        return []

def get_collection_data(spark: SparkSession, collection_id: str) -> Optional[DataFrame]:
    """Load data for a specific collection"""
    path = f"{DELTA_TABLES_PATH}/collection_{collection_id}"
    try:
        return spark.read.format("delta").load(path)
    except Exception as e:
        print(f"Error loading collection {collection_id}: {e}")
        return None

def list_columns(df: DataFrame) -> List[str]:
    """Get column names from a DataFrame"""
    return df.columns if df else []

def run_sql_query(spark: SparkSession, query: str) -> Optional[DataFrame]:
    """Execute a SQL query on the Delta tables"""
    try:
        # Register all collections as a temporary view
        spark.read.format("delta").load(f"{DELTA_TABLES_PATH}/*").createOrReplaceTempView("isic_data")
        return spark.sql(query)
    except Exception as e:
        print(f"SQL query failed: {e}")
        return None

def export_to_csv(df: DataFrame, output_path: str):
    """Export DataFrame to CSV"""
    try:
        df.write.csv(output_path, header=True, mode="overwrite")
        print(f"Data exported to {output_path}")
    except Exception as e:
        print(f"Export failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="ISIC Archive Delta Lake Query Tool")
    parser.add_argument("--list-collections", action="store_true", help="List available collections")
    parser.add_argument("--list-columns", action="store_true", help="List columns in a collection")
    parser.add_argument("--collection", type=str, help="Collection ID to query")
    parser.add_argument("--column", type=str, help="Column name to filter or display")
    parser.add_argument("--value", type=str, help="Value to filter by in the specified column")
    parser.add_argument("--distinct", action="store_true", help="Show distinct values for a column")
    parser.add_argument("--sql", type=str, help="SQL query to execute")
    parser.add_argument("--limit", type=int, default=20, help="Number of rows to display")
    parser.add_argument("--output-csv", type=str, help="Path to export results as CSV")
    parser.add_argument("--test-connection", action="store_true", help="Test MinIO connection")

    args = parser.parse_args()
    spark = init_spark()

    try:
        if args.test_connection:
            if test_minio_connection(spark):
                print("Connection successful")
            else:
                print("Connection failed")
            return

        if args.list_collections:
            collections = list_collections(spark)
            if collections:
                print("Available collections:")
                for i, coll_id in enumerate(collections, 1):
                    print(f"{i}. {coll_id}")
            else:
                print("No collections found")
            return

        if args.sql:
            df = run_sql_query(spark, args.sql)
            if df:
                if args.output_csv:
                    export_to_csv(df, args.output_csv)
                else:
                    df.show(args.limit, truncate=False)
            return

        if args.collection:
            df = get_collection_data(spark, args.collection)
            if not df:
                return

            if args.list_columns:
                columns = list_columns(df)
                if columns:
                    print(f"Columns in collection {args.collection}:")
                    for col in columns:
                        print(f"- {col}")
                return

            if args.column:
                if args.distinct:
                    df.select(args.column).distinct().show(args.limit, truncate=False)
                elif args.value:
                    df.filter(F.col(args.column) == args.value).show(args.limit, truncate=False)
                else:
                    df.select(args.column).show(args.limit, truncate=False)
            else:
                df.show(args.limit)

    finally:
        spark.stop()

if __name__ == "__main__":
    main()