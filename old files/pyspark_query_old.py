import os
import sys
import argparse
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException

# Environment variables (override with docker run -e or your .env file)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
BUCKET_NAME = os.getenv("BUCKET_NAME", "isic-parquet-p")
DELTA_PATH_PREFIX = os.getenv("DELTA_PATH_PREFIX", "delta")
DELTA_METADATA_PREFIX = f"{DELTA_PATH_PREFIX}/delta_tables"

def init_spark() -> SparkSession:
    """
    Initialize the SparkSession with MinIO (S3) support and Delta support.
    """
    print(f"Connecting to MinIO at endpoint: {MINIO_ENDPOINT}")
    print(f"Using bucket: {BUCKET_NAME}")
    print(f"Delta metadata path: {DELTA_METADATA_PREFIX}")
    
    spark = (
        SparkSession.builder.appName("PySparkMinIOExample")
        # S3A Configuration
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        # Additional S3A settings
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "5000")
        .config("spark.hadoop.fs.s3a.connection.timeout", "10000")
        .config("spark.hadoop.fs.s3a.committer.name", "directory")
        # Delta configuration
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

def verify_path_exists(spark: SparkSession, path: str) -> bool:
    """Check if a path exists in S3/MinIO"""
    try:
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark._jvm.java.net.URI(path),
            spark._jsc.hadoopConfiguration()
        )
        return fs.exists(spark._jvm.org.apache.hadoop.fs.Path(path))
    except Exception as e:
        print(f"Error verifying path {path}: {e}")
        return False

def list_path_contents(spark: SparkSession, path: str) -> List[str]:
    """List contents of a directory in S3/MinIO"""
    try:
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark._jvm.java.net.URI(path),
            spark._jsc.hadoopConfiguration()
        )
        return [f.getPath().getName() for f in fs.listStatus(spark._jvm.org.apache.hadoop.fs.Path(path))]
    except Exception as e:
        print(f"Error listing contents of {path}: {e}")
        return []

def test_s3a_connection(spark: SparkSession) -> bool:
    """Test if we can connect to S3A and list the bucket contents."""
    try:
        print("Testing S3A connection...")
        uri = f"s3a://{BUCKET_NAME}"
        
        # List top-level directories
        print(f"Listing contents of {uri}")
        contents = list_path_contents(spark, uri)
        print(f"Found {len(contents)} items in bucket root:")
        for item in contents:
            print(f"  - {item}")
            
        # List delta directory
        delta_path = f"s3a://{BUCKET_NAME}/{DELTA_PATH_PREFIX}"
        print(f"Listing contents of {delta_path}")
        delta_contents = list_path_contents(spark, delta_path)
        print(f"Found {len(delta_contents)} items in delta directory:")
        for item in delta_contents:
            print(f"  - {item}")
            
        return True
    except Exception as e:
        print(f"Error testing S3A connection: {e}")
        return False

def list_collections(spark: SparkSession) -> List[str]:
    """List all collection IDs by scanning the metadata directory."""
    if not test_s3a_connection(spark):
        print("Failed to connect to S3/MinIO - check your connection settings")
        return []
    
    metadata_path = f"s3a://{BUCKET_NAME}/{DELTA_METADATA_PREFIX}"
    print(f"Attempting to list collections in {metadata_path}")
    
    # First try direct listing
    try:
        contents = list_path_contents(spark, metadata_path)
        collection_ids = [
            path.replace("collection_", "") 
            for path in contents 
            if path.startswith("collection_")
        ]
        if collection_ids:
            return sorted(collection_ids, key=lambda x: int(x))
    except Exception as e:
        print(f"Error listing collections with Hadoop API: {e}")
    
    # Fallback to Delta/Parquet pattern matching
    path_pattern = f"s3a://{BUCKET_NAME}/{DELTA_METADATA_PREFIX}/collection_*"
    try:
        print(f"Attempting to detect collections with pattern: {path_pattern}")
        
        # Try Delta first
        try:
            df = spark.read.format("delta").load(path_pattern)
            files = df.select(F.input_file_name().alias("filename")).distinct().collect()
        except AnalysisException:
            # Fall back to Parquet if Delta fails
            df = spark.read.format("parquet").load(path_pattern)
            files = df.select(F.input_file_name().alias("filename")).distinct().collect()
        
        collection_ids = set()
        for row in files:
            file_path = row["filename"]
            parts = file_path.split("collection_")
            if len(parts) > 1:
                coll_id = parts[1].split("/")[0]
                collection_ids.add(coll_id)
                
        return sorted(list(collection_ids), key=lambda x: int(x)) if collection_ids else []
    except Exception as e:
        print(f"Error detecting collections: {e}")
        return []

def read_data(spark: SparkSession, path: str) -> Optional[DataFrame]:
    """Try to read data in Delta or Parquet format"""
    # First try Delta format
    try:
        print(f"Attempting to read as Delta table: {path}")
        return spark.read.format("delta").load(path)
    except AnalysisException as delta_err:
        print(f"Delta read failed: {delta_err}")
        
        # Fall back to Parquet
        try:
            print(f"Attempting to read as Parquet: {path}")
            return spark.read.parquet(path)
        except Exception as parquet_err:
            print(f"Parquet read failed: {parquet_err}")
            return None

def list_columns(spark: SparkSession, collection_id: Optional[str] = None) -> List[str]:
    """Return a list of columns for either a single collection or across all collections."""
    if collection_id:
        path = f"s3a://{BUCKET_NAME}/{DELTA_METADATA_PREFIX}/collection_{collection_id}"
        print(f"Checking collection at path: {path}")
        
        if not verify_path_exists(spark, path):
            print(f"Path does not exist: {path}")
            return []
            
        # Try to read the data
        df = read_data(spark, path)
        return df.columns if df else []
    else:
        cols = set()
        cids = list_collections(spark)
        for coll in cids:
            path = f"s3a://{BUCKET_NAME}/{DELTA_METADATA_PREFIX}/collection_{coll}"
            df = read_data(spark, path)
            if df:
                cols.update(df.columns)
        return sorted(cols)

def print_collections(collections: List[str]):
    """Print collections in a formatted way"""
    if collections:
        print("Collections found:")
        for idx, cid in enumerate(collections, start=1):
            print(f"{idx}. {cid}")
    else:
        print("No collections found.")

def print_columns(columns: List[str], collection_id: Optional[str] = None):
    """Print columns in a formatted way"""
    if columns:
        prefix = f"Columns in collection {collection_id}:" if collection_id else "Columns across all collections:"
        print(prefix)
        for col in columns:
            print(f"- {col}")
    else:
        prefix = f"No columns found for collection {collection_id}." if collection_id else "No columns found across all collections."
        print(prefix)

def main():
    parser = argparse.ArgumentParser(description="PySpark + Delta + MinIO Query Tool")
    parser.add_argument("--list-collections", action="store_true", help="List all collections")
    parser.add_argument("--list-columns", action="store_true", help="List columns (use --collection to specify one)")
    parser.add_argument("--collection", type=str, help="Specific collection ID to query")
    parser.add_argument("--column", type=str, help="Column name to query")
    parser.add_argument("--value", type=str, help="Value to search for in a column")
    parser.add_argument("--distinct", action="store_true", help="Return distinct values for a column")
    parser.add_argument("--sql", type=str, help="Run a custom Spark SQL query")
    parser.add_argument("--limit", type=int, default=20, help="Limit rows in console display")
    parser.add_argument("--output-csv", type=str, help="Output CSV filename for results")
    parser.add_argument("--test-connection", action="store_true", help="Test S3/MinIO connection only")

    args = parser.parse_args()

    spark = init_spark()

    try:
        if args.test_connection:
            test_s3a_connection(spark)
            return
            
        if args.list_collections:
            collections = list_collections(spark)
            print_collections(collections)
        
        elif args.list_columns:
            columns = list_columns(spark, args.collection)
            print_columns(columns, args.collection)

        # Add handling for other commands here...  .

    finally:
        spark.stop()

if __name__ == "__main__":
    main()