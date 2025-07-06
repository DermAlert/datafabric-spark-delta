import os
import sys
import argparse
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Environment variables (override with docker run -e or your .env file)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
BUCKET_NAME = os.getenv("BUCKET_NAME", "isic-parquet-p")
DELTA_PATH_PREFIX = os.getenv("DELTA_PATH_PREFIX", "delta")
DELTA_METADATA_PREFIX = f"{DELTA_PATH_PREFIX}/metadata"

def init_spark():
    """
    Initialize the SparkSession with MinIO (S3) support and Delta support.
    """
    # Print connection details for debugging
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
        # Additional S3A settings that might help
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "5000")
        .config("spark.hadoop.fs.s3a.connection.timeout", "10000")
        # Make listing work better with MinIO
        .config("spark.hadoop.fs.s3a.committer.name", "directory")
        # Delta configuration
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

def test_s3a_connection(spark):
    """
    Test if we can connect to S3A and list the bucket contents.
    This is helpful for debugging connection issues.
    """
    try:
        print("Testing S3A connection...")
        # Test if we can access the bucket directly
        uri = f"s3a://{BUCKET_NAME}"
        hadoop_conf = spark._jsc.hadoopConfiguration()
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark._jvm.java.net.URI(uri),
            hadoop_conf
        )
        
        # List top-level directories in the bucket
        print(f"Listing contents of {uri}")
        status = fs.listStatus(spark._jvm.org.apache.hadoop.fs.Path(uri))
        print(f"Found {len(status)} items in bucket root:")
        for s in status:
            path = s.getPath().getName()
            print(f"  - {path}")
            
        # Try to list the delta directory specifically
        delta_path = f"s3a://{BUCKET_NAME}/{DELTA_PATH_PREFIX}"
        print(f"Listing contents of {delta_path}")
        delta_status = fs.listStatus(spark._jvm.org.apache.hadoop.fs.Path(delta_path))
        print(f"Found {len(delta_status)} items in delta directory:")
        for s in delta_status:
            path = s.getPath().getName()
            print(f"  - {path}")
            
        return True
    except Exception as e:
        print(f"Error testing S3A connection: {e}")
        return False

def list_collections(spark):
    """
    List all collection IDs by scanning the s3 path for each 'collection_X' folder.
    Returns a list of string collection IDs.
    """
    # First test if we can access S3 at all
    if not test_s3a_connection(spark):
        print("Failed to connect to S3/MinIO - check your connection settings")
        return []
    
    # Try to list collections directly using the Hadoop API first
    try:
        metadata_path = f"s3a://{BUCKET_NAME}/{DELTA_METADATA_PREFIX}"
        print(f"Attempting to list collections in {metadata_path}")
        
        hadoop_conf = spark._jsc.hadoopConfiguration()
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark._jvm.java.net.URI(metadata_path),
            hadoop_conf
        )
        
        # List directories matching collection_* pattern
        status = fs.listStatus(spark._jvm.org.apache.hadoop.fs.Path(metadata_path))
        collection_ids = []
        
        print(f"Found {len(status)} items in metadata directory:")
        for s in status:
            path = s.getPath().getName()
            print(f"  - {path}")
            if path.startswith("collection_"):
                coll_id = path.replace("collection_", "")
                collection_ids.append(coll_id)
                
        if collection_ids:
            return sorted(collection_ids, key=lambda x: int(x))
    except Exception as e:
        print(f"Error listing collections with Hadoop API: {e}")
    
    # Original method using Spark Delta
    path_pattern = f"s3a://{BUCKET_NAME}/{DELTA_METADATA_PREFIX}/collection_*/latest.parquet"
    try:
        print(f"Attempting to read Delta tables with pattern: {path_pattern}")
        df = spark.read.format("delta").load(path_pattern)
        df_files = df.withColumn("filename", F.input_file_name()).select("filename").distinct()
        files = df_files.collect()

        collection_ids = set()
        for row in files:
            file_path = row["filename"]
            part = file_path.split("collection_")[-1]
            if "/" in part:
                coll_id = part.split("/")[0]
                collection_ids.add(coll_id)

        return sorted(list(collection_ids), key=lambda x: int(x)) if collection_ids else []
    except Exception as e:
        print(f"Error listing collections with Delta: {e}")
        
        # Try one more approach: just list files without using Delta
        try:
            print("Trying to list parquet files directly without Delta...")
            direct_path = f"s3a://{BUCKET_NAME}/{DELTA_METADATA_PREFIX}/collection_*/latest.parquet"
            df = spark.read.format("parquet").option("recursiveFileLookup", "true").load(direct_path)
            print(f"Found {df.count()} rows in parquet files")
            return []  # Still return empty since we're just testing connectivity
        except Exception as e2:
            print(f"Error listing parquet files: {e2}")
            return []

def list_columns(spark, collection_id=None):
    """
    Return a list of columns for either a single collection or across all collections.
    """
    if collection_id:
        path = f"s3a://{BUCKET_NAME}/{DELTA_METADATA_PREFIX}/collection_{collection_id}/latest.parquet"
        try:
            df = spark.read.format("delta").load(path)
            return df.columns
        except Exception as e:
            print(f"Error reading collection {collection_id}: {e}")
            return []
    else:
        cols = set()
        cids = list_collections(spark)
        for coll in cids:
            path = f"s3a://{BUCKET_NAME}/{DELTA_METADATA_PREFIX}/collection_{coll}/latest.parquet"
            try:
                df = spark.read.format("delta").load(path)
                for col in df.columns:
                    cols.add(col)
            except Exception as e:
                print(f"Error reading collection {coll}: {e}")
        return sorted(cols)

# [rest of functions unchanged]

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
            cids = list_collections(spark)
            if cids:
                print("Collections found:")
                for idx, cid in enumerate(cids, start=1):
                    print(f"{idx}. {cid}")
            else:
                print("No collections found.")
        
        elif args.list_columns:
            if args.collection:
                columns = list_columns(spark, args.collection)
                if columns:
                    print(f"Columns in collection {args.collection}:")
                    for col in columns:
                        print(f"- {col}")
                else:
                    print(f"No columns found for collection {args.collection}.")
            else:
                columns = list_columns(spark)
                if columns:
                    print("Columns across all collections:")
                    for col in columns:
                        print(f"- {col}")
                else:
                    print("No columns found across all collections.")

        # [rest of main function unchanged]

    finally:
        spark.stop()

if __name__ == "__main__":
    main()