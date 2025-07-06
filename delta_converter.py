import os
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException
from dotenv import load_dotenv

load_dotenv()

def init_spark():
    print("Initializing Spark with enhanced configuration...")
    spark = (
        SparkSession.builder.appName("ISICDeltaConverter")
        .config("spark.hadoop.fs.s3a.endpoint", os.getenv("MINIO_ENDPOINT"))
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("MINIO_ACCESS_KEY"))
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("MINIO_SECRET_KEY"))
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.connection.timeout", "10000")
        .config("spark.hadoop.fs.s3a.attempts.maximum", "5")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "5000")
        .config("spark.sql.parquet.mergeSchema", "true")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "2g")  # Added memory config
        .config("spark.executor.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

def test_minio_connection(spark):
    print("Testing MinIO connection with timeout protection...")
    try:
        # Use direct Java calls with timeout protection
        conf = spark.sparkContext._jsc.hadoopConfiguration()
        fs = spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark.sparkContext._jvm.java.net.URI("s3a://" + os.getenv("BUCKET_NAME")),
            conf
        )
        status = fs.listStatus(spark.sparkContext._jvm.org.apache.hadoop.fs.Path("s3a://" + os.getenv("BUCKET_NAME")))
        print(f"Connection successful. Found {len(status)} items in bucket root")
        return True
    except Exception as e:
        print(f"MINIO CONNECTION FAILED: {str(e)}")
        return False

def convert_collections_to_delta(spark):
    base_path = f"s3a://{os.getenv('BUCKET_NAME')}/{os.getenv('DELTA_PATH_PREFIX')}"
    collections = list_collections(spark)  # Reuse your working collection lister
    
    for coll_id in collections:  # Test with first 5 collections
        try:
            input_path = f"{base_path}/metadata/collection_{coll_id}/latest.parquet"
            output_path = f"{base_path}/delta_tables/collection_{coll_id}"
            
            print(f"\nProcessing collection {coll_id}...")
            print(f"Reading from: {input_path}")
            
            # Read with explicit schema handling
            df = spark.read.option("mergeSchema", "true").parquet(input_path)
            print(f"Read {df.count()} rows with {len(df.columns)} columns")
            
            print(f"Writing to: {output_path}")
            df.write.format("delta").mode("overwrite").save(output_path)
            
            print(f"Successfully converted collection {coll_id}")
        except Exception as e:
            print(f"Failed to process collection {coll_id}: {str(e)}")
            continue

def list_collections(spark):
    """Reuse your working collection listing logic from pyspark_query.py"""
    metadata_path = f"s3a://{os.getenv('BUCKET_NAME')}/{os.getenv('DELTA_PATH_PREFIX')}/metadata"
    contents = spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(
        spark.sparkContext._jvm.java.net.URI(metadata_path),
        spark.sparkContext._jsc.hadoopConfiguration()
    ).listStatus(spark.sparkContext._jvm.org.apache.hadoop.fs.Path(metadata_path))
    
    return sorted([
        f.getPath().getName().replace("collection_", "") 
        for f in contents 
        if f.getPath().getName().startswith("collection_")
    ], key=int)

if __name__ == "__main__":
    print("Starting Delta conversion process...")
    spark = None
    try:
        spark = init_spark()
        if test_minio_connection(spark):
            convert_collections_to_delta(spark)
        else:
            print("Cannot proceed without MinIO connection")
    except Exception as e:
        print(f"FATAL ERROR: {str(e)}")
    finally:
        if spark:
            spark.stop()
        print("Process completed")